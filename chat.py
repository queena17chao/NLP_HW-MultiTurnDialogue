
from pathlib import Path

import torch
from opencc import OpenCC
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ==========================
# 1. 基本設定
# ==========================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "qlora" / "final_adapter"

MAX_NEW_TOKENS = 160
MAX_CONTEXT_TOKENS = 2048

converter = OpenCC("s2twp")

SYSTEM_PROMPT = (
    "你是一個土味情話高手，會使用繁體中文，"
    "以親切、溫和、幽默的口吻回答，"
    "能延續前文接梗，並尊重對方意願。"
)


# ==========================
# 2. 載入 QLoRA 模型
# ==========================

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可使用")

    if not (ADAPTER_PATH / "adapter_model.safetensors").exists():
        raise FileNotFoundError("找不到 QLoRA Adapter")

    print("正在載入模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )

    model.eval()

    print("QLoRA 模型載入成功！")

    return model, tokenizer


# ==========================
# 3. 產生回答
# ==========================

def generate_reply(model, tokenizer, history, user_text):

    # 自動保留近期上下文
    while True:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + history + [
            {"role": "user", "content": user_text}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if inputs["input_ids"].shape[-1] <= MAX_CONTEXT_TOKENS:
            break

        # 超過長度時，移除最早的一輪對話
        if len(history) >= 2:
            del history[:2]
        else:
            raise ValueError("輸入內容過長，請縮短問題")

    inputs = inputs.to("cuda:0")

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_length:]

    reply = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    # 統一轉換成台灣繁體中文
    reply = converter.convert(reply)

    # 更新對話紀錄
    history.append({
        "role": "user",
        "content": user_text,
    })

    history.append({
        "role": "assistant",
        "content": reply,
    })

    return reply


# ==========================
# 4. 多輪聊天介面
# ==========================

def main():
    model, tokenizer = load_model()

    history = []

    print("\n===== 土味情話專家 =====")
    print("輸入 /reset 重新開始對話")
    print("輸入 /exit 結束聊天\n")

    while True:
        try:
            user_text = input("你：").strip()

            if not user_text:
                continue

            if user_text.lower() == "/exit":
                print("聊天結束！")
                break

            if user_text.lower() == "/reset":
                history.clear()
                print("已清除對話紀錄！")
                continue

            reply = generate_reply(
                model,
                tokenizer,
                history,
                user_text,
            )

            print(f"\n土味情話專家：{reply}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n聊天結束！")
            break

        except Exception as error:
            print(f"發生錯誤：{error}")


if __name__ == "__main__":
    main()
