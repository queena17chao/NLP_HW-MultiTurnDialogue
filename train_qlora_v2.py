
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# =========================
# 1. 基本設定
# =========================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-1.5B-Instruct"

TRAIN_PATH = ROOT / "data" / "train_v2.jsonl"
EVAL_PATH = ROOT / "data" / "eval.jsonl"

MAX_LENGTH = 512


# =========================
# 2. 讀取多輪對話資料
# =========================

def load_conversations(path):
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            history = []

            for msg in record["messages"]:

                if msg["role"] == "assistant":
                    samples.append({
                        "prompt": history.copy(),
                        "completion": [msg],
                    })

                history.append(msg)

    return Dataset.from_list(samples)


# =========================
# 3. 檢查 Token 長度
# =========================

def check_lengths(dataset, tokenizer, name):
    lengths = []

    for sample in dataset:
        messages = (
            sample["prompt"] + sample["completion"]
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(ids))

    maximum = max(lengths)

    print(f"{name} 最大 Token：{maximum}")

    if maximum > MAX_LENGTH:
        raise ValueError(
            f"{name} 超過 {MAX_LENGTH} Tokens"
        )


# =========================
# 4. QLoRA 微調主程式
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只執行 2 個訓練步驟",
    )

    args = parser.parse_args()

    print("===== QLoRA 微調開始 =====", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可使用")

    print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # 載入 Tokenizer
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # 載入資料集
    # -------------------------

    train_data = load_conversations(TRAIN_PATH)
    eval_data = load_conversations(EVAL_PATH)

    print("訓練樣本：", len(train_data))
    print("驗證樣本：", len(eval_data))

    check_lengths(train_data, tokenizer, "Train")
    check_lengths(eval_data, tokenizer, "Eval")

    if args.smoke_test:
        print("模式：Smoke Test")
        train_data = train_data.select(range(8))
    else:
        print("模式：正式訓練")

    # -------------------------
    # 4-bit NF4 量化
    # -------------------------

    use_bf16 = torch.cuda.is_bf16_supported()

    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    print("載入 4-bit 模型...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=dtype,
    )

    model.config.use_cache = False

    # -------------------------
    # LoRA 設定
    # -------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------------------------
    # 訓練輸出位置
    # -------------------------

    output_dir = ROOT / "outputs" / (
        "smoke_test_v2"
        if args.smoke_test
        else "qlora_v2"
    )

    # -------------------------
    # 訓練參數
    # -------------------------

    training_args = SFTConfig(
        output_dir=str(output_dir),

        num_train_epochs=3,
        max_steps=2 if args.smoke_test else -1,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-4,
        optim="adamw_torch",

        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,

        bf16=use_bf16,
        fp16=not use_bf16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        eval_strategy=(
            "no" if args.smoke_test else "epoch"
        ),
        save_strategy=(
            "no" if args.smoke_test else "epoch"
        ),

        load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,
        logging_steps=1 if args.smoke_test else 10,

        report_to="none",
        seed=42,
    )

    # -------------------------
    # 建立 Trainer
    # -------------------------

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.model.print_trainable_parameters()

    # -------------------------
    # 驗證 Loss Mask
    # -------------------------

    batch = trainer.data_collator(
        [trainer.train_dataset[0]]
    )

    labels = batch["labels"]

    assert (labels != -100).any().item(), (
        "沒有有效的訓練目標"
    )

    assert (labels == -100).any().item(), (
        "Prompt 沒有被遮罩"
    )

    print("Completion Loss Mask: OK")

    # -------------------------
    # 開始訓練
    # -------------------------

    print("開始訓練...", flush=True)

    torch.cuda.reset_peak_memory_stats()

    result = trainer.train()

    print("Training Loss:", result.training_loss)

    peak = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print(f"Peak VRAM: {peak:.2f} GiB")

    # -------------------------
    # Smoke Test 結束
    # -------------------------

    if args.smoke_test:
        print("SMOKE TEST: PASS")
        return

    # -------------------------
    # 儲存微調後 Adapter
    # -------------------------

    adapter_path = output_dir / "final_adapter"

    trainer.save_model(str(adapter_path))

    tokenizer.save_pretrained(
        str(adapter_path)
    )

    print("QLoRA 訓練完成！")
    print("Adapter 儲存位置：", adapter_path)


if __name__ == "__main__":
    main()
