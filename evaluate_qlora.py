
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from opencc import OpenCC

converter = OpenCC("s2twp")


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "qlora" / "final_adapter"
TEST_PATH = ROOT / "data" / "test_scenarios.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

USER_FIELDS = [
    "first_user",
    "follow_up_user",
    "third_user",
]


def generate_reply(model, tokenizer, history):
    inputs = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda:0")

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

        # 統一轉為台灣繁體中文
        return converter.convert(reply)


def run_scenarios(model, tokenizer, scenarios, name):
    results = {}
    model.eval()

    for scenario in scenarios:
        print(
            f"[{name}] {scenario['id']}",
            flush=True
        )

        # 每個模型都有自己的對話歷史
        history = [{
            "role": "system",
            "content": scenario["system_prompt"],
        }]

        replies = []

        for field in USER_FIELDS:
            question = scenario[field]

            history.append({
                "role": "user",
                "content": question,
            })

            reply = generate_reply(
                model, tokenizer, history
            )

            replies.append(reply)

            history.append({
                "role": "assistant",
                "content": reply,
            })

        results[scenario["id"]] = replies

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先測試指定數量的情境",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可使用")

    if not (
        ADAPTER_PATH / "adapter_model.safetensors"
    ).exists():
        raise FileNotFoundError("找不到微調後的 Adapter")

    scenarios = [
        json.loads(line)
        for line in TEST_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit 必須大於 0")
        scenarios = scenarios[:args.limit]

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

    print("載入原始 Qwen 模型...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )

    # 1. 原始模型測試
    base_results = run_scenarios(
        model, tokenizer, scenarios, "BASE"
    )

    # 2. 載入微調 Adapter
    print("載入 QLoRA Adapter...", flush=True)

    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )

    # 3. 微調後模型測試
    tuned_results = run_scenarios(
        model, tokenizer, scenarios, "QLORA"
    )

    comparisons = []

    for scenario in scenarios:
        sid = scenario["id"]

        turns = []
        for index, field in enumerate(USER_FIELDS):
            turns.append({
                "round": index + 1,
                "user": scenario[field],
                "base": base_results[sid][index],
                "qlora": tuned_results[sid][index],
            })

        comparisons.append({
            "id": sid,
            "scenario": scenario["scenario"],
            "checks": scenario["checks"],
            "turns": turns,
        })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"preview_{args.limit}.json"
        if args.limit is not None
        else "base_vs_qlora.json"
    )

    output_path = OUTPUT_DIR / filename

    output_path.write_text(
        json.dumps(
            comparisons,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 產生方便閱讀的 Markdown 報告
    md = ["# QLoRA 微調前後對話比較", ""]

    for item in comparisons:
        md.append(
            f"## {item['id']}：{item['scenario']}"
        )
        md.append("")

        for turn in item["turns"]:
            md.extend([
                f"### 第 {turn['round']} 輪",
                f"**使用者：** {turn['user']}",
                "",
                f"**原始模型：** {turn['base']}",
                "",
                f"**微調後模型：** {turn['qlora']}",
                "",
            ])

        md.append("### 評估條件")
        for check in item["checks"]:
            md.append(f"- [ ] {check}")
        md.append("")

    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print("比較完成！")
    print("JSON:", output_path)
    print("Markdown:", markdown_path)


if __name__ == "__main__":
    main()
