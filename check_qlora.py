
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


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512


def load_dataset(path):
    samples = []

    with path.open(encoding="utf-8") as f:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可使用")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_dataset(
        ROOT / "data" / "train.jsonl"
    )
    eval_data = load_dataset(
        ROOT / "data" / "eval.jsonl"
    )

    print("訓練樣本：", len(train_data))
    print("驗證樣本：", len(eval_data))

    # 使用完整文字計算真正的 Token 數
    for name, dataset in [
        ("train", train_data),
        ("eval", eval_data),
    ]:
        lengths = []

        for sample in dataset:
            text = tokenizer.apply_chat_template(
                sample["prompt"] + sample["completion"],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            lengths.append(len(ids))

        print(name, "最大 Tokens：", max(lengths))

        if max(lengths) > MAX_LENGTH:
            raise ValueError(
                f"{name} 超過 {MAX_LENGTH} Tokens"
            )

    if args.smoke_test:
        train_data = train_data.select(range(8))

    use_bf16 = torch.cuda.is_bf16_supported()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if use_bf16
            else torch.float16
        ),
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=(
            torch.bfloat16 if use_bf16
            else torch.float16
        ),
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = ROOT / "outputs" / (
        "smoke_test" if args.smoke_test
        else "qlora"
    )

    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        max_steps=2 if args.smoke_test else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        optim="adamw_torch",
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

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.model.print_trainable_parameters()

    # 確認至少有部分 Token 參與 Loss 計算
    batch = trainer.data_collator(
        [trainer.train_dataset[0]]
    )
    labels = batch["labels"]
    assert (labels != -100).any().item()
    assert (labels == -100).any().item()
    print("Completion Loss Mask: OK")

    print("開始訓練...")
    result = trainer.train()

    print("Training Loss:", result.training_loss)
    print(
        "Peak VRAM:",
        round(
            torch.cuda.max_memory_reserved() / 1024**3,
            2
        ),
        "GiB"
    )

    if args.smoke_test:
        print("SMOKE TEST: PASS")
        return

    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print("QLoRA 訓練完成")
    print("Adapter:", adapter_path)


if __name__ == "__main__":
    main()
