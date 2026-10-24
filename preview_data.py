import json
from pathlib import Path

from transformers import AutoTokenizer


# 路徑以這支程式所在的資料夾為準
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-1.5B-Instruct"

# 只讀取本機 tokenizer，不下載模型
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

for split in ("train", "eval"):
    data_path = ROOT / "data" / f"{split}.jsonl"
    lengths = []

    with data_path.open("r", encoding="utf-8-sig") as file:
        for index, line in enumerate(file):
            record = json.loads(line)

            # 把 messages 轉成 Qwen 原生的聊天格式
            text = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            # 模板已有特殊標記，不要重複加入
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            lengths.append(len(token_ids))

            if split == "train" and index == 0:
                print("=== 第一筆訓練資料的聊天格式 ===")
                print(text)
                print("=== 預覽結束 ===\n")

    print(f"{split}：{len(lengths)} 筆")
    print(f"最短：{min(lengths)} tokens")
    print(f"最長：{max(lengths)} tokens")
    print(f"超過 512 tokens：{sum(n > 512 for n in lengths)} 筆")
    print()