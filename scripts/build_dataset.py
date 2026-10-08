#!/usr/bin/env python3
"""
Merge raw extracts from iMessage and Slack, clean, format as Qwen3 chat pairs,
and split into train/valid/test JSONL files.

Usage:
  python scripts/build_dataset.py [--your-name Andy]
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

RAW_FILES = [
    Path("data/raw/imessage.jsonl"),
    Path("data/raw/slack.jsonl"),
]
OUT_DIR = Path("data/processed")

TRAIN_FRAC = 0.80
VALID_FRAC = 0.10
# test gets the rest

SYSTEM_TEMPLATE = "You are {name}. You are messaging {contact} in {channel}."

MIN_LEN = 3
MAX_LEN = 1024


def load_raw(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"  loaded {p}: {len(records):,} records so far")
    return records


def is_junk_reply(text: str) -> bool:
    if not text or len(text) < MIN_LEN or len(text) > MAX_LEN:
        return True
    clean = re.sub(r"[^\w]", "", text)
    if not clean:
        return True
    if re.fullmatch(r"https?://\S+", text.strip()):
        return True
    return False


def to_chat_example(record: dict, your_name: str) -> dict | None:
    contact = record.get("contact", "friend")
    channel = record.get("channel", "chat")
    context = record.get("context", [])
    reply = record.get("reply", "")

    if is_junk_reply(reply):
        return None
    if not context:
        return None

    system = SYSTEM_TEMPLATE.format(
        name=your_name,
        contact=contact,
        channel=channel,
    )
    user_content = "\n".join(context)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": reply},
        ]
    }


def write_split(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples):,} examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--your-name", default="Andy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading raw data...")
    records = load_raw(RAW_FILES)

    print(f"\nFormatting {len(records):,} records...")
    examples = []
    skipped = 0
    source_counts: Counter = Counter()
    for rec in records:
        ex = to_chat_example(rec, your_name=args.your_name)
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)
            source_counts[rec.get("source", "unknown")] += 1

    print(f"  kept: {len(examples):,}  dropped: {skipped:,}")
    for src, count in source_counts.most_common():
        print(f"    {src}: {count:,}")

    if not examples:
        print("No examples to write. Run the extract scripts first.")
        return

    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * TRAIN_FRAC)
    n_valid = int(n * VALID_FRAC)

    train = examples[:n_train]
    valid = examples[n_train : n_train + n_valid]
    test = examples[n_train + n_valid :]

    print(f"\nSplit: train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    print(f"Writing to {OUT_DIR}/")
    write_split(train, OUT_DIR / "train.jsonl")
    write_split(valid, OUT_DIR / "valid.jsonl")
    write_split(test, OUT_DIR / "test.jsonl")

    # rough token estimate (4 chars ≈ 1 token)
    total_chars = sum(
        len(m["content"])
        for ex in examples
        for m in ex["messages"]
    )
    print(f"\nEstimated tokens: ~{total_chars // 4:,}")
    print("Done.")


if __name__ == "__main__":
    main()
