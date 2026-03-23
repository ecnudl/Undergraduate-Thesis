#!/usr/bin/env python3
"""
Download WebWalkerQA dataset from HuggingFace and prepare evaluation data.

Outputs:
  data/webwalkerqa/webwalkerqa_main.jsonl      — full dataset
  data/webwalkerqa/webwalkerqa_subset_170.jsonl — sampled 170 tasks (seed=42)

Usage:
  python download_webwalkerqa.py
  python download_webwalkerqa.py --sample_size 200
"""

import json
import os
import random
import argparse

from dotenv import load_dotenv

load_dotenv(override=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "data", "webwalkerqa")


def main():
    parser = argparse.ArgumentParser(description="Download WebWalkerQA dataset")
    parser.add_argument("--sample_size", type=int, default=170,
                        help="Number of tasks to sample (default: 170)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] Install dependency: pip install datasets")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)

    # Step 1: Download from HuggingFace
    print("[1/3] Downloading WebWalkerQA from HuggingFace...")
    hf_token = os.environ.get("HF_TOKEN", None)
    ds = load_dataset("callanwu/WebWalkerQA", token=hf_token)

    # Determine split — typically "test" or "train"
    available_splits = list(ds.keys())
    print(f"      Available splits: {available_splits}")

    # Field mapping: HF uses capitalized keys, eval script expects lowercase
    FIELD_MAP = {
        "Question": "question",
        "Answer": "answer",
        "Root_Url": "root_url",
        "Info": "info",
    }

    all_data = []
    for split_name in available_splits:
        split_ds = ds[split_name]
        print(f"      Split '{split_name}': {len(split_ds)} tasks")
        for item in split_ds:
            item_dict = {}
            for k, v in dict(item).items():
                mapped_key = FIELD_MAP.get(k, k.lower())
                item_dict[mapped_key] = v
            # Normalize info sub-fields to lowercase keys
            if isinstance(item_dict.get("info"), dict):
                info = item_dict["info"]
                normalized = {}
                for ik, iv in info.items():
                    # Map known fields: Difficulty_Level -> difficulty_level, etc.
                    normalized[ik.lower() if ik == ik.capitalize() or "_" in ik else ik] = iv
                item_dict["info"] = normalized
            item_dict["_split"] = split_name
            all_data.append(item_dict)

    print(f"      Total tasks: {len(all_data)}")

    # Step 2: Save full dataset
    print("[2/3] Saving full dataset...")
    main_path = os.path.join(TARGET_DIR, "webwalkerqa_main.jsonl")
    with open(main_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"      Saved to {main_path}")

    # Step 3: Sample subset
    print(f"[3/3] Sampling {args.sample_size} tasks (seed={args.seed})...")
    random.seed(args.seed)

    if args.sample_size >= len(all_data):
        sampled = all_data
        print(f"      Sample size >= total, using all {len(all_data)} tasks")
    else:
        sampled = random.sample(all_data, args.sample_size)

    subset_path = os.path.join(TARGET_DIR, f"webwalkerqa_subset_{args.sample_size}.jsonl")
    with open(subset_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"      Saved to {subset_path}")

    # Summary statistics
    print()
    print("=" * 50)
    print(f"  Download complete!")
    print(f"  Total tasks:   {len(all_data)}")
    print(f"  Sampled tasks: {len(sampled)}")

    # Show difficulty distribution if available
    difficulties = {}
    for item in all_data:
        info = item.get("info", {})
        if isinstance(info, dict):
            d = info.get("difficulty_level", "unknown")
        else:
            d = "unknown"
        difficulties[d] = difficulties.get(d, 0) + 1

    if difficulties:
        print(f"  Difficulty distribution:")
        for d in sorted(difficulties.keys()):
            print(f"    {d}: {difficulties[d]}")

    # Show language distribution
    langs = {}
    for item in all_data:
        info = item.get("info", {})
        if isinstance(info, dict):
            lang = info.get("lang", "unknown")
        else:
            lang = "unknown"
        langs[lang] = langs.get(lang, 0) + 1

    if langs:
        print(f"  Language distribution:")
        for lang in sorted(langs.keys()):
            print(f"    {lang}: {langs[lang]}")

    print(f"  Data directory: {TARGET_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
