#!/usr/bin/env python3
"""
下载 GAIA 完整验证集到 data/gaia/validation/
生成 metadata.jsonl + 附件文件，路径适配 run_flash_searcher_mm_gaia.py
"""

import json
import os
import sys
import shutil

from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "data", "gaia_full")
TARGET_DIR = os.path.join(SCRIPT_DIR, "data", "gaia", "validation")


def main():
    # 延迟导入，便于提前报错
    try:
        from huggingface_hub import login, hf_hub_download, list_repo_files
        import pandas as pd
    except ImportError:
        print("[ERROR] 请先安装依赖: pip install huggingface_hub pandas pyarrow")
        return

    if not HF_TOKEN:
        print("[ERROR] 请设置 HF_TOKEN 环境变量或在 .env 中配置")
        print("  例: HF_TOKEN=hf_xxxx python download_gaia.py")
        sys.exit(1)

    print("[1/4] 登录 HuggingFace...")
    login(token=HF_TOKEN)

    repo_id = "gaia-benchmark/GAIA"

    print("[2/4] 获取文件列表...")
    all_files = list(list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN))
    validation_files = [f for f in all_files if f.startswith("2023/validation/")]
    print(f"      找到 {len(validation_files)} 个验证集文件")

    print("[3/4] 下载文件...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for i, filepath in enumerate(sorted(validation_files), 1):
        filename = os.path.basename(filepath)
        print(f"      [{i}/{len(validation_files)}] {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filepath,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=DOWNLOAD_DIR,
        )

    print("[4/4] 转换为 JSONL 并复制附件...")
    parquet_path = os.path.join(DOWNLOAD_DIR, "2023", "validation", "metadata.parquet")
    df = pd.read_parquet(parquet_path)

    os.makedirs(TARGET_DIR, exist_ok=True)

    # 写 metadata.jsonl
    jsonl_path = os.path.join(TARGET_DIR, "metadata.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "task_id": row["task_id"],
                "Question": row["Question"],
                "Final answer": row["Final answer"],
                "Level": str(row["Level"]),
                "file_name": row["file_name"] if row["file_name"] else "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 复制附件
    src_dir = os.path.join(DOWNLOAD_DIR, "2023", "validation")
    copied = 0
    for _, row in df.iterrows():
        fname = row["file_name"]
        if not fname:
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(TARGET_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    # 统计
    levels = df["Level"].value_counts().sort_index()
    print()
    print("=" * 40)
    print(f"  下载完成!")
    print(f"  总任务数: {len(df)}")
    for lv, cnt in levels.items():
        print(f"    Level {lv}: {cnt}")
    print(f"  附件文件: {copied} 个已复制")
    print(f"  数据路径: {jsonl_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
