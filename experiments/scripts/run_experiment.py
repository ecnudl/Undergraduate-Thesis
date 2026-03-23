#!/usr/bin/env python
"""Unified experiment runner for the modular memory system."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.aggregation import aggregate_run
from common.compat import build_memory_env, build_runner_command, validate_experiment_config
from common.config import ExperimentConfigError, load_experiment_config, normalize_experiment_config


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_runner_python() -> str:
    override = os.environ.get("EXPERIMENT_PYTHON", "").strip()
    if override:
        return override

    preferred = Path("/home/admin123/anaconda3/envs/evolvelab/bin/python")
    if preferred.exists():
        return str(preferred)

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a unified modular-memory experiment")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML experiment config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and materialize run manifest without executing")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw_config = load_experiment_config(config_path)
    config = normalize_experiment_config(raw_config, REPO_ROOT)
    notes = validate_experiment_config(config)

    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    run_dir = output_root / f"{config['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(exist_ok=True)

    env = os.environ.copy()
    api_base = config.get("api_base") or env.get("OPENAI_API_BASE") or "https://api-vip.codex-for.me/v1"
    judge_api_base = config.get("judge_api_base") or api_base
    openai_base_url = config.get("openai_base_url") or api_base

    env["OPENAI_API_BASE"] = api_base
    env["JUDGE_API_BASE"] = judge_api_base
    env["OPENAI_BASE_URL"] = openai_base_url
    env["DEFAULT_MODEL"] = config["model"]
    env["DEFAULT_JUDGE_MODEL"] = config["judge_model"]
    env.setdefault("FORCE_STREAM", "1")
    if config.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    if config.get("embedding_device"):
        env["MEMORY_EMBEDDING_DEVICE"] = config["embedding_device"]
    env.update(build_memory_env(config, run_dir))
    command = build_runner_command(config, REPO_ROOT, run_dir)
    command[0] = _resolve_runner_python()

    manifest = {
        "config_path": str(config_path),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "created_at_utc": now_utc.isoformat(),
        "config": config,
        "compatibility_notes": notes,
        "command": command,
        "memory_env": {k: env[k] for k in sorted(build_memory_env(config, run_dir).keys())},
        "dry_run": bool(args.dry_run or config.get("dry_run")),
        "status": "planned",
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_manifest(manifest_path, manifest)

    if manifest["dry_run"]:
        print(json.dumps({"run_dir": str(run_dir), "command": command, "notes": notes}, ensure_ascii=False, indent=2))
        return 0

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    manifest["status"] = "completed" if process.returncode == 0 else "failed"
    manifest["exit_code"] = process.returncode
    manifest["log_path"] = str(log_path)

    if process.returncode == 0:
        aggregate = aggregate_run(run_dir)
        manifest["artifacts"] = aggregate
    _write_manifest(manifest_path, manifest)

    if process.returncode != 0:
        raise SystemExit(process.returncode)

    print(json.dumps({"run_dir": str(run_dir), "summary_csv": manifest["artifacts"]["summary_csv"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)
