"""Config loading and normalization for unified experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ExperimentConfigError(ValueError):
    """Raised when an experiment config is invalid."""


DATASET_SCRIPTS = {
    "gaia": "run_flash_searcher_mm_gaia.py",
    "longmemeval": "run_flash_searcher_longmemeval.py",
    "webwalkerqa": "run_flash_searcher_webwalkerqa.py",
    "xbench": "run_flash_searcher_mm_xbench.py",
}

DEFAULT_AGENTS = {
    "gaia": "mm_search_agent",
    "longmemeval": "long_context_qa",
    "webwalkerqa": "search_agent",
    "xbench": "mm_search_agent",
}

DEFAULT_DATASET_PATHS = {
    ("gaia", "validation"): "./data/gaia/validation/metadata.jsonl",
    ("longmemeval", "oracle"): "./data/longmemeval/longmemeval_oracle.json",
    ("longmemeval", "s_cleaned"): "./data/longmemeval/longmemeval_s_cleaned.json",
    ("longmemeval", "m_cleaned"): "./data/longmemeval/longmemeval_m_cleaned.json",
    ("webwalkerqa", "subset_170"): "./data/webwalkerqa/webwalkerqa_subset_170.jsonl",
    ("xbench", "default"): "./data/xbench/DeepSearch.csv",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ExperimentConfigError(
            "YAML config requires PyYAML. Use JSON or install pyyaml."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ExperimentConfigError(f"Config root must be an object: {path}")
    return data


def load_experiment_config(path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML experiment config."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    else:
        raise ExperimentConfigError(f"Unsupported config extension: {path.suffix}")

    if not isinstance(data, dict):
        raise ExperimentConfigError(f"Config root must be an object: {path}")
    return data


def _as_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"'{field_name}' must be an object")
    return value


def _default_dataset_path(dataset: str, split: str) -> str:
    try:
        return DEFAULT_DATASET_PATHS[(dataset, split)]
    except KeyError as exc:
        raise ExperimentConfigError(
            f"No default dataset path for dataset='{dataset}' split='{split}'. "
            "Set 'dataset_path' explicitly."
        ) from exc


def normalize_experiment_config(raw: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """Normalize and validate the user config shape."""
    dataset = str(raw.get("dataset", "")).strip().lower()
    if dataset not in DATASET_SCRIPTS:
        raise ExperimentConfigError(
            f"Unsupported dataset '{dataset}'. Supported: {sorted(DATASET_SCRIPTS)}"
        )

    split = str(raw.get("split", "")).strip().lower()
    if not split:
        split = {
            "gaia": "validation",
            "longmemeval": "s_cleaned",
            "webwalkerqa": "subset_170",
            "xbench": "default",
        }[dataset]

    agent = str(raw.get("agent") or DEFAULT_AGENTS[dataset]).strip().lower()
    name = str(raw.get("name") or f"{dataset}_{split}_{raw.get('storage_backend', {}).get('type', 'json')}").strip()
    if not name:
        raise ExperimentConfigError("'name' must not be empty")

    extraction_config = _as_dict(raw.get("extraction_config"), "extraction_config")
    storage_backend = _as_dict(raw.get("storage_backend"), "storage_backend")
    retrieval_strategy = _as_dict(raw.get("retrieval_strategy"), "retrieval_strategy")
    provider = str(raw.get("provider", "modular")).strip().lower()
    storage_type = str(storage_backend.get("type", "")).strip().lower()
    retrieval_type = str(retrieval_strategy.get("type", "")).strip().lower()
    if provider == "modular" and not storage_type:
        raise ExperimentConfigError("'storage_backend.type' is required")
    if provider == "modular" and not retrieval_type:
        raise ExperimentConfigError("'retrieval_strategy.type' is required")

    dataset_path = raw.get("dataset_path") or _default_dataset_path(dataset, split)
    dataset_path = str((repo_root / dataset_path).resolve()) if not Path(dataset_path).is_absolute() else str(Path(dataset_path).resolve())
    selection_file = str(raw.get("selection_file", "")).strip()
    if selection_file:
        selection_file = str((repo_root / selection_file).resolve()) if not Path(selection_file).is_absolute() else str(Path(selection_file).resolve())

    normalized = {
        "name": name,
        "provider": provider,
        "model": str(raw.get("model", "gpt-5")).strip() or "gpt-5",
        "judge_model": str(raw.get("judge_model", "gpt-5")).strip() or "gpt-5",
        "api_base": str(raw.get("api_base", "")).strip(),
        "judge_api_base": str(raw.get("judge_api_base", "")).strip(),
        "openai_base_url": str(raw.get("openai_base_url", "")).strip(),
        "embedding_device": str(raw.get("embedding_device", "")).strip(),
        "cuda_visible_devices": str(raw.get("cuda_visible_devices", "")).strip(),
        "provider_config": _as_dict(raw.get("provider_config"), "provider_config"),
        "agent": agent,
        "dataset": dataset,
        "split": split,
        "dataset_path": dataset_path,
        "selection_file": selection_file,
        "seed": int(raw.get("seed", 42)),
        "max_steps": int(raw.get("max_steps", 40)),
        "token_budget": int(raw.get("token_budget", 8000)),
        "sample_num": raw.get("sample_num"),
        "task_indices": raw.get("task_indices"),
        "concurrency": int(raw.get("concurrency", 1)),
        "summary_interval": int(raw.get("summary_interval", 8)),
        "prompts_type": str(raw.get("prompts_type", "default")),
        "enable_memory_evolution": bool(raw.get("enable_memory_evolution", True)),
        "shared_memory_provider": bool(raw.get("shared_memory_provider", True)),
        "dry_run": bool(raw.get("dry_run", False)),
        "output_root": str(raw.get("output_root", "./experiments/results")),
        "notes": list(raw.get("notes", [])),
        "extraction_config": {
            "enabled_prompts": list(extraction_config.get("enabled_prompts", ["tip", "insight"])),
            "prompt_dir": str(extraction_config.get("prompt_dir", ".")),
            "top_k": int(extraction_config.get("top_k", 5)),
        },
        "storage_backend": {
            "type": storage_type or "json",
            "config": _as_dict(storage_backend.get("config"), "storage_backend.config"),
        },
        "retrieval_strategy": {
            "type": retrieval_type or "semantic",
            "config": _as_dict(retrieval_strategy.get("config"), "retrieval_strategy.config"),
        },
        "management_preset": str(raw.get("management_preset", "none")).strip().lower(),
    }

    if normalized["sample_num"] is not None:
        normalized["sample_num"] = int(normalized["sample_num"])

    return normalized
