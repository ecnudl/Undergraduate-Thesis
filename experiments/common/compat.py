"""Compatibility rules and runner plan helpers for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from common.config import DATASET_SCRIPTS, DEFAULT_AGENTS, ExperimentConfigError


VALID_STORAGE_TYPES = {"json", "vector", "hybrid", "graph", "llm_graph"}
VALID_RETRIEVERS = {
    "semantic",
    "keyword",
    "hybrid",
    "graph",
    "contrastive",
    "hybrid_graph",
}
VALID_PRESETS = {"none", "lightweight", "json_basic", "json_full", "graph_full"}
GRAPH_STORAGE_TYPES = {"graph", "llm_graph"}
GRAPH_RETRIEVER_TYPES = {"graph", "hybrid_graph"}


def validate_experiment_config(config: Dict) -> List[str]:
    """Validate compatibility and return informational notes."""
    notes: List[str] = list(config.get("notes", []))
    provider = config["provider"]
    storage_type = config["storage_backend"]["type"]
    retriever_type = config["retrieval_strategy"]["type"]
    preset = config["management_preset"]
    dataset = config["dataset"]
    agent = config["agent"]

    if provider not in {"modular", "none", "siliconfriend"}:
        raise ExperimentConfigError(
            "Unified experiment runner currently supports provider='modular', "
            f"'siliconfriend' or 'none', got '{provider}'"
        )
    if provider == "modular" and storage_type not in VALID_STORAGE_TYPES:
        raise ExperimentConfigError(
            f"Unsupported storage_backend.type '{storage_type}'"
        )
    if provider == "modular" and retriever_type not in VALID_RETRIEVERS:
        raise ExperimentConfigError(
            f"Unsupported retrieval_strategy.type '{retriever_type}'"
        )
    if provider == "modular" and preset not in VALID_PRESETS:
        raise ExperimentConfigError(f"Unsupported management_preset '{preset}'")

    expected_agent = DEFAULT_AGENTS[dataset]
    if agent != expected_agent:
        raise ExperimentConfigError(
            f"Dataset '{dataset}' expects agent '{expected_agent}', got '{agent}'"
        )

    if dataset == "longmemeval":
        if provider not in {"none", "modular", "siliconfriend"}:
            raise ExperimentConfigError(
                "LongMemEval supports provider='none', 'modular', or 'siliconfriend'."
            )
        if provider != "none" and config["shared_memory_provider"]:
            raise ExperimentConfigError(
                "LongMemEval memory runs must use shared_memory_provider=false because each benchmark item needs its own isolated memory store."
            )

    if provider == "modular":
        if retriever_type in GRAPH_RETRIEVER_TYPES and storage_type not in GRAPH_STORAGE_TYPES:
            raise ExperimentConfigError(
                f"Retriever '{retriever_type}' requires GraphStore or LLMGraphStore"
            )

        if preset == "graph_full" and storage_type not in GRAPH_STORAGE_TYPES:
            raise ExperimentConfigError(
                "management_preset 'graph_full' requires GraphStore or LLMGraphStore"
            )

        if storage_type not in GRAPH_STORAGE_TYPES:
            notes.append(
                "Non-graph storage uses the existing management pipeline fallback path; graph-enhanced logic is skipped or downgraded with logs."
            )
    elif provider == "none":
        notes.append("provider=none runs the no-memory baseline and ignores extraction/storage/retrieval/management settings.")
    else:
        notes.append(
            "provider=siliconfriend reuses the MemoryBank-SiliconFriend JSON+FAISS baseline and ignores modular extraction/storage/retrieval/management settings."
        )

    if config["concurrency"] != 1:
        notes.append(
            "concurrency>1 is allowed, but unified experiments are most stable with concurrency=1 because the memory store is shared across tasks."
        )

    return notes


def build_memory_env(config: Dict, run_dir: Path) -> Dict[str, str]:
    """Build environment overrides consumed by ModularMemoryProvider."""
    if config["provider"] == "none":
        return {}
    if config["provider"] == "siliconfriend":
        provider_cfg = dict(config.get("provider_config", {}))
        store_dir = (run_dir / "storage").resolve()
        index_dir = (store_dir / "index").resolve()
        env = {
            "SILICONFRIEND_STORE_DIR": str(store_dir),
            "SILICONFRIEND_MEMORY_FILE": provider_cfg.get("memory_file", "gaia_memory.json"),
            "SILICONFRIEND_INDEX_DIR": str(index_dir),
            "SILICONFRIEND_USER_NAME": str(provider_cfg.get("user_name", "gaia_eval")),
            "SILICONFRIEND_TOP_K": str(provider_cfg.get("top_k", 3)),
            "SILICONFRIEND_LANGUAGE": str(provider_cfg.get("language", "en")),
            "SILICONFRIEND_EMBEDDING_MODEL": str(provider_cfg.get("embedding_model", "minilm-l6")),
            "SILICONFRIEND_RESPONSE_MODE": str(provider_cfg.get("response_mode", "trajectory_summary")),
        }
        if config.get("embedding_device"):
            env["SILICONFRIEND_EMBEDDING_DEVICE"] = config["embedding_device"]
        elif provider_cfg.get("embedding_device"):
            env["SILICONFRIEND_EMBEDDING_DEVICE"] = str(provider_cfg["embedding_device"])
        return env

    extraction = config["extraction_config"]
    env = {
        "MODULAR_STORAGE_DIR": str((run_dir / "storage").resolve()),
        "MODULAR_STORAGE_TYPE": config["storage_backend"]["type"],
        "MODULAR_STORAGE_CONFIG": json.dumps(config["storage_backend"]["config"], ensure_ascii=False),
        "MODULAR_RETRIEVER_TYPE": config["retrieval_strategy"]["type"],
        "MODULAR_RETRIEVER_CONFIG": json.dumps(config["retrieval_strategy"]["config"], ensure_ascii=False),
        "MODULAR_ENABLED_PROMPTS": ",".join(extraction["enabled_prompts"]),
        "MODULAR_PROMPT_DIR": extraction["prompt_dir"],
        "MODULAR_TOP_K": str(extraction["top_k"]),
    }

    preset = config["management_preset"]
    if preset == "none":
        env["MODULAR_MANAGEMENT_ENABLED"] = "false"
    else:
        env["MODULAR_MANAGEMENT_ENABLED"] = "true"
        env["MODULAR_MANAGEMENT_PRESET"] = preset

    return env


def build_runner_command(config: Dict, repo_root: Path, run_dir: Path) -> List[str]:
    """Build the dataset runner CLI command."""
    script = repo_root / DATASET_SCRIPTS[config["dataset"]]
    tasks_dir = run_dir / "tasks"
    results_path = run_dir / "results.jsonl"

    cmd = [
        "python",
        str(script),
        "--infile",
        config["dataset_path"],
        "--outfile",
        str(results_path),
        "--max_steps",
        str(config["max_steps"]),
        "--concurrency",
        str(config["concurrency"]),
        "--summary_interval",
        str(config["summary_interval"]),
        "--prompts_type",
        config["prompts_type"],
        "--seed",
        str(config["seed"]),
        "--token_budget",
        str(config["token_budget"]),
        "--direct_output_dir",
        str(tasks_dir),
    ]

    if config["provider"] != "none":
        cmd.extend(["--memory_provider", config["provider"]])

    if config["model"]:
        cmd.extend(["--model", config["model"]])
    if config["judge_model"]:
        cmd.extend(["--judge_model", config["judge_model"]])
    if config["sample_num"] is not None:
        cmd.extend(["--sample_num", str(config["sample_num"])])
    if config.get("selection_file"):
        cmd.extend(["--selection_file", str(config["selection_file"])])
    if config["task_indices"]:
        cmd.extend(["--task_indices", str(config["task_indices"])])
    if config["provider"] != "none":
        if config["enable_memory_evolution"]:
            cmd.append("--enable_memory_evolution")
        else:
            cmd.append("--disable_memory_evolution")
    if config["provider"] != "none" and config["shared_memory_provider"]:
        cmd.append("--shared_memory_provider")

    return cmd
