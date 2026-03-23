#!/usr/bin/env python
"""
run_adaptive_search.py — Adaptive Architecture Search with Tie-Breaking

Wraps the existing search_memory_architecture.py framework with an
adaptive proxy set expansion strategy:

  1. Run Phase A with proxy_indices_stage1 (e.g., 10 tasks)
  2. Check if top candidates are tied (spread < tie_threshold)
  3. If tied, re-run ONLY the tied candidates on proxy_indices_stage2 (e.g., 20 tasks)
  4. Proceed to Phase B with the same adaptive logic
  5. Phase D uses holdout set (no adaptation needed)

This solves the problem where a small proxy set (5 tasks) gives 100%
accuracy to all configs, making differentiation impossible.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from search_memory_architecture import (
    ArchConfig,
    ExperimentResult,
    ExperimentRunner,
    RunRegistry,
    compute_composite_scores,
    check_compatibility,
    is_config_valid,
    phase_a,
    phase_b,
    phase_d,
    EXTRACTION_PRESETS,
    STORAGE_CANDIDATES,
    RETRIEVAL_CANDIDATES,
    MANAGEMENT_CANDIDATES,
    DEFAULT_ANCHOR,
    _write_results_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adaptive_search")


def check_tied(results, scoring_weights, tie_threshold=0.03):
    """Check if top results are too close to differentiate."""
    if len(results) < 2:
        return False, 0.0

    completed = [r for r in results if r.status in ("completed", "dry_run")]
    if len(completed) < 2:
        return False, 0.0

    scores = compute_composite_scores(completed, scoring_weights)

    # Also check raw task_score spread (more interpretable)
    task_scores = sorted([r.task_score for r in completed], reverse=True)
    spread = task_scores[0] - task_scores[-1]

    logger.info(f"  Score spread: {spread:.4f} (threshold: {tie_threshold})")
    return spread < tie_threshold, spread


def expand_and_rerun(runner, configs, stage2_indices, scoring_weights,
                     run_label, id_prefix="s2_"):
    """Re-run tied configs on expanded proxy set for tie-breaking."""
    logger.info(f"\n  === ADAPTIVE EXPANSION: re-running {len(configs)} tied configs "
                f"on expanded set ({stage2_indices}) ===\n")

    expanded_results = []
    for cfg in configs:
        if isinstance(cfg, ExperimentResult):
            arch_cfg = cfg.config
        else:
            arch_cfg = cfg

        result = runner.run(
            arch_cfg,
            run_label=f"{run_label}_expanded",
            task_indices=stage2_indices,
            id_prefix=id_prefix,
        )
        expanded_results.append(result)

    return expanded_results


def adaptive_phase_a(runner, search_config, output_dir):
    """Phase A with adaptive expansion on tied modules."""
    stage1_indices = search_config["dataset"].get("proxy_indices_stage1", "[level1]1-10")
    stage2_indices = search_config["dataset"].get("proxy_indices_stage2", "[level1]1-20")
    a_cfg = search_config.get("phase_a", {})
    tie_threshold = a_cfg.get("tie_threshold", 0.03)
    adaptive = a_cfg.get("adaptive_expansion", True)
    scoring_weights = search_config.get("scoring", {})
    anchor = search_config.get("anchor", DEFAULT_ANCHOR)
    top_k_cfg = a_cfg.get("top_k", {})

    # Override proxy_indices to stage1 for initial run
    modified_config = json.loads(json.dumps(search_config))
    modified_config["dataset"]["proxy_indices"] = stage1_indices

    logger.info("=" * 60)
    logger.info("ADAPTIVE PHASE A: Module Screening (Stage 1)")
    logger.info(f"  Proxy set: {stage1_indices}")
    logger.info("=" * 60)

    # Run standard Phase A
    retained, module_results = phase_a(runner, modified_config, output_dir)

    if not adaptive:
        return retained, module_results

    # Check each module for ties and expand if needed
    for module_name, results in module_results.items():
        if len(results) < 2:
            continue

        is_tied, spread = check_tied(results, scoring_weights, tie_threshold)
        if is_tied:
            logger.info(f"\n  Module '{module_name}' is tied (spread={spread:.4f}). "
                        f"Expanding to {stage2_indices}...")

            # Get the configs that need re-evaluation
            configs_to_expand = [r.config for r in results if r.status == "completed"]

            expanded = expand_and_rerun(
                runner, configs_to_expand, stage2_indices, scoring_weights,
                run_label=f"phase_a/{module_name}",
                id_prefix=f"s2_{module_name}_",
            )

            # Replace module_results with expanded results
            module_results[module_name] = expanded

            # Re-select top-k based on expanded scores
            k = top_k_cfg.get(module_name, 2)
            completed = [r for r in expanded if r.status in ("completed", "dry_run")]
            if completed:
                scores = compute_composite_scores(completed, scoring_weights)
                ranked = sorted(zip(scores, completed), key=lambda x: -x[0])
                retained[module_name] = [
                    r.config.__dict__[module_name] for _, r in ranked[:k]
                ]
                logger.info(f"  [{module_name}] After expansion, retained: {retained[module_name]}")
        else:
            logger.info(f"  Module '{module_name}': spread={spread:.4f}, no expansion needed")

    # Save updated checkpoint
    checkpoint = {
        "retained": retained,
        "module_results": {
            mod: [r.to_dict() for r in results]
            for mod, results in module_results.items()
        },
        "adaptive_expanded": True,
    }
    with (output_dir / "checkpoint_a_adaptive.json").open("w") as f:
        json.dump(checkpoint, f, indent=2)

    return retained, module_results


def adaptive_phase_b(runner, retained, search_config, output_dir, baseline_result=None):
    """Phase B with adaptive expansion on tied elites."""
    stage1_indices = search_config["dataset"].get("proxy_indices_stage1", "[level1]1-10")
    stage2_indices = search_config["dataset"].get("proxy_indices_stage2", "[level1]1-20")
    b_cfg = search_config.get("phase_b", {})
    tie_threshold = b_cfg.get("tie_threshold", 0.03)
    adaptive = b_cfg.get("adaptive_expansion", True)
    scoring_weights = search_config.get("scoring", {})

    # Run standard Phase B with stage1
    modified_config = json.loads(json.dumps(search_config))
    modified_config["dataset"]["proxy_indices"] = stage1_indices

    logger.info("\n" + "=" * 60)
    logger.info("ADAPTIVE PHASE B: Reduced Space Search (Stage 1)")
    logger.info(f"  Proxy set: {stage1_indices}")
    logger.info("=" * 60)

    elites = phase_b(runner, retained, modified_config, output_dir, baseline_result)

    if not adaptive or len(elites) < 2:
        return elites

    # Check if top elites are tied
    is_tied, spread = check_tied(elites, scoring_weights, tie_threshold)
    if is_tied:
        logger.info(f"\n  Top elites are tied (spread={spread:.4f}). "
                    f"Expanding to {stage2_indices}...")

        expanded = expand_and_rerun(
            runner, elites, stage2_indices, scoring_weights,
            run_label="phase_b",
            id_prefix="s2_b_",
        )

        # Re-rank
        completed = [r for r in expanded if r.status in ("completed", "dry_run")]
        scores = compute_composite_scores(completed, scoring_weights)
        ranked = sorted(zip(scores, completed), key=lambda x: -x[0])

        top_n = b_cfg.get("top_elites", 5)
        elites = [r for _, r in ranked[:top_n]]

        # Save expanded checkpoint
        with (output_dir / "checkpoint_b_adaptive.json").open("w") as f:
            json.dump({
                "elites": [e.to_dict() for e in elites],
                "expanded": True,
                "stage2_indices": stage2_indices,
            }, f, indent=2)

        logger.info(f"\n  After expansion, top elites:")
        for i, (s, r) in enumerate(ranked[:top_n], 1):
            logger.info(f"    #{i} {r.config.short_name}: score={r.task_score:.3f}")
    else:
        logger.info(f"  Elites are well separated (spread={spread:.4f}), no expansion needed")

    return elites


def run_no_memory_baseline(runner, search_config, output_dir, task_indices):
    """Run no-memory baseline for comparison."""
    logger.info("\n--- Running No-Memory Baseline ---")

    baseline_dir = output_dir / "baseline" / "no_memory"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = baseline_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    dataset_cfg = search_config.get("dataset", {})
    runner_cfg = search_config.get("runner", {})
    dataset_name = dataset_cfg.get("name", "gaia")

    script_map = {
        "gaia": "run_flash_searcher_mm_gaia.py",
        "webwalkerqa": "run_flash_searcher_webwalkerqa.py",
    }
    script = str(REPO_ROOT / script_map.get(dataset_name, script_map["gaia"]))
    infile = dataset_cfg.get("infile", "./data/gaia/validation/metadata.jsonl")
    if not Path(infile).is_absolute():
        infile = str(REPO_ROOT / infile)

    env = dict(os.environ)
    model = runner_cfg.get("model", "gpt-5")
    if model:
        env["DEFAULT_MODEL"] = model
    api_base = runner_cfg.get("api_base")
    if api_base:
        env["OPENAI_API_BASE"] = api_base
        env["JUDGE_API_BASE"] = api_base
    if runner_cfg.get("force_stream"):
        env["FORCE_STREAM"] = "true"

    cmd = [
        "python", script,
        "--infile", infile,
        "--outfile", str(baseline_dir / "results.jsonl"),
        "--task_indices", task_indices,
        "--max_steps", str(runner_cfg.get("max_steps", 40)),
        "--concurrency", "1",
        "--direct_output_dir", str(tasks_dir),
    ]
    if model:
        cmd.extend(["--model", model])
    judge = runner_cfg.get("judge_model")
    if judge:
        cmd.extend(["--judge_model", judge])

    import subprocess
    log_path = baseline_dir / "run.log"
    try:
        with log_path.open("w") as log_file:
            subprocess.run(cmd, env=env, cwd=str(REPO_ROOT),
                           stdout=log_file, stderr=subprocess.STDOUT, timeout=7200)
    except Exception as e:
        logger.error(f"Baseline failed: {e}")

    # Parse results
    task_files = sorted(tasks_dir.glob("*.json"))
    correct = 0
    total = 0
    for tf in task_files:
        try:
            data = json.load(tf.open())
            total += 1
            if data.get("task_score", 0) > 0:
                correct += 1
        except Exception:
            pass

    score = correct / total if total > 0 else 0.0
    logger.info(f"  No-memory baseline: {correct}/{total} = {score:.3f}")

    return ExperimentResult(
        config=ArchConfig("no_memory", "none", "none", "none"),
        task_score=score,
        run_dir=str(baseline_dir),
        status="completed",
        num_tasks=total,
    )


def main():
    parser = argparse.ArgumentParser(description="Adaptive Architecture Search")
    parser.add_argument("--config", required=True, help="Search config JSON")
    parser.add_argument("--phase", default="all", choices=["all", "a", "b", "d"],
                        help="Which phase to run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-a", action="store_true",
                        help="Run Phase A modules in parallel (requires 4 GPUs)")
    parser.add_argument("--gpu", default="4", help="GPU device ID")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        search_config = json.load(f)

    # Override GPU
    search_config.setdefault("runner", {})["cuda_visible_devices"] = args.gpu

    search_name = search_config.get("search_name", "adaptive_search")
    output_root = Path(search_config.get("output_root",
                       "./experiments/results/architecture_search"))
    output_dir = output_root / search_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    with (output_dir / "search_config.json").open("w") as f:
        json.dump(search_config, f, indent=2, ensure_ascii=False)

    registry = RunRegistry(output_dir / "registry.json")
    runner = ExperimentRunner(
        search_config=search_config,
        repo_root=REPO_ROOT,
        output_root=output_dir,
        registry=registry,
        dry_run=args.dry_run,
    )

    logger.info(f"Search name: {search_name}")
    logger.info(f"Output: {output_dir}")

    # --- No-memory baseline ---
    baseline_result = None
    if search_config.get("phase_a", {}).get("run_no_memory_baseline", True):
        stage1 = search_config["dataset"].get("proxy_indices_stage1", "[level1]1-10")
        baseline_result = run_no_memory_baseline(runner, search_config, output_dir, stage1)

    # --- Phase A ---
    retained = None
    if args.phase in ("all", "a"):
        retained, module_results = adaptive_phase_a(runner, search_config, output_dir)
        logger.info(f"\nPhase A retained: {retained}")

    elif args.phase in ("b", "d"):
        # Load from checkpoint
        cp_path = output_dir / "checkpoint_a_adaptive.json"
        if not cp_path.exists():
            cp_path = output_dir / "checkpoint_a.json"
        if cp_path.exists():
            with cp_path.open() as f:
                cp = json.load(f)
            retained = cp["retained"]
            logger.info(f"Loaded Phase A checkpoint: {retained}")
        else:
            logger.error("No Phase A checkpoint found. Run Phase A first.")
            return

    # --- Phase B ---
    elites = None
    if args.phase in ("all", "b"):
        if retained is None:
            logger.error("No retained candidates from Phase A")
            return
        elites = adaptive_phase_b(runner, retained, search_config, output_dir, baseline_result)

    elif args.phase == "d":
        cp_path = output_dir / "checkpoint_b_adaptive.json"
        if not cp_path.exists():
            cp_path = output_dir / "checkpoint_b.json"
        if cp_path.exists():
            with cp_path.open() as f:
                cp = json.load(f)
            elites = [ExperimentResult.from_dict(d) for d in cp["elites"]]
            logger.info(f"Loaded Phase B checkpoint: {len(elites)} elites")
        else:
            logger.error("No Phase B checkpoint found. Run Phase B first.")
            return

    # --- Phase D ---
    if args.phase in ("all", "d"):
        if elites is None:
            logger.error("No elites from Phase B")
            return
        phase_d(runner, elites, search_config, output_dir)

    # --- Final summary ---
    logger.info("\n" + "=" * 60)
    logger.info("ADAPTIVE SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Results: {output_dir}")

    if baseline_result:
        logger.info(f"  No-memory baseline: {baseline_result.task_score:.3f}")
    if elites:
        for i, e in enumerate(elites[:3], 1):
            logger.info(f"  #{i} {e.config.short_name}: {e.task_score:.3f}")


if __name__ == "__main__":
    main()
