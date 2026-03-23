#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./experiments/results/gaia_siliconfriend_eval_${TIMESTAMP}"
INFILE="./data/gaia/validation/metadata.jsonl"
TASK_INDICES="${TASK_INDICES:-[level2]1-43}"
MAX_STEPS="${MAX_STEPS:-40}"
TOKEN_BUDGET="${TOKEN_BUDGET:-8192}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-1}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-8}"
PROMPTS_TYPE="${PROMPTS_TYPE:-default}"

SILICON_NAME="gaia_siliconfriend_baseline"
BASELINE_NAME="gaia_no_memory_baseline"

GPT_MODEL="${GPT_MODEL:-gpt-5}"
GPT_BASE="${GPT_BASE:-https://api-vip.codex-for.me/v1}"

eval "$(conda shell.bash hook)"
conda activate evolvelab

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

export DEFAULT_MODEL="${GPT_MODEL}"
export DEFAULT_JUDGE_MODEL="${GPT_MODEL}"
export OPENAI_API_BASE="${GPT_BASE}"
export JUDGE_API_BASE="${GPT_BASE}"
export OPENAI_BASE_URL="${GPT_BASE}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export SILICONFRIEND_EMBEDDING_DEVICE="${SILICONFRIEND_EMBEDDING_DEVICE:-cuda:0}"

if [ ! -f "${INFILE}" ]; then
    echo "[ERROR] GAIA dataset not found: ${INFILE}"
    exit 1
fi

mkdir -p "${EVAL_DIR}"
echo "config,tasks,correct,elapsed_seconds,exit_code" > "${EVAL_DIR}/summary.csv"

echo "============================================================"
echo "  GAIA SiliconFriend Evaluation — Baseline vs No Memory"
echo "============================================================"
echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output dir:  ${EVAL_DIR}"
echo "  Tasks:       ${TASK_INDICES}"
echo "  Model:       ${GPT_MODEL}"
echo "  API:         ${GPT_BASE}"
echo "  GPU:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  SF Device:   ${SILICONFRIEND_EMBEDDING_DEVICE}"
echo "============================================================"
echo ""

run_experiment() {
    local name="$1"
    local memory_provider="$2"
    shift 2

    local run_dir="${EVAL_DIR}/${name}"
    local log_file="${run_dir}/run.log"
    local start_time
    local end_time
    local elapsed
    local total
    local correct
    local exit_code

    mkdir -p "${run_dir}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${name} (model=${GPT_MODEL})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    start_time=$(date +%s)

    if [ "${memory_provider}" = "none" ]; then
        env \
            DEFAULT_MODEL="${GPT_MODEL}" \
            DEFAULT_JUDGE_MODEL="${GPT_MODEL}" \
            OPENAI_API_BASE="${GPT_BASE}" \
            JUDGE_API_BASE="${GPT_BASE}" \
            OPENAI_BASE_URL="${GPT_BASE}" \
            FORCE_STREAM="1" \
        python run_flash_searcher_mm_gaia.py \
            --infile "${INFILE}" \
            --outfile "${run_dir}/results.jsonl" \
            --task_indices "${TASK_INDICES}" \
            --seed "${SEED}" \
            --token_budget "${TOKEN_BUDGET}" \
            --judge_model "${GPT_MODEL}" \
            --concurrency "${CONCURRENCY}" \
            --summary_interval "${SUMMARY_INTERVAL}" \
            --prompts_type "${PROMPTS_TYPE}" \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${run_dir}/tasks" \
            2>&1 | tee "${log_file}"
        exit_code=${PIPESTATUS[0]}
    else
        env \
            DEFAULT_MODEL="${GPT_MODEL}" \
            DEFAULT_JUDGE_MODEL="${GPT_MODEL}" \
            OPENAI_API_BASE="${GPT_BASE}" \
            JUDGE_API_BASE="${GPT_BASE}" \
            OPENAI_BASE_URL="${GPT_BASE}" \
            FORCE_STREAM="1" \
            "$@" \
        python run_flash_searcher_mm_gaia.py \
            --infile "${INFILE}" \
            --outfile "${run_dir}/results.jsonl" \
            --task_indices "${TASK_INDICES}" \
            --memory_provider "${memory_provider}" \
            --enable_memory_evolution \
            --shared_memory_provider \
            --seed "${SEED}" \
            --token_budget "${TOKEN_BUDGET}" \
            --judge_model "${GPT_MODEL}" \
            --concurrency "${CONCURRENCY}" \
            --summary_interval "${SUMMARY_INTERVAL}" \
            --prompts_type "${PROMPTS_TYPE}" \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${run_dir}/tasks" \
            2>&1 | tee "${log_file}"
        exit_code=${PIPESTATUS[0]}
    fi

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    total=$(find "${run_dir}/tasks" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
    if compgen -G "${run_dir}/tasks/*.json" > /dev/null; then
        correct=$(grep -rl '"judgement".*"[Cc]orrect"' "${run_dir}/tasks"/*.json 2>/dev/null | wc -l | tr -d ' ')
    else
        correct=0
    fi

    echo "  [${name}] Tasks: ${total}, Correct: ${correct}, Time: ${elapsed}s, Exit: ${exit_code}"
    echo "${name},${total},${correct},${elapsed},${exit_code}" >> "${EVAL_DIR}/summary.csv"

    return "${exit_code}"
}

FAIL=0

run_experiment \
    "${SILICON_NAME}" \
    "siliconfriend" \
    SILICONFRIEND_STORE_DIR="${EVAL_DIR}/storage_siliconfriend" \
    SILICONFRIEND_MEMORY_FILE="gaia_memory.json" \
    SILICONFRIEND_INDEX_DIR="${EVAL_DIR}/storage_siliconfriend/index" \
    SILICONFRIEND_USER_NAME="gaia_eval" \
    SILICONFRIEND_TOP_K="${SILICONFRIEND_TOP_K:-3}" \
    SILICONFRIEND_LANGUAGE="en" \
    SILICONFRIEND_EMBEDDING_MODEL="${SILICONFRIEND_EMBEDDING_MODEL:-minilm-l6}" \
    SILICONFRIEND_EMBEDDING_DEVICE="${SILICONFRIEND_EMBEDDING_DEVICE}" \
    SILICONFRIEND_RESPONSE_MODE="${SILICONFRIEND_RESPONSE_MODE:-trajectory_summary}" \
    > "${EVAL_DIR}/siliconfriend.stdout.log" 2>&1 &
PID_SF=$!

run_experiment \
    "${BASELINE_NAME}" \
    "none" \
    > "${EVAL_DIR}/no_memory.stdout.log" 2>&1 &
PID_BASE=$!

echo "siliconfriend pid=${PID_SF}"
echo "no_memory pid=${PID_BASE}"

wait "${PID_SF}" || FAIL=1
wait "${PID_BASE}" || FAIL=1

echo ""
echo "Results: ${EVAL_DIR}"

if [ "${FAIL}" -ne 0 ]; then
    echo "At least one experiment failed."
    exit 1
fi

echo "Both experiments completed."
