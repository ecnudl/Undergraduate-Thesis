#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./experiments/results/gaia_triple_eval_${TIMESTAMP}"
INFILE="./data/gaia/validation/metadata.jsonl"
MAX_STEPS="${MAX_STEPS:-40}"
TOKEN_BUDGET="${TOKEN_BUDGET:-8192}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-1}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-8}"
PROMPTS_TYPE="${PROMPTS_TYPE:-default}"

TIP_NAME="gaia_l2_half_gpt5_json_semantic"
TIP_TASK_INDICES="[level2]1-43"
TIP_MEMORY_PROVIDER="modular"

BASELINE_NAME="gaia_l1_l2_half_no_memory_gpt5"
BASELINE_TASK_INDICES="[level1]1-26 [level2]1-43"

SILICON_NAME="gaia_l2_half_siliconfriend_gpt5"
SILICON_TASK_INDICES="[level2]1-43"

GPT_MODEL="${GPT_MODEL:-gpt-5}"
GPT_BASE="${GPT_BASE:-https://api-vip.codex-for.me/v1}"

eval "$(conda shell.bash hook)"
conda activate memevolve

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
export MEMORY_EMBEDDING_DEVICE="${MEMORY_EMBEDDING_DEVICE:-cuda:0}"
export SILICONFRIEND_EMBEDDING_DEVICE="${SILICONFRIEND_EMBEDDING_DEVICE:-cuda:0}"

if [ ! -f "${INFILE}" ]; then
    echo "[ERROR] GAIA dataset not found: ${INFILE}"
    exit 1
fi

mkdir -p "${EVAL_DIR}"
echo "config,tasks,correct,elapsed_seconds,exit_code" > "${EVAL_DIR}/summary.csv"

echo "============================================================"
echo "  GAIA Triple Evaluation — Tip+Insight vs No Memory vs SF"
echo "============================================================"
echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output dir:  ${EVAL_DIR}"
echo "  Model:       ${GPT_MODEL}"
echo "  API:         ${GPT_BASE}"
echo "  GPU:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Embedding:   modular=${MEMORY_EMBEDDING_DEVICE}, siliconfriend=${SILICONFRIEND_EMBEDDING_DEVICE}"
echo "============================================================"
echo ""

run_experiment() {
    local name="$1"
    local task_indices="$2"
    local memory_provider="$3"
    shift 3

    local run_dir="${EVAL_DIR}/${name}"
    local log_file="${run_dir}/run.log"
    local storage_dir="${EVAL_DIR}/storage_${name}"
    local start_time
    local end_time
    local elapsed
    local total
    local correct
    local exit_code

    mkdir -p "${run_dir}" "${storage_dir}"

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
            --task_indices "${task_indices}" \
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
            MODULAR_STORAGE_DIR="${storage_dir}" \
            "$@" \
        python run_flash_searcher_mm_gaia.py \
            --infile "${INFILE}" \
            --outfile "${run_dir}/results.jsonl" \
            --task_indices "${task_indices}" \
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
    "${TIP_NAME}" \
    "${TIP_TASK_INDICES}" \
    "${TIP_MEMORY_PROVIDER}" \
    MODULAR_ENABLED_PROMPTS="tip,insight" \
    MODULAR_STORAGE_TYPE="json" \
    MODULAR_RETRIEVER_TYPE="semantic" \
    MODULAR_MANAGEMENT_ENABLED="false" \
    MEMORY_EMBEDDING_DEVICE="${MEMORY_EMBEDDING_DEVICE}" \
    > "${EVAL_DIR}/tip_insight.stdout.log" 2>&1 &
PID_TIP=$!

run_experiment \
    "${BASELINE_NAME}" \
    "${BASELINE_TASK_INDICES}" \
    "none" \
    > "${EVAL_DIR}/no_memory.stdout.log" 2>&1 &
PID_BASE=$!

run_experiment \
    "${SILICON_NAME}" \
    "${SILICON_TASK_INDICES}" \
    "siliconfriend" \
    SILICONFRIEND_STORE_DIR="${EVAL_DIR}/storage_${SILICON_NAME}" \
    SILICONFRIEND_MEMORY_FILE="gaia_memory.json" \
    SILICONFRIEND_INDEX_DIR="${EVAL_DIR}/storage_${SILICON_NAME}/index" \
    SILICONFRIEND_USER_NAME="gaia_eval" \
    SILICONFRIEND_TOP_K="${SILICONFRIEND_TOP_K:-3}" \
    SILICONFRIEND_LANGUAGE="en" \
    SILICONFRIEND_EMBEDDING_MODEL="${SILICONFRIEND_EMBEDDING_MODEL:-minilm-l6}" \
    SILICONFRIEND_EMBEDDING_DEVICE="${SILICONFRIEND_EMBEDDING_DEVICE}" \
    SILICONFRIEND_RESPONSE_MODE="${SILICONFRIEND_RESPONSE_MODE:-trajectory_summary}" \
    > "${EVAL_DIR}/siliconfriend.stdout.log" 2>&1 &
PID_SF=$!

echo "tip_insight pid=${PID_TIP}"
echo "no_memory pid=${PID_BASE}"
echo "siliconfriend pid=${PID_SF}"

wait "${PID_TIP}" || FAIL=1
wait "${PID_BASE}" || FAIL=1
wait "${PID_SF}" || FAIL=1

echo ""
echo "Results: ${EVAL_DIR}"

if [ "${FAIL}" -ne 0 ]; then
    echo "At least one experiment failed."
    exit 1
fi

echo "All three experiments completed."
