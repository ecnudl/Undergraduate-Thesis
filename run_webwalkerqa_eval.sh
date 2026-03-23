#!/bin/bash
# ============================================================
# WebWalkerQA Evaluation — 4 Memory Configs in Parallel
# ============================================================
# Runs 4 memory configurations on WebWalkerQA in parallel,
# each pinned to a separate GPU (4, 5, 6, 7).
#
# Configurations:
#   0. no_memory                 — baseline (no memory provider)
#   1. agent_kb                  — Agent-KB provider
#   2. workflow_shortcut_hybrid  — workflow+shortcut + HybridRetriever + JsonStorage + json_full management
#   3. g_memory                  — tip+insight + GraphStore + GraphRetriever
#
# Usage:
#   chmod +x run_webwalkerqa_eval.sh
#   ./run_webwalkerqa_eval.sh
#
# Override:
#   TASK_RANGE="1-50"  ./run_webwalkerqa_eval.sh     # custom range
#   SAMPLE_NUM=50      ./run_webwalkerqa_eval.sh     # first N tasks
#   MAX_STEPS=30       ./run_webwalkerqa_eval.sh     # fewer steps
#   GPUS="4,5,6,7"    ./run_webwalkerqa_eval.sh     # custom GPU ids
#   SEQUENTIAL=1       ./run_webwalkerqa_eval.sh     # run sequentially
# ============================================================

set -euo pipefail

# ---------- Configurable parameters ----------
TASK_RANGE="${TASK_RANGE:-}"              # e.g., "1-170"; empty = use SAMPLE_NUM or all
SAMPLE_NUM="${SAMPLE_NUM:-}"              # e.g., 170; empty = all tasks
MAX_STEPS="${MAX_STEPS:-40}"
GPUS="${GPUS:-4,5,6,7}"                   # 4 GPUs for 4 configs
SEQUENTIAL="${SEQUENTIAL:-0}"             # set 1 to run sequentially

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./experiments/webwalkerqa_eval_${TIMESTAMP}"
INFILE="./data/webwalkerqa/webwalkerqa_stratified_150.jsonl"

# ---------- Environment ----------
eval "$(conda shell.bash hook)"
conda activate memevolve

source .env 2>/dev/null || true

# ---------- Proxy (SSH tunnel: -R 15236:127.0.0.1:7897) ----------
export http_proxy="http://127.0.0.1:15236"
export https_proxy="http://127.0.0.1:15236"
export HTTP_PROXY="http://127.0.0.1:15236"
export HTTPS_PROXY="http://127.0.0.1:15236"
export no_proxy="localhost,127.0.0.1"

# ---------- LLM API (codex-for.me via proxy) ----------
LLM_API_BASE="${LLM_API_BASE:-https://api-vip.codex-for.me/v1}"
LLM_API_KEY="${OPENAI_API_KEY}"
LLM_MODEL="${DEFAULT_MODEL:-qwen3-max}"
export OPENAI_API_BASE="${LLM_API_BASE}"
export OPENAI_BASE_URL="${LLM_API_BASE}"

# ---------- Verify dataset ----------
if [ ! -f "${INFILE}" ]; then
    echo "[WARN] Default dataset not found: ${INFILE}"
    # Try fallbacks
    for fallback in "./data/webwalkerqa/webwalkerqa_subset_170.jsonl" \
                    "./data/webwalkerqa/webwalkerqa_main.jsonl"; do
        if [ -f "${fallback}" ]; then
            INFILE="${fallback}"
            echo "       Using fallback: ${INFILE}"
            break
        fi
    done
    if [ ! -f "${INFILE}" ]; then
        echo "[ERROR] No WebWalkerQA dataset found."
        echo "        Run: python download_webwalkerqa.py && python sample_webwalkerqa_stratified.py"
        exit 1
    fi
fi

# Parse GPU list into array
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "[ERROR] Need at least 4 GPUs. Got: ${GPUS}"
    exit 1
fi

# ---------- Build task selection args ----------
TASK_ARGS=""
if [ -n "${TASK_RANGE}" ]; then
    TASK_ARGS="--task_indices ${TASK_RANGE}"
elif [ -n "${SAMPLE_NUM}" ]; then
    TASK_ARGS="--sample_num ${SAMPLE_NUM}"
fi

# ---------- Print header ----------
echo "============================================================"
echo "  WebWalkerQA Evaluation — 4 Configs Parallel"
echo "============================================================"
echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:     ${INFILE}"
echo "  Task args:   ${TASK_ARGS:-all tasks}"
echo "  Max steps:   ${MAX_STEPS}"
echo "  GPUs:        ${GPUS}"
echo "  Output dir:  ${EVAL_DIR}"
echo "  Model:       ${LLM_MODEL}"
echo "  API:         ${LLM_API_BASE}"
echo "  Proxy:       http://127.0.0.1:15236"
echo "  Mode:        $([ "${SEQUENTIAL}" = "1" ] && echo "sequential" || echo "parallel")"
echo "============================================================"
echo ""
echo "  Config map:"
echo "    GPU ${GPU_ARRAY[0]} → no_memory              (pure LLM baseline)"
echo "    GPU ${GPU_ARRAY[1]} → agent_kb               (Agent-KB provider)"
echo "    GPU ${GPU_ARRAY[2]} → workflow_shortcut_hybrid (workflow+shortcut + hybrid retrieval + json_full mgmt)"
echo "    GPU ${GPU_ARRAY[3]} → g_memory               (tip+insight + graph storage + graph retrieval)"
echo ""

mkdir -p "${EVAL_DIR}"

# ---------- Initialize summary ----------
echo "config,tasks,correct,elapsed_seconds" > "${EVAL_DIR}/summary.csv"

# ---------- Helper: run a single config ----------
# Uses env to force API settings, overriding load_dotenv(override=True) in Python
run_config() {
    local name="$1"
    local gpu_id="$2"
    shift 2
    # Remaining args are env var overrides + python command

    local run_dir="${EVAL_DIR}/${name}"
    local storage_dir="${EVAL_DIR}/storage_${name}"
    local log_file="${run_dir}/run.log"

    mkdir -p "${run_dir}"
    rm -rf "${storage_dir}"
    mkdir -p "${storage_dir}"

    echo "[$(date '+%H:%M:%S')] Starting ${name} on GPU ${gpu_id}..."

    local start_time=$(date +%s)

    # Force API settings via env to override .env's load_dotenv(override=True)
    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        OPENAI_API_BASE="${LLM_API_BASE}" \
        OPENAI_BASE_URL="${LLM_API_BASE}" \
        OPENAI_API_KEY="${LLM_API_KEY}" \
        FORCE_STREAM="1" \
        "$@" 2>&1 | tee "${log_file}"

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Count results
    local total=$(ls "${run_dir}"/*.json 2>/dev/null | grep -cv results 2>/dev/null || echo 0)
    local correct=$(grep -rl '"judgement".*"[Cc]orrect"' "${run_dir}"/*.json 2>/dev/null | wc -l || echo 0)

    echo "[$(date '+%H:%M:%S')] ${name} done: Tasks=${total}, Correct=${correct}, Time=${elapsed}s"
    echo "${name},${total},${correct},${elapsed}" >> "${EVAL_DIR}/summary.csv"
}

# ---------- Config 0: no_memory ----------
run_no_memory() {
    run_config "no_memory" "${GPU_ARRAY[0]}" \
        python run_flash_searcher_webwalkerqa.py \
            --infile "${INFILE}" \
            --outfile "${EVAL_DIR}/no_memory/results.jsonl" \
            ${TASK_ARGS} \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${EVAL_DIR}/no_memory"
}

# ---------- Config 1: agent_kb ----------
run_agent_kb() {
    AGENT_KB_DATABASE_PATH="${EVAL_DIR}/storage_agent_kb/agent_kb_database.json" \
    run_config "agent_kb" "${GPU_ARRAY[1]}" \
        python run_flash_searcher_webwalkerqa.py \
            --infile "${INFILE}" \
            --outfile "${EVAL_DIR}/agent_kb/results.jsonl" \
            ${TASK_ARGS} \
            --memory_provider agent_kb \
            --enable_memory_evolution \
            --shared_memory_provider \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${EVAL_DIR}/agent_kb"
}

# ---------- Config 2: workflow_shortcut_hybrid ----------
# Extraction: workflow + shortcut
# Storage: json
# Retrieval: hybrid (0.7 semantic + 0.3 keyword)
# Management: json_full preset
run_workflow_shortcut_hybrid() {
    MODULAR_ENABLED_PROMPTS="workflow,shortcut" \
    MODULAR_STORAGE_TYPE="json" \
    MODULAR_RETRIEVER_TYPE="hybrid" \
    MODULAR_RETRIEVER_CONFIG='{"weights":{"SemanticRetriever":0.7,"KeywordRetriever":0.3}}' \
    MODULAR_STORAGE_DIR="${EVAL_DIR}/storage_workflow_shortcut_hybrid" \
    MODULAR_MANAGEMENT_ENABLED="true" \
    MODULAR_MANAGEMENT_PRESET="json_full" \
    run_config "workflow_shortcut_hybrid" "${GPU_ARRAY[2]}" \
        python run_flash_searcher_webwalkerqa.py \
            --infile "${INFILE}" \
            --outfile "${EVAL_DIR}/workflow_shortcut_hybrid/results.jsonl" \
            ${TASK_ARGS} \
            --memory_provider modular \
            --enable_memory_evolution \
            --shared_memory_provider \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${EVAL_DIR}/workflow_shortcut_hybrid"
}

# ---------- Config 3: g_memory ----------
# Extraction: tip + insight
# Storage: graph (LLM-powered graph structure)
# Retrieval: graph (seed_k=3, max_hops=1, decay=0.7)
run_g_memory() {
    MODULAR_ENABLED_PROMPTS="tip,insight" \
    MODULAR_STORAGE_TYPE="graph" \
    MODULAR_RETRIEVER_TYPE="graph" \
    MODULAR_RETRIEVER_CONFIG='{"seed_k":3,"max_hops":1,"decay_factor":0.7}' \
    MODULAR_STORAGE_DIR="${EVAL_DIR}/storage_g_memory" \
    MODULAR_MANAGEMENT_ENABLED="true" \
    MODULAR_MANAGEMENT_PRESET="graph_full" \
    run_config "g_memory" "${GPU_ARRAY[3]}" \
        python run_flash_searcher_webwalkerqa.py \
            --infile "${INFILE}" \
            --outfile "${EVAL_DIR}/g_memory/results.jsonl" \
            ${TASK_ARGS} \
            --memory_provider modular \
            --enable_memory_evolution \
            --shared_memory_provider \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${EVAL_DIR}/g_memory"
}

# ---------- Execute ----------
if [ "${SEQUENTIAL}" = "1" ]; then
    echo "Running sequentially..."
    run_no_memory
    run_agent_kb
    run_workflow_shortcut_hybrid
    run_g_memory
else
    echo "Running 4 configs in parallel..."
    run_no_memory &
    PID_0=$!

    run_agent_kb &
    PID_1=$!

    run_workflow_shortcut_hybrid &
    PID_2=$!

    run_g_memory &
    PID_3=$!

    echo ""
    echo "  PIDs: no_memory=${PID_0}, agent_kb=${PID_1}, workflow_shortcut_hybrid=${PID_2}, g_memory=${PID_3}"
    echo "  Waiting for all to complete..."
    echo ""

    # Wait for each and report
    FAIL=0
    for pid_var in PID_0 PID_1 PID_2 PID_3; do
        pid=${!pid_var}
        if wait "${pid}"; then
            echo "  [OK] PID ${pid} (${pid_var}) completed successfully"
        else
            echo "  [FAIL] PID ${pid} (${pid_var}) exited with error"
            FAIL=$((FAIL + 1))
        fi
    done

    if [ "${FAIL}" -gt 0 ]; then
        echo ""
        echo "  WARNING: ${FAIL} config(s) failed. Check logs in ${EVAL_DIR}/<config>/run.log"
    fi
fi

# ---------- Final comparison report ----------
echo ""
echo "============================================================"
echo "  WEBWALKERQA EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "  Results directory: ${EVAL_DIR}"
echo ""

if [ -f "${EVAL_DIR}/summary.csv" ]; then
    echo "  ┌────────────────────────────────┬───────┬─────────┬──────────┬──────────┐"
    echo "  │ Configuration                  │ Tasks │ Correct │ Accuracy │ Time (s) │"
    echo "  ├────────────────────────────────┼───────┼─────────┼──────────┼──────────┤"

    tail -n +2 "${EVAL_DIR}/summary.csv" | sort | while IFS=',' read -r name tasks correct elapsed; do
        if [ "${tasks}" -gt 0 ] 2>/dev/null; then
            accuracy=$(echo "scale=1; ${correct} * 100 / ${tasks}" | bc 2>/dev/null || echo "N/A")
        else
            accuracy="N/A"
        fi
        printf "  │ %-30s │ %5s │ %7s │ %6s%% │ %8s │\n" \
            "${name}" "${tasks}" "${correct}" "${accuracy}" "${elapsed}"
    done

    echo "  └────────────────────────────────┴───────┴─────────┴──────────┴──────────┘"
fi

echo ""
echo "  Detailed results:  ${EVAL_DIR}/<config_name>/"
echo "  Per-config logs:   ${EVAL_DIR}/<config_name>/run.log"
echo "  Summary CSV:       ${EVAL_DIR}/summary.csv"
echo "============================================================"
