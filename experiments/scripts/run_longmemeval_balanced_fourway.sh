#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)"
conda activate evolvelab

GPT_MODEL="${GPT_MODEL:-gpt-5}"
GPT_BASE="${GPT_BASE:-https://api-vip.codex-for.me/v1}"

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
export MEMORY_EMBEDDING_DEVICE="${MEMORY_EMBEDDING_DEVICE:-cpu}"
export SILICONFRIEND_EMBEDDING_DEVICE="${SILICONFRIEND_EMBEDDING_DEVICE:-cpu}"

CONFIGS=(
    "experiments/configs/longmemeval_s_balanced_no_memory_gpt5.json"
    "experiments/configs/longmemeval_s_balanced_tip_insight_gpt5.json"
    "experiments/configs/longmemeval_s_balanced_graph_gpt5.json"
    "experiments/configs/longmemeval_s_balanced_siliconfriend_gpt5.json"
)

LOG_DIR="./experiments/results/longmemeval_balanced_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Logs: ${LOG_DIR}"
echo "Model: ${GPT_MODEL}"
echo "API: ${GPT_BASE}"
echo "Running API preflight..."
python experiments/scripts/check_openai_preflight.py \
    --api-base "${OPENAI_API_BASE}" \
    --model "${DEFAULT_MODEL}" \
    2>&1 | tee "${LOG_DIR}/preflight.log"

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "${cfg}" .json)"
    echo "============================================================"
    echo "Running ${name}"
    echo "============================================================"
    python experiments/scripts/run_experiment.py --config "${cfg}" 2>&1 | tee "${LOG_DIR}/${name}.log"
done

echo "All LongMemEval balanced runs completed."
