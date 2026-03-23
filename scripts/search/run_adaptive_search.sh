#!/bin/bash
# ============================================================
# Adaptive Architecture Search — with tie-breaking expansion
# ============================================================
# Runs hierarchical memory architecture search with adaptive
# proxy set expansion: starts with 10 tasks, if top configs
# are tied (spread < threshold), automatically expands to 20.
#
# Improvements over previous search:
#   1. Larger initial proxy set (10 vs 5)
#   2. Adaptive expansion to 20 if configs are tied
#   3. Uses updated memory gating (gate_threshold + min_relevance)
#   4. Separate holdout set for Phase D validation
#
# Usage:
#   ./run_adaptive_search.sh                    # full run on GPU 4
#   GPU=5 ./run_adaptive_search.sh              # use GPU 5
#   DRY_RUN=1 ./run_adaptive_search.sh          # dry run (no real experiments)
#   PHASE=a ./run_adaptive_search.sh            # run only Phase A
#   PARALLEL_A=1 ./run_adaptive_search.sh       # parallel Phase A (4 GPUs)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU="${GPU:-4}"
DRY_RUN="${DRY_RUN:-0}"
PHASE="${PHASE:-all}"
PARALLEL_A="${PARALLEL_A:-0}"
CONFIG_FILE="${CONFIG_FILE:-experiments/configs/architecture_search/adaptive_search.json}"

# ---------- Environment ----------
eval "$(conda shell.bash hook)"
conda activate evolvelab

source .env 2>/dev/null || true

# ---------- Proxy ----------
export http_proxy="http://127.0.0.1:15236"
export https_proxy="http://127.0.0.1:15236"
export HTTP_PROXY="http://127.0.0.1:15236"
export HTTPS_PROXY="http://127.0.0.1:15236"
export no_proxy="localhost,127.0.0.1"

# ---------- Header ----------
echo "============================================================"
echo "  Adaptive Architecture Search"
echo "============================================================"
echo "  Time:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Config:   ${CONFIG_FILE}"
echo "  GPU:      ${GPU}"
echo "  Phase:    ${PHASE}"
echo "  Dry run:  ${DRY_RUN}"
echo "  Model:    ${DEFAULT_MODEL:-gpt-5}"
echo "  API:      ${OPENAI_API_BASE}"
echo "============================================================"
echo ""

# ---------- Update GPU in config dynamically ----------
CUDA_VISIBLE_DEVICES="${GPU}" \
python -u experiments/scripts/run_adaptive_search.py \
    --config "${CONFIG_FILE}" \
    --phase "${PHASE}" \
    $([ "${DRY_RUN}" = "1" ] && echo "--dry-run") \
    $([ "${PARALLEL_A}" = "1" ] && echo "--parallel-a") \
    --gpu "${GPU}"
