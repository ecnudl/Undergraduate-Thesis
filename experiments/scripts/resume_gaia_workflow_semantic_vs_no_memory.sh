#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)"
conda activate memevolve

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

export DEFAULT_MODEL="${DEFAULT_MODEL:-gpt-5}"
export DEFAULT_JUDGE_MODEL="${DEFAULT_JUDGE_MODEL:-${DEFAULT_MODEL}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api-vip.codex-for.me/v1}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-${OPENAI_API_BASE}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
export FORCE_STREAM="${FORCE_STREAM:-1}"

python experiments/scripts/resume_gaia_workflow_semantic_vs_no_memory.py "$@"
