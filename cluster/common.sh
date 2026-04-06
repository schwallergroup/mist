#!/bin/bash

if [[ -z "${MIST_REPO_ROOT:-}" ]]; then
    export MIST_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "${MIST_REPO_ROOT}/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "${MIST_REPO_ROOT}/.env.local"
fi

if [[ -n "${MIST_ENV_FILE:-}" && -f "${MIST_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${MIST_ENV_FILE}"
fi

export MIST_SAMPLING_PARAMS_DIR="${MIST_SAMPLING_PARAMS_DIR:-${MIST_REPO_ROOT}/sampling_params}"
export MIST_MODEL_REGISTRY="${MIST_MODEL_REGISTRY:-${MIST_REPO_ROOT}/model_paths.txt}"
export MIST_DATA_DIR="${MIST_DATA_DIR:-${MIST_REPO_ROOT}/data}"
export MIST_CACHE_DIR="${MIST_CACHE_DIR:-${MIST_REPO_ROOT}/.cache}"
export MIST_OUTPUT_DIR="${MIST_OUTPUT_DIR:-${MIST_REPO_ROOT}/output}"
export MIST_LOG_DIR="${MIST_LOG_DIR:-${MIST_REPO_ROOT}/slurm_logs}"
export MIST_CHECKPOINT_DIR="${MIST_CHECKPOINT_DIR:-${MIST_CACHE_DIR}/checkpoints}"
export MIST_HF_HOME="${MIST_HF_HOME:-${MIST_CACHE_DIR}/huggingface}"
export MIST_WANDB_API_KEY_FILE="${MIST_WANDB_API_KEY_FILE:-${MIST_REPO_ROOT}/wandb_api_key.txt}"

mist_expand_path() {
    local raw_path="$1"
    eval "printf '%s\n' \"$raw_path\""
}

mist_resolve_model_path() {
    local model_id="$1"
    local model_path
    model_path=$(grep "^${model_id}:" "${MIST_MODEL_REGISTRY}" | cut -d':' -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')

    if [[ -z "${model_path}" ]]; then
        echo "Unknown model id '${model_id}' in ${MIST_MODEL_REGISTRY}" >&2
        return 1
    fi

    mist_expand_path "${model_path}"
}
