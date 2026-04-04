#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ID="${PROJECT_ID:-fluid-house-463510-i7}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-mist-scs-$(date +%Y%m%d-%H%M%S)}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-570}"
IMAGE_NAME="${IMAGE_NAME:-}"
REPO_URL="${REPO_URL:-https://github.com/schwallergroup/mist.git}"
BRANCH="${BRANCH:-main}"
NETWORK="${NETWORK:-mist-demo-network}"
SUBNET="${SUBNET:-mist-demo-subnet}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B}"
FIGSHARE_ARTICLE_ID="${FIGSHARE_ARTICLE_ID:-29132657}"
FIGSHARE_DATASET_FILENAME="${FIGSHARE_DATASET_FILENAME:-datasets.zip}"
SCS_NUM_ROWS="${SCS_NUM_ROWS:-10000}"
SCS_DATASET_BASENAME="${SCS_DATASET_BASENAME:-CRLLM-PubChem-compounds1M.csv}"
SCS_TENSOR_PARALLEL_SIZE="${SCS_TENSOR_PARALLEL_SIZE:-1}"
SCS_OUTPUT_DIR="${SCS_OUTPUT_DIR:-/var/log/mist-scs}"
INSTANCE_TAGS="${INSTANCE_TAGS:-}"

NETWORK_ARGS=(--network "${NETWORK}")
if [[ -n "${SUBNET}" ]]; then
  NETWORK_ARGS+=(--subnet "${SUBNET}")
fi

METADATA_ITEMS=(
  "REPO_URL=${REPO_URL}"
  "BRANCH=${BRANCH}"
  "MODEL_ID=${MODEL_ID}"
  "FIGSHARE_ARTICLE_ID=${FIGSHARE_ARTICLE_ID}"
  "FIGSHARE_DATASET_FILENAME=${FIGSHARE_DATASET_FILENAME}"
  "SCS_NUM_ROWS=${SCS_NUM_ROWS}"
  "SCS_DATASET_BASENAME=${SCS_DATASET_BASENAME}"
  "SCS_TENSOR_PARALLEL_SIZE=${SCS_TENSOR_PARALLEL_SIZE}"
  "SCS_OUTPUT_DIR=${SCS_OUTPUT_DIR}"
)
METADATA_STRING="$(IFS=,; echo "${METADATA_ITEMS[*]}")"

cat <<EOF
Creating GPU VM with:
  project: ${PROJECT_ID}
  zone: ${ZONE}
  instance: ${INSTANCE_NAME}
  machine: ${MACHINE_TYPE}
  accelerator: ${GPU_TYPE}
  image: ${IMAGE_PROJECT}/${IMAGE_NAME:-$IMAGE_FAMILY}
  network: ${NETWORK}
  model: ${MODEL_ID}
  rows: ${SCS_NUM_ROWS}
EOF

IMAGE_ARGS=(--image-project "${IMAGE_PROJECT}")
if [[ -n "${IMAGE_NAME}" ]]; then
  IMAGE_ARGS+=(--image "${IMAGE_NAME}")
else
  IMAGE_ARGS+=(--image-family "${IMAGE_FAMILY}")
fi

ACCELERATOR_ARGS=()
if [[ -n "${GPU_TYPE}" && "${GPU_TYPE}" != "integrated" && "${GPU_TYPE}" != "none" ]]; then
  ACCELERATOR_ARGS+=(--accelerator "type=${GPU_TYPE},count=${GPU_COUNT}")
fi

TAG_ARGS=()
if [[ -n "${INSTANCE_TAGS}" ]]; then
  TAG_ARGS+=(--tags "${INSTANCE_TAGS}")
fi

CREATE_CMD=(
  gcloud compute instances create "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --machine-type "${MACHINE_TYPE}"
  --boot-disk-size "${BOOT_DISK_SIZE}"
  "${IMAGE_ARGS[@]}"
  --scopes cloud-platform
  --maintenance-policy TERMINATE
  "${NETWORK_ARGS[@]}"
  --metadata "${METADATA_STRING}"
  --metadata-from-file "startup-script=${SCRIPT_DIR}/gcp_scs_startup.sh"
)

if [[ ${#ACCELERATOR_ARGS[@]} -gt 0 ]]; then
  CREATE_CMD+=("${ACCELERATOR_ARGS[@]}")
fi

if [[ ${#TAG_ARGS[@]} -gt 0 ]]; then
  CREATE_CMD+=("${TAG_ARGS[@]}")
fi

"${CREATE_CMD[@]}"

cat <<EOF

Instance created. Useful follow-up commands:

  gcloud compute instances tail-serial-port-output "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute ssh "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute scp "${INSTANCE_NAME}:/var/log/mist-scs/summary.json" ./mist-scs-summary.json --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute instances delete "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
EOF
