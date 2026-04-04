#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ID="${PROJECT_ID:-fluid-house-463510-i7}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-mist-demo-$(date +%Y%m%d-%H%M%S)}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50GB}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2404-lts-amd64}"
REPO_URL="${REPO_URL:-https://github.com/schwallergroup/mist.git}"
BRANCH="${BRANCH:-main}"
NETWORK="${NETWORK:-default}"
SUBNET="${SUBNET:-}"
FIGSHARE_URL="${FIGSHARE_URL:-}"
EXTRACT_DEMO_FIXTURES="${EXTRACT_DEMO_FIXTURES:-0}"
DEMO_SAMPLE_ROWS="${DEMO_SAMPLE_ROWS:-50}"

NETWORK_ARGS=(--network "${NETWORK}")
if [[ -n "${SUBNET}" ]]; then
  NETWORK_ARGS+=(--subnet "${SUBNET}")
fi

METADATA_ITEMS=("REPO_URL=${REPO_URL}" "BRANCH=${BRANCH}")
if [[ -n "${FIGSHARE_URL}" ]]; then
  METADATA_ITEMS+=("FIGSHARE_URL=${FIGSHARE_URL}")
fi
if [[ -n "${EXTRACT_DEMO_FIXTURES}" ]]; then
  METADATA_ITEMS+=("EXTRACT_DEMO_FIXTURES=${EXTRACT_DEMO_FIXTURES}")
fi
if [[ -n "${DEMO_SAMPLE_ROWS}" ]]; then
  METADATA_ITEMS+=("DEMO_SAMPLE_ROWS=${DEMO_SAMPLE_ROWS}")
fi
METADATA_STRING="$(IFS=,; echo "${METADATA_ITEMS[*]}")"

cat <<EOF
Creating VM with:
  project: ${PROJECT_ID}
  zone: ${ZONE}
  instance: ${INSTANCE_NAME}
  machine: ${MACHINE_TYPE}
  image: ${IMAGE_PROJECT}/${IMAGE_FAMILY}
  network: ${NETWORK}
EOF

gcloud compute instances create "${INSTANCE_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --boot-disk-size "${BOOT_DISK_SIZE}" \
  --image-project "${IMAGE_PROJECT}" \
  --image-family "${IMAGE_FAMILY}" \
  --scopes cloud-platform \
  "${NETWORK_ARGS[@]}" \
  --metadata "${METADATA_STRING}" \
  --metadata-from-file startup-script="${SCRIPT_DIR}/gcp_demo_startup.sh"

cat <<EOF

Instance created. Useful follow-up commands:

  gcloud compute instances tail-serial-port-output "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute ssh "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute scp "${INSTANCE_NAME}:/var/log/mist-demo-output.json" ./mist-demo-output.json --project "${PROJECT_ID}" --zone "${ZONE}"
  gcloud compute instances delete "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}"
EOF
