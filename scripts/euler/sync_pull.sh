#!/usr/bin/env bash
# Pull simulation outputs (runs/) back from Euler. Small JSON/npz only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "${LOCAL_ROOT}/runs"
rsync -az \
  --include '*/' --include '*.json' --include '*.npz' --include '*.txt' --include '*.csv' \
  --exclude '*' \
  -e "ssh -o BatchMode=yes" \
  "${EULER_USER}@${EULER_HOST}:${EULER_REMOTE}/runs/" \
  "${LOCAL_ROOT}/runs/"
echo "pulled runs/ <- ${EULER_USER}@${EULER_HOST}:${EULER_REMOTE}/runs/"
