#!/usr/bin/env bash
# Push the working tree to Euler (code only; runs/downloads/figures stay local).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
rsync -az --delete \
  --exclude '.git/' --exclude 'runs/' --exclude 'downloads/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude '*.pt' --exclude '*.pth' --exclude 'figures/' --exclude '*.gif' --exclude '*.mp4' \
  -e "ssh -o BatchMode=yes" \
  "${LOCAL_ROOT}/" \
  "${EULER_USER}@${EULER_HOST}:${EULER_REMOTE}/"
echo "pushed ${LOCAL_ROOT} -> ${EULER_USER}@${EULER_HOST}:${EULER_REMOTE}/"
