#!/usr/bin/env bash
# Shared Euler config for the CV-7 remaining-phases runs.
# Source this:  source scripts/euler/env.sh
export EULER_HOST="${EULER_HOST:-euler.civil.columbia.edu}"
export EULER_USER="${EULER_USER:-ws2414}"
export EULER_REMOTE="${EULER_REMOTE:-neural_atlas_MPMcontact}"   # path relative to remote $HOME
# The 'atlas' conda env already carries torch 2.5.1 + scipy 1.15.3 + numpy + gudhi + matplotlib + pyvista.
export EULER_PY="${EULER_PY:-/home/ws2414/miniconda3/envs/atlas/bin/python}"
export EULER_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "${EULER_USER}@${EULER_HOST}")

# Local repo root (two levels up from this file)
EULER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_ROOT="$(cd "${EULER_ENV_DIR}/../.." && pwd)"
