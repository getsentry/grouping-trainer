#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_startup.sh"

NCCL_NET=Socket LD_LIBRARY_PATH="" accelerate launch --multi_gpu train.py --run_shortname ddp
