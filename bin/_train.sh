#!/bin/bash
# Startup script for the H100 training VM.
# Sets up the environment, then trains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_startup.sh"

python train.py
