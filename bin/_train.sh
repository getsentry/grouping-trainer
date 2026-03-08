#!/bin/bash
# Startup script for the H100 training VM.
# Sets up the environment, then trains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_startup.sh"

accelerate launch train.py
