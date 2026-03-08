#!/usr/bin/env bash
set -euo pipefail

direnv allow
python3.13 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -e ".[dev]"
