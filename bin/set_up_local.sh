#!/usr/bin/env sh
set -eu

direnv allow
python3.13 -m venv .venv
# shellcheck source=/dev/null
. .venv/bin/activate
python -m pip install -e ".[dev,sheets]"
