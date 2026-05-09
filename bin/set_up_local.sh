#!/usr/bin/env sh
set -eu

direnv allow
uv sync --extra dev --extra sheets
uv run pre-commit install
