#!/usr/bin/env sh
set -eu

direnv allow
uv sync --all-extras
uv run pre-commit install
