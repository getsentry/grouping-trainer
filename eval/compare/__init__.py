"""Head-to-head model comparison. See eval/compare/__main__.py for the CLI entry point."""

from .metrics import CompareResult, compare_models

__all__ = ["CompareResult", "compare_models"]
