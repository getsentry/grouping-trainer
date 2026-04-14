"""
GCP Cloud Logging compatible JSON formatter.
Adapted from https://github.com/getsentry/seer/blob/main/src/seer/logging.py by Josh Ferge.

Usage:
    from grouping_trainer.logging import configure_logging
    configure_logging(run_name="2026-03-08-my-run", process_type="training")
"""

import json
import logging
import os
from datetime import UTC, datetime


class GCPJsonFormatter(logging.Formatter):
    """
    Formats log records as JSON for GCP Cloud Logging. GCP expects 'severity' instead of 'level', and parses JSON logs
    automatically when they're written to stdout/stderr.
    """

    RESERVED_ATTRS = frozenset(
        {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }
    )

    PROTECTED_OUTPUT_KEYS = frozenset(
        {
            "timestamp",
            "severity",
            "logger",
            "message",
            "exc_info",
            "stack_info",
            "logging.googleapis.com/sourceLocation",
        }
    )

    def __init__(self, extra_fields: dict[str, str] | None = None):
        super().__init__()
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self.extra_fields,
        }

        if record.pathname and record.lineno:
            log_obj["logging.googleapis.com/sourceLocation"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_obj["stack_info"] = record.stack_info

        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and key not in self.PROTECTED_OUTPUT_KEYS:
                try:
                    json.dumps(value)
                    log_obj[key] = value
                except (TypeError, ValueError):
                    log_obj[key] = str(value)

        return json.dumps(log_obj)


_DEV_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    run_name: str | None = None,
    process_type: str | None = None,
):
    """
    Configure root logging. Uses JSON formatting by default (for GCP), plain text if DEV=1.

    Parameters
    ----------
    run_name
        Training run name (e.g. "2026-03-08-20-54-08-gte"). Attached to every log line.
    process_type
        "training" or "eval_poller". Attached to every log line.
    """
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    if os.environ.get("DEV"):
        handler.setFormatter(logging.Formatter(_DEV_FORMAT))
    else:
        extra_fields: dict[str, str] = {}
        if run_name is not None:
            extra_fields["run_name"] = run_name
        if process_type is not None:
            extra_fields["process_type"] = process_type
        handler.setFormatter(GCPJsonFormatter(extra_fields=extra_fields))

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
