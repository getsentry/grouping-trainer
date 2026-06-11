"""
Delete PENDING multi-flex-start losers — instances whose same-named sibling is already RUNNING.

Useful for clearing a prior launch's stale PENDING flex-start requests so a freshly staged run doesn't queue behind
them. A `--multi_flex_start` launch already does this automatically before fanning out; this is the manual entrypoint.

Pass --dry_run to print what would be deleted without deleting.
"""

from tap import tapify

import grouping_trainer as gt

if __name__ == "__main__":
    gt.logging.configure_logging(process_type="launch")
    tapify(gt.launch.prune_decided_pending_instances)
