"""
NOTE: this script was vibe-coded.

Delete `checkpoint-*` dirs from training runs that already have an `inference/`
dir.

A run gets an `inference/` dir once the final model has been extracted, so its
intermediate `checkpoint-*` dirs are no longer needed. Runs still missing an
`inference/` dir are left untouched.

Defaults to a dry run that only prints what would be deleted. Pass --no_dry_run
to actually delete.
"""

import os
import subprocess

from tap import tapify

RUNS_PREFIX = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/runs/"


def list_dir(prefix):
    """
    Return the immediate child paths of a gs:// prefix.
    """
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def find_checkpoints_to_delete():
    """
    Yield checkpoint dirs belonging to runs that already have an inference/ dir.
    """
    for run in list_dir(RUNS_PREFIX):
        if not run.endswith("/"):
            continue
        children = list_dir(run)
        has_inference = any(child.rstrip("/").endswith("/inference") for child in children)
        if not has_inference:
            continue
        for child in children:
            if child.endswith("/") and child.rstrip("/").rsplit("/", 1)[-1].startswith("checkpoint-"):
                yield child


def main(no_dry_run: bool = False):
    """
    Delete checkpoint dirs from runs that already have an inference/ dir.

    By default, only print what would be deleted. Pass --no_dry_run to actually
    delete.
    """
    count = 0
    for checkpoint in find_checkpoints_to_delete():
        count += 1
        if no_dry_run:
            print(f"Deleting {checkpoint} ...")
            subprocess.run(
                ["gcloud", "storage", "rm", "--recursive", checkpoint],
                check=True,
            )
        else:
            print(checkpoint)

    if not count:
        print("No checkpoints to delete.")
    elif no_dry_run:
        print(f"\nDeleted {count} checkpoint dir(s).")
    else:
        print(f"\n{count} checkpoint dir(s) across runs with an inference/ dir.")
        print("Dry run — nothing deleted. Pass --no_dry_run to delete.")


if __name__ == "__main__":
    tapify(main)
