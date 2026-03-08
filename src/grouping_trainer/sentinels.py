"""GCS sentinel filenames used to coordinate between training and eval polling."""

CHECKPOINT_DONE = ".checkpoint_done"
"""Written inside a checkpoint dir after it's fully uploaded to GCS."""
EVAL_DONE = ".eval_done"
"""Written inside a checkpoint dir after the eval poller has evaluated it."""
TRAINING_DONE = ".training_done"
"""Written at the run's GCS root when training is complete."""
