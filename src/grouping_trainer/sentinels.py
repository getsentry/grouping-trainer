"""
GCS filenames used to coordinate between training and eval polling.
"""

CHECKPOINT_DONE = ".checkpoint_done"
"""
Written when the trainer finishes uploading a checkpoint dir to GCS.
Writes to the checkpoint's GCS path.
Causes the eval poller to consider this checkpoint ready for evaluation.
"""

EVAL_DONE = ".eval_done"
"""
Written when the eval poller finishes evaluating a checkpoint.
Writes to the checkpoint's GCS path.
Causes the eval poller to skip this checkpoint on future polling cycles.
"""

BASELINE_EVAL_DONE = ".baseline_eval_done"
"""
Written when the eval poller finishes evaluating the base model (step 0).
Writes to the run's GCS root.
Causes the eval poller to skip the baseline evaluation on restart.
"""

TRAINING_DONE = ".training_done"
"""
Written when training ends (on_train_end).
Writes to the run's GCS root.
Causes the eval poller to run a final backfill pass and then exit.
"""
