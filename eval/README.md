# Eval scripts

Typical steps:

1. If a training run did well enough on validation to warrant an evaluation on the test set, run
   [`./save_embeddings.py`](./save_embeddings.py) to launch a GPU to save embeddings and similarities for that model in
   GCS.

   ```bash
   python eval/save_embeddings.py \
       --run_gcs_dir gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix \
       --truncate_dims 64 128 256 512 768 \
       --use_compiled  # should work but you can run w/o it first to make sure
   ```

   Consider tuning the token buckets for a compiled model using [`../benchmark`](../benchmark/).

2. Run `eval.compare` to compare the model to another model on the test set.

   ```bash
   python -m eval.compare \
       --gcs_model1 gs://$GROUPING_TRAINER_BUCKET/runs/v1/similarities/test_full3 \
       --threshold_model1 0.99 \
       --gcs_model2 gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
       --threshold_model2 0.90 \
       --dim_model2 64 \
       --overwrite
   ```

   Consider adding the `--upload_sheets` flag to upload the most impacted projects to Google Sheets and qualitatively
   assess how merges and non-merges have changed. The OAuth client JSON is fetched automatically from GCP Secret Manager
   using the secret name in `OAUTH_CLIENT_SECRET_NAME` (set in your `.env`) — you just need read access to that secret.
   One-time setup, if the secret doesn't exist yet:

   ```bash
   gcloud secrets create $OAUTH_CLIENT_SECRET_NAME --data-file=client_secret.json
   ```

3. To estimate the throughput a model can handle, run [`./export_for_db.py`](./export_for_db.py) to export the
   embeddings for loading into a DB and running a load test—

   ```bash
   python eval/export_for_db.py \
       --gcs_prod gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3 \
       --gcs_finetuned gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
       --dim_finetuned 64
   ```

   —and then run the Seer load test in https://github.com/getsentry/seer/tree/main/benchmark.


## Production

To sample matches from production, automatically label them using Claude, and produce a report, use
https://github.com/getsentry/data-analysis/tree/main/grouping/data#label-matches-from-prod.
