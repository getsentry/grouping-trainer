# grouping-trainer

Training code for Sentry's AI grouping model.

Sampling and labeling is in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data).


## Set up locally

```bash
./bin/set_up_local.sh
```


## Usage

### Train

```bash
python train.py --gpu h100 --run_shortname my-run
```

Train and eval metrics are logged to https://wandb.ai/sentry-seer/grouping-trainer.

For DDP:

```bash
python train.py --gpu h100-ddp-4 --run_shortname my-run
```


### Debug

<details>
<summary>Launch a bare instance to SSH into</summary>

```bash
python -m grouping_trainer.launch --gpu h100
```

</details>


<details>
<summary>SSH into an instance from local</summary>

Add this function to your `~/.zshrc`:

```bash
gssh() {
    gcloud compute ssh "$1" --zone="${2:-us-central1-a}" --project=ml-ai-420606 --tunnel-through-iap
}
```

Then:

```bash
gssh grouping-trainer-h100
# or override the zone:
gssh grouping-trainer-a100 europe-west4-a
```

</details>


<details>
<summary>Check instance output</summary>

SSH into the instance and run:

```bash
logs
# shortcut for:
# sudo tail -f /var/log/grouping_trainer_run.log
```

If that file doesn't exist, the startup script never reached the `eval $COMMAND` block. Check what it actually did:

```bash
sudo journalctl -u google-startup-scripts.service --no-pager
```

From local (use when you can't SSH in, e.g., the boot itself failed):

```bash
gcloud compute instances get-serial-port-output grouping-trainer-l4-eval --zone=us-central1-a --project=ml-ai-420606 | tail -100
```

</details>


### Eval

<details>
<summary>Launch a GPU to run a model on held-out projects</summary>

```bash
python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix \
    --truncate_dims 64 128 256 512 768 \
    --use_compiled
```

</details>

<details>
<summary>Compare two models head-to-head on held-out projects</summary>

```bash
python eval/compare.py \
    --name_model1 v1 \
    --name_model2 large-no-prefix \
    --gcs_model1 gs://grouping-data/runs/issue_grouping_v1/similarities/test_full3 \
    --gcs_model2 gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
    --threshold_model1 0.99 \
    --threshold_model2 0.90 \
    --dim_model2 64 \
    --overwrite
```

</details>
