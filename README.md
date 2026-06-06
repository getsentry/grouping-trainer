# grouping-trainer

Training and eval code for [Sentry's AI grouping
model](https://docs.sentry.io/concepts/data-management/event-grouping/#ai-enhanced-grouping).


## Set up local

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), [direnv](https://direnv.net/), [Google Cloud
   SDK](https://docs.cloud.google.com/sdk/docs/install-sdk).

1. Sign up for [WandB](https://wandb.ai/site) and (optionally) add a secret at `wandb-api-key` in your GCP project.

1. Create `.env`

   Sentry employees:

   ```bash
   gcloud secrets versions access latest --secret=grouping-trainer-env > .env
   ```

   Others:

   ```bash
   cp .env.example .env
   ```

   And fill in the values.

1. Set up the local Python environment:

   ```bash
   bin/set_up_local.sh
   ```


## Usage

Assumes data w/ the columns in [`src/grouping_trainer/data.py`](./src/grouping_trainer/data.py) are written to GCS in
`gs://$GROUPING_TRAINER_BUCKET/final_csvs/`.


### Train

Sanity check that plumbing works locally:

```
python train.py --tiny_run
```

To run real experiments, check out a branch `experiment/...`, make changes, push to remote, and run:

```bash
python train.py --gpu h100-4 --run_shortname my-run --multi_flex_start
```

This command async-launches a DDP training run on 4 H100s which will run the commit at your local `HEAD`.


### Debug

<details>
<summary>Launch a bare instance to SSH into</summary>

```bash
python -m grouping_trainer.launch --gpu h100 --sync_start --git_ref main
```

</details>


<details>
<summary>SSH into an instance from local</summary>

Find your instance:

```bash
bin/gtlist
```

And SSH in:

```bash
bin/gtssh instance-name instance-zone
```

Run `sudo -i` to switch to `root`, which ran the startup script.

</details>


<details>
<summary>Check instance output</summary>

SSH into the instance and run:

```bash
logs
# shortcut for:
# sudo tail -n 50 -f /var/log/grouping_trainer_run.log
```

If that file doesn't exist, the startup script never reached the `eval $COMMAND` block. Check what it did reach:

```bash
startup-logs
# shortcut for:
# sudo journalctl -u google-startup-scripts.service --no-pager
```

From local (use when you can't SSH in b/c, e.g., the boot itself failed):

```bash
gcloud compute instances get-serial-port-output your-instance --zone=your-instance-zone | tail -100
```

</details>


### Eval

See [`./eval/`](./eval/).
