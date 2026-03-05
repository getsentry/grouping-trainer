# grouping-trainer

Training code for Sentry's AI grouping model.

(Sampling and labeling is still in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data). Will migrate that over here.)


## Usage

### Training + async eval

1. Create the VMs (`bin/_startup.sh` auto-installs deps, clones repo, downloads data):

    ```bash
    ./bin/flex_h100.sh   # training
    ./bin/flex_l4.sh     # async eval
    ```

2. SSH into the H100, wait for setup (`cat /tmp/setup-done`), then:

    ```bash
    cd grouping-trainer
    screen -S train
    conda activate gt-env
    python train.py  # prints the eval_poller.py command for L4
    ```

3. SSH into the L4, wait for setup, then paste the printed command:

    ```bash
    cd grouping-trainer
    screen -S eval
    conda activate gt-env
    python eval/eval_poller.py --gcs-dir <printed> --wandb-run-id <printed>
    ```

4. After training, upload wandb artifacts:

    ```bash
    gsutil -m cp -r wandb gs://grouping-data/runs/{OUTPUT_DIR}
    gsutil -m rsync -r {OUTPUT_DIR} gs://grouping-data/runs/{OUTPUT_DIR}/training
    ```

### screen cheatsheet

```bash
screen -S name        # create session
# Ctrl+A, then D      # detach
screen -ls            # list sessions
screen -r name        # reattach
```

TODO: enable DDP training. Finetuning [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) w/
sdpa takes ~5 hours w/ 1 A100 80 GB. Our quota in
[us-central1](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D_2C%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aus-central1_5C_22_22_2C_22i_22_3A_22displayDimensions_22%257D%255D%22,%22p%22:0,%22r%22:200)))
and
[europe-west4](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aeurope-west4_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayDimensions_22%257D_2C%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D%255D%22,%22p%22:0,%22r%22:200)))
is 4.


## Set up locally

```
direnv allow
```

```bash
python3.13 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -e ".[dev]"
```
