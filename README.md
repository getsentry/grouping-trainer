# grouping-trainer

Training code for Sentry's AI grouping model.

(Sampling and labeling is still in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data). Will migrate that over here.)


## Usage

In a Workbench GPU instance (at least an L4, A100 recommended), open the terminal and:

```bash
export GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token-grouping-trainer-temp --project=996102297610)
```

```bash
git clone https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git
```

Then open [`./train.ipynb`](./train.ipynb) and run the cells.

To enable DDP training, simply spin up a cluster of ≥2 GPUs in Workbench and run the cells as usual. Finetuning
[gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) w/ sdpa takes ~5 hours w/ 1 A100 80 GB,
~3.5 hours on 2. Our quota in us-central1 is 4.


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
