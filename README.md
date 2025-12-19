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

```bash
cd grouping-trainer
```

Open [`./train.ipynb`](./train.ipynb) and run the cells.


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
