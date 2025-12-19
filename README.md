# grouping-trainer

Training code for Sentry's AI grouping model.

(Sampling and labeling is still in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data). Will migrate that over here.)


## Usage

In a Workbench GPU instance (at least L4):

```
!git clone https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git
```

```
%cd grouping-trainer
```

Open [`./train.ipynb`](./train.ipynb).


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
