# grouping-trainer

Training code for Sentry's AI grouping model.

(Sampling and labeling is still in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data). Will migrate that over here.)


## Install

In Google Colab, e.g., store a secret for your `GITHUB_TOKEN` and:

```python
from google.colab import userdata
import os

os.environ["GITHUB_TOKEN"] = userdata["GITHUB_TOKEN"]
!pip install git+https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git
```


## Usage

See [`./train.ipynb`](./train.ipynb)


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
