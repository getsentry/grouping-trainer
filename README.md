# grouping-trainer

Labeling, training, and evaluation code for Sentry's AI grouping model.


## Install

In Google Colab, e.g., store a secret for your `GITHUB_TOKEN` and:

```python
from google.colab import userdata
import os

os.environ["GITHUB_TOKEN"] = userdata["GITHUB_TOKEN"]
!pip install git+https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git
```


## Usage

Coming soon.


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
