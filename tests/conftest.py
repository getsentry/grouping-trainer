import pytest

import grouping_trainer as gt


@pytest.fixture(scope="session")
def model_name() -> str:
    return "Alibaba-NLP/gte-modernbert-base"  # 150M param


@pytest.fixture(scope="session")
def encoder(model_name: str) -> gt.utils.SentenceTransformer:
    return gt.utils.SentenceTransformer(model_name)
