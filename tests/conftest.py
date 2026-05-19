import pytest

import grouping_trainer as gt


@pytest.fixture(scope="session")
def encoder() -> gt.utils.SentenceTransformer:
    return gt.utils.SentenceTransformer("Alibaba-NLP/gte-modernbert-base")  # 150M param
