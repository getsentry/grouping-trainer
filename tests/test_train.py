# TODO: test real dataloader:
# - set seed and check good mix across batches
# - each batch has one project
# - each batch is sorted by query
#
# TODO: test deduplication forward
#
from collections.abc import Iterable

import pytest
import torch

import grouping_trainer as gt


def _reconcat(sub_batches: Iterable[gt.train.Batch]) -> gt.train.Batch:
    queries = []
    candidates = []
    labels = []
    sample_weights = []
    confidence_scores = []
    for sb in sub_batches:
        queries.extend(sb["query_stacktrace_string"])
        candidates.extend(sb["candidate_stacktrace_string"])
        labels.append(sb["label"])
        sample_weights.append(sb["sample_weight"])
        confidence_scores.append(sb["confidence_score"])
    labels = torch.cat(labels, dim=0) if labels else torch.empty((0,), dtype=torch.int64)
    sample_weights = torch.cat(sample_weights, dim=0) if sample_weights else torch.empty((0,))
    confidence_scores = torch.cat(confidence_scores, dim=0) if confidence_scores else torch.empty((0,))
    return {
        "query_stacktrace_string": queries,
        "candidate_stacktrace_string": candidates,
        "label": labels,
        "sample_weight": sample_weights,
        "confidence_score": confidence_scores,
    }


@pytest.mark.parametrize(
    "token_budget",
    [
        1,  # extreme: forces 1-pair microbatches (budget is best-effort)
        8,
        32,
        128,
        1024,
        10_000,
        1_000_000,  # should produce a single batch
    ],
)
def test_batch_pairs_by_token_budget_roundtrip_preserves_pairs_order_and_alignment(
    token_budget,
):
    # Construct strings with widely varying lengths.
    # Default heuristic is len(text)//4, so make lengths multiples of 4.
    token_counts = [1, 2, 50, 3, 2000, 4, 10, 100, 7, 512, 9]
    queries = ["q" * (t * 4) for t in token_counts]
    candidates = ["c" * (t * 4 + 4) for t in token_counts]  # offset to differ
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.int64)

    sample_weights = torch.ones(len(queries))
    confidence_scores = torch.ones(len(queries))
    batch = {
        "query_stacktrace_string": queries,
        "candidate_stacktrace_string": candidates,
        "label": labels,
        "sample_weight": sample_weights,
        "confidence_score": confidence_scores,
    }

    sub_batches = list(gt.train.batch_pairs_by_token_budget(batch, token_budget=token_budget))
    assert sub_batches, "expected at least one yielded sub-batch"

    # No empty sub-batches
    assert all(len(sb["label"]) > 0 for sb in sub_batches)

    # Re-concat must exactly match the input
    roundtrip = _reconcat(sub_batches)
    assert roundtrip["query_stacktrace_string"] == queries
    assert roundtrip["candidate_stacktrace_string"] == candidates
    assert torch.equal(roundtrip["label"], labels)
    assert torch.equal(roundtrip["sample_weight"], sample_weights)
    assert torch.equal(roundtrip["confidence_score"], confidence_scores)


def test_batch_pairs_by_token_budget_rejects_empty_batch():
    empty = {
        "query_stacktrace_string": [],
        "candidate_stacktrace_string": [],
        "label": torch.empty((0,), dtype=torch.int64),
        "sample_weight": torch.empty((0,)),
        "confidence_score": torch.empty((0,)),
    }
    with pytest.raises(ValueError):
        list(gt.train.batch_pairs_by_token_budget(empty, token_budget=128))


def test_batch_pairs_by_token_budget_rejects_inconsistent_lengths():
    bad = {
        "query_stacktrace_string": ["q" * 4, "q" * 8],
        "candidate_stacktrace_string": ["c" * 4],
        "label": torch.tensor([1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(2),
        "confidence_score": torch.ones(2),
    }
    with pytest.raises(ValueError):
        list(gt.train.batch_pairs_by_token_budget(bad, token_budget=128))
