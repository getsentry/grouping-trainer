from collections.abc import Iterable

import pytest
import torch

import grouping_trainer as gt


def _reconcat(sub_batches: Iterable[gt.data.Batch]) -> gt.data.Batch:
    queries: list[str] = []
    candidates: list[str] = []
    labels_per_batch: list[torch.Tensor] = []
    sample_weights_per_batch: list[torch.Tensor] = []
    for sb in sub_batches:
        queries.extend(sb["query_stacktrace_string"])
        candidates.extend(sb["candidate_stacktrace_string"])
        labels_per_batch.append(sb["label"])
        sample_weights_per_batch.append(sb["sample_weight"])
    labels = torch.cat(labels_per_batch, dim=0) if labels_per_batch else torch.empty((0,), dtype=torch.int64)
    sample_weights = torch.cat(sample_weights_per_batch, dim=0) if sample_weights_per_batch else torch.empty((0,))
    return {
        "query_stacktrace_string": queries,
        "candidate_stacktrace_string": candidates,
        "label": labels,
        "sample_weight": sample_weights,
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
    queries: list[str] = ["q" * (token_count * 4) for token_count in token_counts]
    candidates: list[str] = ["c" * (token_count * 4 + 4) for token_count in token_counts]  # offset to differ
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.int64)

    sample_weights = torch.ones(len(queries))
    batch: gt.data.Batch = {
        "query_stacktrace_string": queries,
        "candidate_stacktrace_string": candidates,
        "label": labels,
        "sample_weight": sample_weights,
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


def test_batch_pairs_by_token_budget_rejects_empty_batch():
    empty: gt.data.Batch = {
        "query_stacktrace_string": [],
        "candidate_stacktrace_string": [],
        "label": torch.empty((0,), dtype=torch.int64),
        "sample_weight": torch.empty((0,)),
    }
    with pytest.raises(ValueError, match="Batch is empty"):
        list(gt.train.batch_pairs_by_token_budget(empty, token_budget=128))


def test_batch_pairs_by_token_budget_rejects_inconsistent_lengths():
    bad: gt.data.Batch = {
        "query_stacktrace_string": ["q" * 4, "q" * 8],
        "candidate_stacktrace_string": ["c" * 4],
        "label": torch.tensor([1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(2),
    }
    with pytest.raises(ValueError, match="inconsistent lengths"):
        list(gt.train.batch_pairs_by_token_budget(bad, token_budget=128))


@pytest.mark.parametrize("token_budget", [0, -1])
def test_batch_pairs_by_token_budget_rejects_non_positive_budget(token_budget: int) -> None:
    batch: gt.data.Batch = {
        "query_stacktrace_string": ["q"],
        "candidate_stacktrace_string": ["c"],
        "label": torch.tensor([1], dtype=torch.int64),
        "sample_weight": torch.ones(1),
    }
    with pytest.raises(ValueError, match="token_budget must be positive"):
        list(gt.train.batch_pairs_by_token_budget(batch, token_budget=token_budget))


def test_batch_pairs_by_token_budget_oversized_pair_yielded_as_singleton() -> None:
    """An individual pair that already exceeds the budget is still yielded (best-effort)."""
    batch: gt.data.Batch = {
        "query_stacktrace_string": ["q" * 4000],  # ~1000 tokens via len // 4 heuristic
        "candidate_stacktrace_string": ["c"],
        "label": torch.tensor([1], dtype=torch.int64),
        "sample_weight": torch.ones(1),
    }
    sub_batches = list(gt.train.batch_pairs_by_token_budget(batch, token_budget=10))
    assert len(sub_batches) == 1
    assert len(sub_batches[0]["label"]) == 1


def test_batch_pairs_by_token_budget_honors_custom_count_tokens() -> None:
    """If count_tokens always returns 1000 and budget is 512, every pair becomes its own sub-batch."""
    batch: gt.data.Batch = {
        "query_stacktrace_string": ["q"] * 4,
        "candidate_stacktrace_string": ["c"] * 4,
        "label": torch.tensor([1, 0, 1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(4),
    }
    sub_batches = list(gt.train.batch_pairs_by_token_budget(batch, token_budget=512, count_tokens=lambda _: 1000))
    assert [len(sb["label"]) for sb in sub_batches] == [1, 1, 1, 1]


def test_batch_pairs_by_token_budget_splits_at_expected_indices() -> None:
    """
    With count_tokens=10 for every text and budget=100, the cost formula is
    `2 * num_pairs * max_tokens = 20 * num_pairs`. Adding the 6th pair would push cost to 120
    and trigger a flush. So 8 pairs total should split as [5, 3].
    """
    batch: gt.data.Batch = {
        "query_stacktrace_string": ["q"] * 8,
        "candidate_stacktrace_string": ["c"] * 8,
        "label": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(8),
    }
    sub_batches = list(gt.train.batch_pairs_by_token_budget(batch, token_budget=100, count_tokens=lambda _: 10))
    assert [len(sb["label"]) for sb in sub_batches] == [5, 3]


# ModelForTraining.encode — the cache-hit-via-deduplication forward path


@pytest.fixture(scope="module")
def model_for_training(encoder: gt.utils.SentenceTransformer) -> gt.train.ModelForTraining:
    """ModelForTraining wrapping the session encoder. Module-scoped: built once for all encode tests."""
    return gt.train.ModelForTraining(encoder=encoder, loss=gt.loss.ContrastiveLoss()).eval()


def _make_batch(queries: list[str], candidates: list[str]) -> gt.data.Batch:
    n = len(queries)
    assert len(candidates) == n
    return {
        "query_stacktrace_string": queries,
        "candidate_stacktrace_string": candidates,
        "label": torch.zeros(n, dtype=torch.int64),
        "sample_weight": torch.ones(n),
    }


@torch.inference_mode()
def test_encode_duplicate_queries_get_identical_embeddings(
    model_for_training: gt.train.ModelForTraining,
) -> None:
    """Cache-hit invariant: positions with the same query text must have identical query embeddings."""
    batch = _make_batch(
        queries=["foo bar baz", "foo bar baz", "completely different content here"],
        candidates=["one candidate", "another candidate", "third candidate"],
    )
    features = model_for_training.encode(batch)
    assert torch.equal(features["query_embeddings"][0], features["query_embeddings"][1])
    assert not torch.equal(features["query_embeddings"][0], features["query_embeddings"][2])


@torch.inference_mode()
def test_encode_query_candidate_text_overlap_yields_equal_embeddings(
    model_for_training: gt.train.ModelForTraining,
) -> None:
    """If a query and a candidate happen to be the same string, dedup must give them the same embedding."""
    shared = "an entirely shared stacktrace string"
    batch = _make_batch(
        queries=[shared, "another query text"],
        candidates=["a different candidate text", shared],
    )
    features = model_for_training.encode(batch)
    assert torch.equal(features["query_embeddings"][0], features["candidate_embeddings"][1])


@torch.inference_mode()
def test_encode_preserves_input_order_after_internal_unique_sort(
    model_for_training: gt.train.ModelForTraining,
) -> None:
    """
    `np.unique` sorts the deduplicated texts internally, so the inverse-index reconstruction must put
    embeddings back into input order. Reverse-alphabetical input would expose any order bug.
    """
    queries = ["zebra query text", "yankee query text", "alpha query text"]
    candidates = ["zulu candidate text", "yulia candidate text", "anna candidate text"]
    batch = _make_batch(queries=queries, candidates=candidates)
    features = model_for_training.encode(batch)

    # Reference: encode each list directly through the same low-level path, no dedup.
    encoder = model_for_training.encoder
    expected_queries = encoder(
        {k: v.to(encoder.device) for k, v in encoder.tokenize(queries, return_tensors="pt", padding=True).items()}
    )["sentence_embedding"]
    expected_candidates = encoder(
        {k: v.to(encoder.device) for k, v in encoder.tokenize(candidates, return_tensors="pt", padding=True).items()}
    )["sentence_embedding"]

    assert torch.allclose(features["query_embeddings"], expected_queries, atol=1e-5)
    assert torch.allclose(features["candidate_embeddings"], expected_candidates, atol=1e-5)


@torch.inference_mode()
def test_encode_query_candidate_split_at_correct_boundary(
    model_for_training: gt.train.ModelForTraining,
) -> None:
    """The `num_queries` split must produce shape (len(queries), D) and (len(candidates), D), not a swap/off-by-one."""
    queries = ["query alpha text", "query beta text", "query gamma text"]
    candidates = ["candidate delta text", "candidate epsilon text", "candidate zeta text"]
    batch = _make_batch(queries=queries, candidates=candidates)
    features = model_for_training.encode(batch)

    assert features["query_embeddings"].shape[0] == len(queries)
    assert features["candidate_embeddings"].shape[0] == len(candidates)
    # No query text equals any candidate text, so the two embedding sets must not overlap
    for query_embedding in features["query_embeddings"]:
        for candidate_embedding in features["candidate_embeddings"]:
            assert not torch.equal(query_embedding, candidate_embedding)


# make_dummy_batch


@torch.inference_mode()
def test_make_dummy_batch_runs_through_model_for_training_forward(
    model_for_training: gt.train.ModelForTraining,
) -> None:
    """
    `training_step` uses make_dummy_batch() to pad ranks with fewer sub-batches so DDP collectives stay in sync.
    The shape contract must match ModelForTraining.forward exactly; this guards against silent breakage if either
    Batch or the forward signature changes. Mirrors the post-_prepare_inputs device placement.
    """
    device = model_for_training.encoder.device
    dummy_batch = gt.data.make_dummy_batch()
    dummy_batch["label"] = dummy_batch["label"].to(device)
    dummy_batch["sample_weight"] = dummy_batch["sample_weight"].to(device)
    loss = model_for_training(
        dummy_batch,
        dummy_batch["label"],
        sample_weight=dummy_batch["sample_weight"],
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar loss
