from collections.abc import Iterable

import polars as pl
import pytest
import torch

import grouping_trainer as gt


def _reconcat(sub_batches: Iterable[gt.train.Batch]) -> gt.train.Batch:
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
    batch: gt.train.Batch = {
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
    empty: gt.train.Batch = {
        "query_stacktrace_string": [],
        "candidate_stacktrace_string": [],
        "label": torch.empty((0,), dtype=torch.int64),
        "sample_weight": torch.empty((0,)),
    }
    with pytest.raises(ValueError, match="Batch is empty"):
        list(gt.train.batch_pairs_by_token_budget(empty, token_budget=128))


def test_batch_pairs_by_token_budget_rejects_inconsistent_lengths():
    bad: gt.train.Batch = {
        "query_stacktrace_string": ["q" * 4, "q" * 8],
        "candidate_stacktrace_string": ["c" * 4],
        "label": torch.tensor([1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(2),
    }
    with pytest.raises(ValueError, match="inconsistent lengths"):
        list(gt.train.batch_pairs_by_token_budget(bad, token_budget=128))


@pytest.mark.parametrize("token_budget", [0, -1])
def test_batch_pairs_by_token_budget_rejects_non_positive_budget(token_budget: int) -> None:
    batch: gt.train.Batch = {
        "query_stacktrace_string": ["q"],
        "candidate_stacktrace_string": ["c"],
        "label": torch.tensor([1], dtype=torch.int64),
        "sample_weight": torch.ones(1),
    }
    with pytest.raises(ValueError, match="token_budget must be positive"):
        list(gt.train.batch_pairs_by_token_budget(batch, token_budget=token_budget))


def test_batch_pairs_by_token_budget_oversized_pair_yielded_as_singleton() -> None:
    """An individual pair that already exceeds the budget is still yielded (best-effort)."""
    batch: gt.train.Batch = {
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
    batch: gt.train.Batch = {
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
    batch: gt.train.Batch = {
        "query_stacktrace_string": ["q"] * 8,
        "candidate_stacktrace_string": ["c"] * 8,
        "label": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int64),
        "sample_weight": torch.ones(8),
    }
    sub_batches = list(gt.train.batch_pairs_by_token_budget(batch, token_budget=100, count_tokens=lambda _: 10))
    assert [len(sb["label"]) for sb in sub_batches] == [5, 3]


# _record_from_dict


@pytest.mark.parametrize("label_str, expected_int", [("GROUP", 1), ("SEPARATE", 0)])
def test_record_from_dict_converts_known_labels(label_str: str, expected_int: int) -> None:
    record = gt.train._record_from_dict(
        {
            "query_stacktrace_string": "q",
            "candidate_stacktrace_string": "c",
            "label": label_str,
        }
    )
    assert record["label"] == expected_int


def test_record_from_dict_rejects_unknown_label() -> None:
    """A typo or unexpected label must surface, not silently become SEPARATE."""
    with pytest.raises(ValueError, match="Unknown label"):
        gt.train._record_from_dict(
            {
                "query_stacktrace_string": "q",
                "candidate_stacktrace_string": "c",
                "label": "GROUP ",  # trailing space — the kind of typo we want to catch
            }
        )


def test_record_from_dict_defaults_when_optional_keys_missing() -> None:
    record = gt.train._record_from_dict(
        {"query_stacktrace_string": "q", "candidate_stacktrace_string": "c", "label": "GROUP"}
    )
    assert record["sample_weight"] == 1.0


def test_record_from_dict_coerces_string_sample_weight() -> None:
    record = gt.train._record_from_dict(
        {
            "query_stacktrace_string": "q",
            "candidate_stacktrace_string": "c",
            "label": "GROUP",
            "sample_weight": "2.5",
        }
    )
    assert record["sample_weight"] == 2.5
    assert isinstance(record["sample_weight"], float)


# df_to_dataset


def _query_runs(queries: list[str]) -> list[str]:
    """Compress consecutive duplicates: ['a','a','b','a'] -> ['a','b','a']."""
    runs: list[str] = []
    for q in queries:
        if not runs or runs[-1] != q:
            runs.append(q)
    return runs


def test_df_to_dataset_no_grouping_preserves_row_order() -> None:
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["q3", "q1", "q2"],
            "candidate_stacktrace_string": ["c3", "c1", "c2"],
            "label": ["GROUP", "SEPARATE", "GROUP"],
        }
    )
    dataset = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=False)
    assert dataset["query_stacktrace_string"] == ["q3", "q1", "q2"]
    assert dataset["candidate_stacktrace_string"] == ["c3", "c1", "c2"]
    assert dataset["label"] == [1, 0, 1]


def test_df_to_dataset_grouping_keeps_same_query_contiguous() -> None:
    """Cache-hit invariant: ModelForTraining.encode dedupes within a batch, so same-query rows must be adjacent."""
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["qA", "qB", "qC", "qA", "qB", "qC"],  # interleaved
            "candidate_stacktrace_string": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "label": ["GROUP"] * 6,
        }
    )
    dataset = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
    runs = _query_runs(dataset["query_stacktrace_string"])
    assert len(runs) == len(set(runs)), f"each query should appear in one contiguous run; got: {runs}"


def test_df_to_dataset_sorts_candidates_by_length_within_query_group() -> None:
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["q", "q", "q"],
            "candidate_stacktrace_string": ["xxxxxxx", "x", "xxx"],  # lengths 7, 1, 3
            "label": ["GROUP"] * 3,
        }
    )
    dataset = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
    assert dataset["candidate_stacktrace_string"] == ["x", "xxx", "xxxxxxx"]


def test_df_to_dataset_no_shuffle_groups_in_alphabetical_order() -> None:
    """DDP determinism without shuffle: polars group_by is non-deterministic, so we sort groups by query string."""
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["qC", "qA", "qB"],
            "candidate_stacktrace_string": ["c1", "c2", "c3"],
            "label": ["GROUP"] * 3,
        }
    )
    dataset = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
    assert dataset["query_stacktrace_string"] == ["qA", "qB", "qC"]


def test_df_to_dataset_shuffle_with_seed_is_deterministic_across_calls() -> None:
    """DDP cross-process determinism: same seed must produce same group order on every call."""
    df = pl.DataFrame(
        {
            "query_stacktrace_string": [f"q{i:02d}" for i in range(10)],
            "candidate_stacktrace_string": [f"c{i}" for i in range(10)],
            "label": ["GROUP"] * 10,
        }
    )
    dataset_a = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=7)
    dataset_b = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=7)
    assert dataset_a["query_stacktrace_string"] == dataset_b["query_stacktrace_string"]


def test_df_to_dataset_default_seed_is_deterministic() -> None:
    """seed=None falls back to the hard-coded seed=42 (per source), so two calls still match."""
    df = pl.DataFrame(
        {
            "query_stacktrace_string": [f"q{i:02d}" for i in range(10)],
            "candidate_stacktrace_string": [f"c{i}" for i in range(10)],
            "label": ["GROUP"] * 10,
        }
    )
    dataset_a = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=None)
    dataset_b = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=None)
    assert dataset_a["query_stacktrace_string"] == dataset_b["query_stacktrace_string"]


def test_df_to_dataset_different_seeds_produce_different_orders() -> None:
    df = pl.DataFrame(
        {
            "query_stacktrace_string": [f"q{i:02d}" for i in range(10)],  # 10! permutations
            "candidate_stacktrace_string": [f"c{i}" for i in range(10)],
            "label": ["GROUP"] * 10,
        }
    )
    dataset_a = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=1)
    dataset_b = gt.train.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=2)
    assert dataset_a["query_stacktrace_string"] != dataset_b["query_stacktrace_string"]


# create_project_dataset_dict


def _project_df(project_id_to_size: dict[int, int]) -> pl.DataFrame:
    rows = []
    for project_id, size in project_id_to_size.items():
        for i in range(size):
            rows.append(
                {
                    "query_stacktrace_string": f"q_p{project_id}_{i}",
                    "candidate_stacktrace_string": f"c_p{project_id}_{i}",
                    "label": "GROUP",
                    "project_id": project_id,
                }
            )
    return pl.DataFrame(rows)


def test_create_project_dataset_dict_no_min_size_keeps_all_projects_separate() -> None:
    df = _project_df({1: 2, 2: 1})
    dataset_dict = gt.train.create_project_dataset_dict(df, min_dataset_size=None)
    assert "__packed__" not in dataset_dict
    assert set(dataset_dict.keys()) == {"1", "2"}
    assert dataset_dict["1"].num_rows == 2
    assert dataset_dict["2"].num_rows == 1


def test_create_project_dataset_dict_small_projects_get_packed() -> None:
    df = _project_df({1: 10, 2: 2, 3: 3})  # only project 1 meets min_dataset_size=5
    dataset_dict = gt.train.create_project_dataset_dict(df, min_dataset_size=5)
    assert set(dataset_dict.keys()) == {"1", "__packed__"}
    assert dataset_dict["1"].num_rows == 10
    assert dataset_dict["__packed__"].num_rows == 5  # 2 + 3


def test_create_project_dataset_dict_all_small_only_packed_key() -> None:
    df = _project_df({1: 1, 2: 1, 3: 1})
    dataset_dict = gt.train.create_project_dataset_dict(df, min_dataset_size=5)
    assert set(dataset_dict.keys()) == {"__packed__"}
    assert dataset_dict["__packed__"].num_rows == 3


def test_create_project_dataset_dict_all_large_no_packed_key() -> None:
    df = _project_df({1: 3, 2: 3})
    dataset_dict = gt.train.create_project_dataset_dict(df, min_dataset_size=2)
    assert "__packed__" not in dataset_dict
    assert set(dataset_dict.keys()) == {"1", "2"}


def test_create_project_dataset_dict_keys_are_strings() -> None:
    """DatasetDict's __getitem__ accepts both int (positional) and str (named) keys; we use strings."""
    df = _project_df({42: 1, 99: 1})
    dataset_dict = gt.train.create_project_dataset_dict(df, min_dataset_size=None)
    for key in dataset_dict.keys():
        assert isinstance(key, str)
    assert "42" in dataset_dict
    assert "99" in dataset_dict


# ModelForTraining.encode — the cache-hit-via-deduplication forward path


@pytest.fixture(scope="module")
def model_for_training(encoder: gt.utils.SentenceTransformer) -> gt.train.ModelForTraining:
    """ModelForTraining wrapping the session encoder. Module-scoped: built once for all encode tests."""
    return gt.train.ModelForTraining(encoder=encoder, loss=gt.loss.ContrastiveLoss()).eval()


def _make_batch(queries: list[str], candidates: list[str]) -> gt.train.Batch:
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
    dummy_batch = gt.train.make_dummy_batch()
    dummy_batch["label"] = dummy_batch["label"].to(device)
    dummy_batch["sample_weight"] = dummy_batch["sample_weight"].to(device)
    loss = model_for_training(
        dummy_batch,
        dummy_batch["label"],
        sample_weight=dummy_batch["sample_weight"],
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar loss
