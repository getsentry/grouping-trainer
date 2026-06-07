import polars as pl
import pytest

import grouping_trainer as gt


def _make_df_platforms(counts_by_platform: dict[str, int]) -> pl.DataFrame:
    """
    Build a minimal pairs DataFrame with the given number of rows per platform.
    """
    platforms_repeated = [platform for platform, count in counts_by_platform.items() for _ in range(count)]
    return pl.DataFrame({"platform": platforms_repeated}).with_row_index("row_id")


COUNTS_BY_PLATFORM = {"javascript": 50, "python": 20, "ruby": 5, "java": 25}
PLATFORMS_HOLDOUT = ("python", "ruby")
N_HOLDOUT = COUNTS_BY_PLATFORM["python"] + COUNTS_BY_PLATFORM["ruby"]


def test_drop_platforms_removes_all_held_out_rows():
    """
    Treatment arm: every held-out-platform row is gone, and the row count drops by exactly their count.
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    df_treatment = gt.data._apply_platform_holdout(
        df, platforms_holdout=PLATFORMS_HOLDOUT, holdout_mode="drop_platforms", holdout_seed=0
    )
    assert df_treatment.height == df.height - N_HOLDOUT
    assert df_treatment.filter(pl.col("platform").is_in(PLATFORMS_HOLDOUT)).height == 0


def test_drop_random_match_is_volume_matched_and_keeps_platforms():
    """
    Control arm: same row count as treatment, but the held-out platforms are still present (dropped at random).
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    df_control = gt.data._apply_platform_holdout(
        df, platforms_holdout=PLATFORMS_HOLDOUT, holdout_mode="drop_random_match", holdout_seed=0
    )
    assert df_control.height == df.height - N_HOLDOUT
    assert df_control.filter(pl.col("platform").is_in(PLATFORMS_HOLDOUT)).height > 0


def test_treatment_and_control_have_equal_row_counts():
    """
    The whole point of the control: both arms train on the same number of pairs.
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    df_treatment = gt.data._apply_platform_holdout(
        df, platforms_holdout=PLATFORMS_HOLDOUT, holdout_mode="drop_platforms", holdout_seed=0
    )
    df_control = gt.data._apply_platform_holdout(
        df, platforms_holdout=PLATFORMS_HOLDOUT, holdout_mode="drop_random_match", holdout_seed=0
    )
    assert df_treatment.height == df_control.height


def test_drop_random_match_is_deterministic_per_seed():
    """
    A given seed reproduces the exact same control draw; different seeds differ.
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    df_seed_0a = gt.data._apply_platform_holdout(df, PLATFORMS_HOLDOUT, "drop_random_match", holdout_seed=0)
    df_seed_0b = gt.data._apply_platform_holdout(df, PLATFORMS_HOLDOUT, "drop_random_match", holdout_seed=0)
    df_seed_1 = gt.data._apply_platform_holdout(df, PLATFORMS_HOLDOUT, "drop_random_match", holdout_seed=1)
    assert df_seed_0a["row_id"].to_list() == df_seed_0b["row_id"].to_list()
    assert df_seed_0a["row_id"].to_list() != df_seed_1["row_id"].to_list()


def test_empty_holdout_is_noop():
    """
    No `platforms_holdout` means ordinary training: the frame is returned unchanged.
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    df_noop = gt.data._apply_platform_holdout(df, platforms_holdout=(), holdout_mode="drop_platforms", holdout_seed=0)
    assert df_noop.height == df.height


def test_missing_platform_raises():
    """
    A typo'd platform must fail loudly rather than silently dropping nothing and aliasing the control to the baseline.
    """
    df = _make_df_platforms(COUNTS_BY_PLATFORM)
    with pytest.raises(AssertionError, match="not found in data"):
        gt.data._apply_platform_holdout(df, platforms_holdout=("js",), holdout_mode="drop_platforms", holdout_seed=0)


# _record_from_dict


@pytest.mark.parametrize("label_str, expected_int", [("GROUP", 1), ("SEPARATE", 0)])
def test_record_from_dict_converts_known_labels(label_str: str, expected_int: int) -> None:
    record = gt.data._record_from_dict(
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
        gt.data._record_from_dict(
            {
                "query_stacktrace_string": "q",
                "candidate_stacktrace_string": "c",
                "label": "GROUP ",  # trailing space — the kind of typo we want to catch
            }
        )


def test_record_from_dict_defaults_when_optional_keys_missing() -> None:
    record = gt.data._record_from_dict(
        {"query_stacktrace_string": "q", "candidate_stacktrace_string": "c", "label": "GROUP"}
    )
    assert record["sample_weight"] == 1.0


def test_record_from_dict_coerces_string_sample_weight() -> None:
    record = gt.data._record_from_dict(
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
    dataset = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=False)
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
    dataset = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
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
    dataset = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
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
    dataset = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=False)
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
    dataset_a = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=7)
    dataset_b = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=7)
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
    dataset_a = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=None)
    dataset_b = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=None)
    assert dataset_a["query_stacktrace_string"] == dataset_b["query_stacktrace_string"]


def test_df_to_dataset_different_seeds_produce_different_orders() -> None:
    df = pl.DataFrame(
        {
            "query_stacktrace_string": [f"q{i:02d}" for i in range(10)],  # 10! permutations
            "candidate_stacktrace_string": [f"c{i}" for i in range(10)],
            "label": ["GROUP"] * 10,
        }
    )
    dataset_a = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=1)
    dataset_b = gt.data.df_to_dataset(df, group_by_query_stacktrace_string=True, shuffle_groups=True, seed=2)
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
    dataset_dict = gt.data.create_project_dataset_dict(df, min_dataset_size=None)
    assert "__packed__" not in dataset_dict
    assert set(dataset_dict.keys()) == {"1", "2"}
    assert dataset_dict["1"].num_rows == 2
    assert dataset_dict["2"].num_rows == 1


def test_create_project_dataset_dict_small_projects_get_packed() -> None:
    df = _project_df({1: 10, 2: 2, 3: 3})  # only project 1 meets min_dataset_size=5
    dataset_dict = gt.data.create_project_dataset_dict(df, min_dataset_size=5)
    assert set(dataset_dict.keys()) == {"1", "__packed__"}
    assert dataset_dict["1"].num_rows == 10
    assert dataset_dict["__packed__"].num_rows == 5  # 2 + 3


def test_create_project_dataset_dict_all_small_only_packed_key() -> None:
    df = _project_df({1: 1, 2: 1, 3: 1})
    dataset_dict = gt.data.create_project_dataset_dict(df, min_dataset_size=5)
    assert set(dataset_dict.keys()) == {"__packed__"}
    assert dataset_dict["__packed__"].num_rows == 3


def test_create_project_dataset_dict_all_large_no_packed_key() -> None:
    df = _project_df({1: 3, 2: 3})
    dataset_dict = gt.data.create_project_dataset_dict(df, min_dataset_size=2)
    assert "__packed__" not in dataset_dict
    assert set(dataset_dict.keys()) == {"1", "2"}


def test_create_project_dataset_dict_keys_are_strings() -> None:
    """DatasetDict's __getitem__ accepts both int (positional) and str (named) keys; we use strings."""
    df = _project_df({42: 1, 99: 1})
    dataset_dict = gt.data.create_project_dataset_dict(df, min_dataset_size=None)
    for key in dataset_dict.keys():
        assert isinstance(key, str)
    assert "42" in dataset_dict
    assert "99" in dataset_dict
