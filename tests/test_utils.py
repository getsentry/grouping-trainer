import polars as pl
import pytest

import grouping_trainer as gt

# deduplicate_pairs


def test_deduplicate_pairs_keeps_first_of_identical_pair():
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["a", "a"],
            "candidate_stacktrace_string": ["b", "b"],
            "label": ["GROUP", "SEPARATE"],
        }
    )
    result = gt.utils.deduplicate_pairs(df)
    assert result.height == 1
    assert result["label"].to_list() == ["GROUP"]


def test_deduplicate_pairs_keeps_first_of_swapped_pair():
    """Symmetric dedup: (q, c) and (c, q) are the same pair."""
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["a", "b"],
            "candidate_stacktrace_string": ["b", "a"],
            "label": ["GROUP", "SEPARATE"],
        }
    )
    result = gt.utils.deduplicate_pairs(df)
    assert result.height == 1
    assert result["query_stacktrace_string"].to_list() == ["a"]
    assert result["candidate_stacktrace_string"].to_list() == ["b"]
    assert result["label"].to_list() == ["GROUP"]


def test_deduplicate_pairs_distinct_pairs_all_kept_in_order():
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["a", "c", "e"],
            "candidate_stacktrace_string": ["b", "d", "f"],
        }
    )
    result = gt.utils.deduplicate_pairs(df)
    assert result.height == 3
    assert result["query_stacktrace_string"].to_list() == ["a", "c", "e"]
    assert result["candidate_stacktrace_string"].to_list() == ["b", "d", "f"]


def test_deduplicate_pairs_preserves_other_columns_on_kept_row():
    df = pl.DataFrame(
        {
            "query_stacktrace_string": ["a", "b"],
            "candidate_stacktrace_string": ["b", "a"],
            "label": ["GROUP", "SEPARATE"],
            "project_id": [42, 99],
        }
    )
    result = gt.utils.deduplicate_pairs(df)
    assert result.height == 1
    assert result["label"].to_list() == ["GROUP"]
    assert result["project_id"].to_list() == [42]


def test_deduplicate_pairs_preserves_input_column_order():
    df = pl.DataFrame(
        {
            "label": ["GROUP"],
            "query_stacktrace_string": ["a"],
            "candidate_stacktrace_string": ["b"],
            "project_id": [1],
        }
    )
    result = gt.utils.deduplicate_pairs(df)
    assert result.columns == df.columns


def test_deduplicate_pairs_with_custom_column_names():
    df = pl.DataFrame(
        {
            "x": ["a", "b"],
            "y": ["b", "a"],
            "label": ["GROUP", "SEPARATE"],
        }
    )
    result = gt.utils.deduplicate_pairs(df, column1="x", column2="y")
    assert result.height == 1
    assert result["label"].to_list() == ["GROUP"]


# concat_vertical_unordered


def test_concat_vertical_unordered_reorders_to_first_df_columns():
    df_first = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    df_second = pl.DataFrame({"c": [6], "a": [4], "b": [5]})
    result = gt.utils.concat_vertical_unordered([df_first, df_second])
    assert result.columns == ["a", "b", "c"]
    assert result["a"].to_list() == [1, 4]
    assert result["b"].to_list() == [2, 5]
    assert result["c"].to_list() == [3, 6]


def test_concat_vertical_unordered_raises_on_missing_column():
    df_first = pl.DataFrame({"a": [1], "b": [2]})
    df_second = pl.DataFrame({"a": [3]})
    with pytest.raises(ValueError, match="Columns are not the same"):
        gt.utils.concat_vertical_unordered([df_first, df_second])


def test_concat_vertical_unordered_raises_on_extra_column():
    df_first = pl.DataFrame({"a": [1]})
    df_second = pl.DataFrame({"a": [2], "b": [3]})
    with pytest.raises(ValueError, match="Columns are not the same"):
        gt.utils.concat_vertical_unordered([df_first, df_second])


def test_concat_vertical_unordered_single_df_passes_through():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = gt.utils.concat_vertical_unordered(iter([df]))
    assert result.equals(df)
