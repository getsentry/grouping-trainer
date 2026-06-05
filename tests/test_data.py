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
