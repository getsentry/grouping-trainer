"""Combine val.csv and test.csv into a single val_and_test.csv.

Validates both inputs, checks for no cross-dataset duplicate pairs,
and writes final_csvs/val_and_test.csv.
"""

import polars as pl

import utils

VAL_PATH = "final_csvs/val.csv"
TEST_PATH = "final_csvs/test.csv"
OUTPUT_PATH = "final_csvs/val_and_test.csv"

PAIR_KEY = ["query_seer_event_sent", "candidate_seer_event_sent"]


def main():
    print(f"Loading val: {VAL_PATH}")
    val_df = utils.load_val_df(path=VAL_PATH)
    print(f"  {len(val_df)} rows")

    print(f"Loading test: {TEST_PATH}")
    test_df = utils.load_val_df(path=TEST_PATH)
    print(f"  {len(test_df)} rows")

    # Validate columns match exactly
    if set(val_df.columns) != set(test_df.columns):
        missing_in_val = set(test_df.columns) - set(val_df.columns)
        missing_in_test = set(val_df.columns) - set(test_df.columns)
        raise ValueError(
            f"Column mismatch! In test but not val: {missing_in_val or 'none'}. "
            f"In val but not test: {missing_in_test or 'none'}."
        )

    # Check for cross-dataset duplicates
    cross_dupes = val_df.join(test_df, on=PAIR_KEY, how="inner")
    if len(cross_dupes) > 0:
        raise ValueError(f"{len(cross_dupes)} duplicate pairs found across val and test!")
    print("No cross-dataset duplicates.")

    # Reorder val columns to match test
    val_df = val_df.select(test_df.columns)
    combined = pl.concat([test_df, val_df])
    print(f"Combined: {len(combined)} rows ({len(test_df)} + {len(val_df)})")

    # Final uniqueness check
    n_dupes = len(combined) - combined.n_unique(subset=PAIR_KEY)
    assert n_dupes == 0, f"Combined result has {n_dupes} duplicate pairs"

    combined.write_csv(OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
