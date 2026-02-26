"""Combine test and val similarity CSVs into a single file.

Drops timing columns from val, validates no duplicate pairs exist,
and writes the combined result to similarities/test-val-combined/similarities.csv.
"""

import polars as pl
import sys

TEST_PATH = "similarities/2026-01-08-13-19-08-test/similarities.csv"
VAL_PATH = "similarities/2026-02-19-01-55-09-val/similarities.csv"
OUTPUT_PATH = "similarities/test-val-combined/similarities.csv"

TIMING_COLS = [
    "query_encode_time_gte-finetuned",
    "candidate_encode_time_gte-finetuned",
    "query_encode_time_prod",
    "candidate_encode_time_prod",
]

PAIR_KEY = ["query_seer_event_sent", "candidate_seer_event_sent"]


def main():
    print(f"Reading test: {TEST_PATH}")
    test_df = pl.read_csv(TEST_PATH)
    print(f"  rows: {len(test_df)}, cols: {test_df.columns}")

    print(f"Reading val: {VAL_PATH}")
    val_df = pl.read_csv(VAL_PATH)
    print(f"  rows: {len(val_df)}, cols: {val_df.columns}")

    # Drop timing columns from val
    val_df = val_df.drop([c for c in TIMING_COLS if c in val_df.columns])
    print(f"  after dropping timing cols: {val_df.columns}")

    # Validate columns match
    if set(test_df.columns) != set(val_df.columns):
        missing_in_val = set(test_df.columns) - set(val_df.columns)
        missing_in_test = set(val_df.columns) - set(test_df.columns)
        print("ERROR: Column mismatch!")
        if missing_in_val:
            print(f"  In test but not val: {missing_in_val}")
        if missing_in_test:
            print(f"  In val but not test: {missing_in_test}")
        sys.exit(1)

    # Reorder val columns to match test
    val_df = val_df.select(test_df.columns)

    # Check for duplicates within each dataset
    for name, df in [("test", test_df), ("val", val_df)]:
        n_dupes = len(df) - df.n_unique(subset=PAIR_KEY)
        if n_dupes > 0:
            print(f"WARNING: {name} has {n_dupes} duplicate pairs within itself")

    # Check for cross-dataset duplicates
    cross_dupes = test_df.join(val_df, on=PAIR_KEY, how="inner")
    if len(cross_dupes) > 0:
        print(f"ERROR: {len(cross_dupes)} duplicate pairs found across test and val!")
        print(cross_dupes.select(PAIR_KEY).head(5))
        sys.exit(1)
    print("No cross-dataset duplicates found.")

    # Combine
    combined = pl.concat([test_df, val_df])
    print(f"Combined: {len(combined)} rows ({len(test_df)} + {len(val_df)})")

    # Final uniqueness check
    n_dupes = len(combined) - combined.n_unique(subset=PAIR_KEY)
    if n_dupes > 0:
        print(f"WARNING: Combined result has {n_dupes} duplicate pairs")

    combined.write_csv(OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
