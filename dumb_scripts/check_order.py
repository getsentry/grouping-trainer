import polars as pl

file1 = "similarities/2026-01-08-13-19-08-test/similarities.csv"
file2 = "similarities/2026-02-20-17-50-21-test-lr-2e-5/similarities.csv"

join_keys = ["query_seer_event_sent", "candidate_seer_event_sent"]

df1 = pl.read_csv(file1)
df2 = pl.read_csv(file2)

# Only grab the unique columns from file2 (plus the join keys)
cols_only_in_f2 = [c for c in df2.columns if c not in df1.columns]
df2_slim = df2.select(join_keys + cols_only_in_f2)

joined = df1.join(df2_slim, on=join_keys, how="inner")

print(f"File 1 rows:  {df1.height}")
print(f"File 2 rows:  {df2.height}")
print(f"Joined rows:  {joined.height}")

assert joined.height == df1.height == df2.height, "Row count mismatch after join!"

null_counts = joined.null_count()
any_nulls = {c: null_counts[c][0] for c in null_counts.columns if null_counts[c][0] > 0}
if any_nulls:
    print(f"WARNING: nulls found: {any_nulls}")
else:
    print("No nulls — all rows matched cleanly.")

print(f"Final columns ({len(joined.columns)}): {joined.columns}")

out = "similarities/similarities_all.csv"
joined.write_csv(out)
print(f"Wrote {out}")
