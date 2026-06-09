# benchmark_compiled report

This benchmarks the compiled vs. base model on stacktrace data. For the model-agnostic, cross-model speedup benchmark, see [the sentence-transformers compilation example](https://github.com/kddubey/sentence-transformers/tree/kddubey/examples/compilation/examples/sentence_transformer/applications/compilation).

## Run

```
run_gcs_dir=gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix
df_path=final_csvs/test_full2.csv
stamp=2026-04-24-23-44-59
sample_size=66753
text_prefix=''
model_kwargs={'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}
```

- Token bucket boundaries used for analysis: `(64, 128, 256, 512, 1024)`
- Rows: 66,753

## Headline

- Median compiled: **14.7 ms**
- Median base:     **36.8 ms**
- Per-row speedup p10/p50/p90: **1.01x / 2.51x / 3.45x**
- Compiled wins on **92.6%** of rows

## Per-bucket

| bucket   | n     | tok_p50 | compiled_ms_p50 | base_ms_p50 | compiled_ms_p90 | base_ms_p90 | speedup_p50 |
|----------|-------|---------|-----------------|-------------|-----------------|-------------|-------------|
| <=64     | 8301  | 35.0    | 10.58           | 35.17       | 11.72           | 35.99       | 3.32        |
| 65-128   | 7628  | 93.0    | 10.68           | 35.4        | 11.65           | 36.21       | 3.31        |
| 129-256  | 13930 | 194.0   | 11.62           | 36.05       | 12.85           | 37.02       | 3.1         |
| 257-512  | 14804 | 369.0   | 15.54           | 36.97       | 17.3            | 38.1        | 2.38        |
| 513-1024 | 13858 | 682.0   | 26.9            | 38.86       | 28.97           | 40.85       | 1.44        |
| >1024    | 8232  | 1494.5  | 48.06           | 47.04       | 115.66          | 115.89      | 0.98        |

## Worst 5 rows for compiled

| num_tokens | compiled_ms | base_ms | speedup |
|------------|-------------|---------|---------|
| 1252       | 54.8        | 43.26   | 0.789   |
| 1043       | 50.24       | 42.14   | 0.839   |
| 2020       | 82.6        | 70.98   | 0.859   |
| 1122       | 49.67       | 42.91   | 0.864   |
| 2040       | 79.23       | 69.04   | 0.871   |
