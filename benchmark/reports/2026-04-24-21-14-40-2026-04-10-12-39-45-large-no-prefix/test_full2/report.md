# benchmark_compiled report

## Run

```
run_gcs_dir=gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix
df_path=final_csvs/test_full2.csv
stamp=2026-04-24-21-14-40
sample_size=66753
text_prefix=''
model_kwargs={'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}
```

- Token bucket boundaries used for analysis: `(64, 128, 256, 512, 1024)`
- Rows: 66,753

## Headline

- Median compiled: **14.9 ms**
- Median base:     **38.1 ms**
- Per-row speedup p10/p50/p90: **1.03x / 2.57x / 3.51x**
- Compiled wins on **96.0%** of rows

## Per-bucket

| bucket   | n     | tok_p50 | compiled_ms_p50 | base_ms_p50 | compiled_ms_p90 | base_ms_p90 | speedup_p50 |
|----------|-------|---------|-----------------|-------------|-----------------|-------------|-------------|
| <=64     | 8301  | 35.0    | 10.77           | 36.24       | 11.89           | 37.64       | 3.37        |
| 65-128   | 7628  | 93.0    | 10.94           | 36.48       | 11.87           | 37.86       | 3.33        |
| 129-256  | 13930 | 194.0   | 11.82           | 37.2        | 13.07           | 38.74       | 3.15        |
| 257-512  | 14804 | 369.0   | 15.72           | 38.25       | 17.46           | 39.8        | 2.43        |
| 513-1024 | 13858 | 682.0   | 27.03           | 40.19       | 29.03           | 42.22       | 1.49        |
| >1024    | 8232  | 1494.5  | 48.09           | 47.95       | 115.46          | 116.76      | 1.0         |

## Worst 5 rows for compiled

| num_tokens | compiled_ms | base_ms | speedup |
|------------|-------------|---------|---------|
| 1223       | 3088.66     | 42.73   | 0.014   |
| 1217       | 57.45       | 43.47   | 0.757   |
| 1366       | 56.84       | 44.62   | 0.785   |
| 1725       | 63.73       | 53.91   | 0.846   |
| 1693       | 61.93       | 52.79   | 0.852   |
