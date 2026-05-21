"""
Combine prod- and finetuned-model save_embeddings outputs into DB-ready CSVs.

Syncs both models' similarities dirs from GCS, writes two combined exports locally under
`eval/similarities/{finetuned_run}/{dataset}/{export_for_db,export_for_load_test}/` and uploads each to the matching
subdir of `<gcs_finetuned>/`. The outputs in each subdir are:

  - grouping_records_db.csv: one row per unique (project_id, candidate_hash), with both models' candidate embeddings
    formatted as pgvector strings under the column names given by --db_column_prod / --db_column_finetuned.
  - query_stacktraces.csv: unique query stacktrace strings with all query-side metadata.
  - candidate_stacktraces.csv: unique candidate stacktrace strings with all candidate-side metadata.

`export_for_load_test/` is the same export but with candidates (and optionally queries) randomly downsampled for load
testing.

Usage:

python eval/export_for_db.py \
    --gcs_prod gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3 \
    --gcs_finetuned gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
    --dim_finetuned 64
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from tap import tapify

# Pair keys used to verify the two model dirs were produced from the same df_path.
_PAIR_KEY_COLS = ["project_id", "query_hash", "candidate_hash"]


def _embedding_to_pgvector(emb: np.ndarray) -> str:
    """Format a 1D numpy array as a pgvector string: [0.123,0.456,...]"""
    return "[" + ",".join(f"{x:.6g}" for x in emb) + "]"


def _load_model_dir(similarities_dir: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Load similarities.csv + candidate_embeddings.npy from one save_embeddings dir."""
    df = pl.read_csv(f"{similarities_dir}/similarities.csv")
    embs = np.load(f"{similarities_dir}/candidate_embeddings.npy")
    print(f"  {similarities_dir}: {len(df)} pairs, candidate embeddings {embs.shape}")
    assert len(df) == len(embs), f"Row count mismatch: {len(df)} CSV rows vs {len(embs)} embedding rows"
    return df, embs


def _sync_gcs_dir(gcs_dir: str) -> Path:
    """Sync a GCS similarities dir to eval/similarities/{run}/{dataset}/ and return the local dir.

    Expects ``gs://bucket/runs/{run}/similarities/{dataset}``.
    """
    gcs_dir = gcs_dir.rstrip("/")
    parts = gcs_dir.split("/")
    idx_similarities = parts.index("similarities")
    name_run = parts[idx_similarities - 1]
    name_dataset = parts[idx_similarities + 1]
    dir_local = Path("eval/similarities") / name_run / name_dataset
    dir_local.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {gcs_dir} → {dir_local}")
    subprocess.run(["gcloud", "storage", "rsync", "-r", gcs_dir, str(dir_local)], check=True)
    return dir_local


def _truncate(embs: np.ndarray, dim: int | None, label: str) -> np.ndarray:
    """Truncate the trailing axis to ``dim`` (MRL-style). No-op if dim is None."""
    if dim is None:
        return embs
    if dim > embs.shape[-1]:
        raise ValueError(f"{label}: requested dim {dim} exceeds embedding dim {embs.shape[-1]}")
    truncated = embs[..., :dim]
    print(f"  Truncated {label} embeddings: {embs.shape} -> {truncated.shape}")
    return truncated


def _assert_same_pairs(df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
    """Verify both similarities CSVs cover the same pairs in the same order.

    save_embeddings.py encodes texts in CSV row order, so npy[i] aligns with CSV row i.
    If the two model dirs were produced from the same df_path, the pair keys should match exactly.
    """
    keys_a = df_a.select(_PAIR_KEY_COLS)
    keys_b = df_b.select(_PAIR_KEY_COLS)
    if not keys_a.equals(keys_b):
        raise ValueError(
            f"Pair-key mismatch between the two similarities dirs ({len(keys_a)} vs {len(keys_b)} rows). "
            f"Both must be produced by save_embeddings.py on the same df_path so row order aligns."
        )


def _export(
    df: pl.DataFrame,
    candidate_embs_by_column: dict[str, np.ndarray],
    export_dir: str,
    full_df: pl.DataFrame | None = None,
    keep_fraction_queries: float = 1.0,
    seed: int = 42,
) -> None:
    """Export db records and stacktrace CSVs from a pairs DataFrame.

    df must already have a "_row_idx" column indexing into the candidate embedding arrays.
    candidate_embs_by_column maps each output pgvector column name to its candidate embedding array.
    If full_df is provided, query_stacktraces.csv is derived from it instead of df (useful when df
    has been thinned but you want all queries). If keep_fraction_queries < 1.0, randomly downsample
    queries.
    """
    os.makedirs(export_dir, exist_ok=True)

    # --- grouping_records_db.csv ---
    candidate_deduped = df.unique(subset=["project_id", "candidate_hash"], keep="first", maintain_order=True)
    row_indices = candidate_deduped["_row_idx"].to_numpy()
    print(f"  Unique (project_id, candidate_hash): {len(candidate_deduped)} (from {len(df)} pairs)")

    embedding_columns = {
        col: [_embedding_to_pgvector(emb) for emb in embs[row_indices]]
        for col, embs in candidate_embs_by_column.items()
    }
    db_records = pl.DataFrame(
        {
            "id": candidate_deduped["candidate_seer_gr_id"],
            "project_id": candidate_deduped["project_id"],
            "hash": candidate_deduped["candidate_hash"],
            "error_type": candidate_deduped["candidate_error_type"],
            **embedding_columns,
        }
    )
    db_records.write_csv(f"{export_dir}/grouping_records_db.csv")
    print(f"  Saved {export_dir}/grouping_records_db.csv ({len(db_records)} rows)")

    # --- query_stacktraces.csv ---
    query_source = full_df if full_df is not None else df
    other_cols = [
        "project_id",
        "platform",
        "org_id",
        "distance",
        "source",
        "path",
        "label",
        "thinking_output",
        "response_output",
        "confidence_score",
        "prompt",
    ]
    query_cols = [c for c in query_source.columns if c.startswith("query_")] + other_cols
    query_df = query_source.select(query_cols).unique(
        subset=["query_stacktrace_string"], keep="first", maintain_order=True
    )
    if keep_fraction_queries < 1.0:
        n_keep = int(len(query_df) * keep_fraction_queries)
        query_df = query_df.sample(n=n_keep, seed=seed)
        print(f"  Downsampled queries: keeping {n_keep} ({keep_fraction_queries:.0%})")
    query_df.write_csv(f"{export_dir}/query_stacktraces.csv")
    print(f"  Saved {export_dir}/query_stacktraces.csv ({len(query_df)} unique query strings)")

    # --- candidate_stacktraces.csv ---
    candidate_cols = [c for c in df.columns if c.startswith("candidate_")] + other_cols
    candidate_df = df.select(candidate_cols).unique(
        subset=["candidate_stacktrace_string"], keep="first", maintain_order=True
    )
    candidate_df.write_csv(f"{export_dir}/candidate_stacktraces.csv")
    print(f"  Saved {export_dir}/candidate_stacktraces.csv ({len(candidate_df)} unique candidate strings)")


def thin_candidates(
    df: pl.DataFrame,
    keep_fraction_candidates: float = 0.5,
    seed: int = 42,
) -> pl.DataFrame:
    """Randomly drop a fraction of all candidates to reduce DB density.

    Removes (1 - keep_fraction_candidates) of unique (project_id, candidate_hash) pairs
    and all rows involving them. Fewer candidates in the DB means fewer queries
    find a match during retrieval.
    """
    all_candidates = df.select(["project_id", "candidate_hash"]).unique()
    n_keep = int(len(all_candidates) * keep_fraction_candidates)
    kept = all_candidates.sample(n=n_keep, seed=seed)

    result = df.join(kept, on=["project_id", "candidate_hash"])

    print(f"  Total unique candidates: {len(all_candidates)}")
    print(f"  Keeping {n_keep} ({keep_fraction_candidates:.0%}), dropping {len(all_candidates) - n_keep}")
    print(f"  Pairs: {len(df)} -> {len(result)}")
    return result


def main(
    gcs_prod: str,
    gcs_finetuned: str,
    db_column_prod: str = "stacktrace_embedding",
    db_column_finetuned: str = "stacktrace_embedding_v2_1",
    dim_prod: int | None = None,
    dim_finetuned: int | None = None,
    keep_fraction_candidates: float = 0.5,
    keep_fraction_queries: float = 1.0,
) -> None:
    """Combine prod- and finetuned-model save_embeddings outputs into DB-ready CSVs (full + load-test).

    Both models' similarities dirs are synced from GCS; the per-row alignment is verified before
    embeddings are paired with metadata. Each of the two output variants is uploaded under the
    finetuned run's GCS dir (export_for_db and export_for_load_test).

    Parameters
    ----------
    gcs_prod
        GCS dir of save_embeddings output for the prod (baseline) model,
        e.g. gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3
    gcs_finetuned
        Same, for the finetuned model. Pairs must match gcs_prod row-for-row.
    db_column_prod
        pgvector column name for the prod embedding in grouping_records_db.csv.
    db_column_finetuned
        pgvector column name for the finetuned embedding in grouping_records_db.csv.
    dim_prod
        MRL truncation dim for the prod embedding. None (default) uses the full embedding dim.
    dim_finetuned
        MRL truncation dim for the finetuned embedding, e.g. 64 for the short variant.
        None (default) uses the full embedding dim.
    keep_fraction_candidates
        Fraction of unique (project_id, candidate_hash) pairs to keep in the load-test export.
    keep_fraction_queries
        Fraction of unique query stacktraces to keep in the load-test export.
    """
    dir_prod = _sync_gcs_dir(gcs_prod)
    dir_finetuned = _sync_gcs_dir(gcs_finetuned)

    df_prod, embs_prod = _load_model_dir(str(dir_prod))
    df_finetuned, embs_finetuned = _load_model_dir(str(dir_finetuned))
    _assert_same_pairs(df_prod, df_finetuned)

    embs_prod = _truncate(embs_prod, dim_prod, "prod")
    embs_finetuned = _truncate(embs_finetuned, dim_finetuned, "finetuned")
    candidate_embs_by_column = {db_column_prod: embs_prod, db_column_finetuned: embs_finetuned}

    df = df_finetuned.with_row_index("_row_idx")

    dir_full_local = dir_finetuned / "export_for_db"
    print(f"Writing full export to {dir_full_local}/")
    _export(df, candidate_embs_by_column, str(dir_full_local))

    dir_loadtest_local = dir_finetuned / "export_for_load_test"
    print(f"Writing load-test export to {dir_loadtest_local}/")
    thinned_df = thin_candidates(df, keep_fraction_candidates=keep_fraction_candidates)
    _export(
        thinned_df,
        candidate_embs_by_column,
        str(dir_loadtest_local),
        full_df=df,
        keep_fraction_queries=keep_fraction_queries,
    )

    gcs_finetuned = gcs_finetuned.rstrip("/")
    for dir_export_local, name in [(dir_full_local, "export_for_db"), (dir_loadtest_local, "export_for_load_test")]:
        gcs_export_dir = f"{gcs_finetuned}/{name}"
        print(f"Uploading {dir_export_local} → {gcs_export_dir}")
        subprocess.run(["gcloud", "storage", "rsync", "-r", str(dir_export_local), gcs_export_dir], check=True)
        print(f"Uploaded to {gcs_export_dir}")


if __name__ == "__main__":
    tapify(main)
