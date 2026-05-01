"""Export save_embeddings output into DB-ready CSVs.

Takes an output directory from save_embeddings.py and saves, inside <similarities_dir>/export_for_db/:
  - grouping_records_db.csv: one row per unique (project_id, candidate_hash), with embeddings
    formatted as pgvector strings. Schema matches DbGroupingRecord.
  - query_stacktraces.csv: unique query stacktrace strings with all query-side metadata.
  - candidate_stacktraces.csv: unique candidate stacktrace strings with all candidate-side metadata.

Usage:
    python eval/export_for_db.py <similarities_dir>
    python eval/export_for_db.py eval/similarities/2026-02-26-12-00-00-val-and-test
"""

import json
import os
import sys

import numpy as np
import polars as pl

# model name (from model_configs.json) -> DB column name
MODEL_NAME_TO_DB_COLUMN = {
    "prod": "stacktrace_embedding",
    "gte-finetuned": "stacktrace_embedding_v2_short",
}

# DB embedding columns not populated by any model
EMPTY_DB_COLUMNS = ["stacktrace_embedding_v2"]


def _embedding_to_pgvector(emb: np.ndarray) -> str:
    """Format a 1D numpy array as a pgvector string: [0.123,0.456,...]"""
    return "[" + ",".join(f"{x:.6g}" for x in emb) + "]"


def _export(
    df: pl.DataFrame,
    candidate_embs_by_model: dict[str, np.ndarray],
    export_dir: str,
    full_df: pl.DataFrame | None = None,
    keep_fraction_queries: float = 1.0,
    seed: int = 42,
) -> None:
    """Export db records and stacktrace CSVs from a pairs DataFrame.

    df must already have a "_row_idx" column indexing into the .npy files.
    If full_df is provided, query_stacktraces.csv is derived from it instead
    of df (useful when df has been thinned but you want all queries).
    If keep_fraction_queries < 1.0, randomly downsample queries.
    """
    os.makedirs(export_dir, exist_ok=True)

    # --- grouping_records_db.csv ---
    candidate_deduped = df.unique(subset=["project_id", "candidate_hash"], keep="first", maintain_order=True)
    row_indices = candidate_deduped["_row_idx"].to_numpy()
    print(f"  Unique (project_id, candidate_hash): {len(candidate_deduped)} (from {len(df)} pairs)")

    embedding_columns = {}
    for model_name, db_column in MODEL_NAME_TO_DB_COLUMN.items():
        embs = candidate_embs_by_model[model_name][row_indices]
        embedding_columns[db_column] = [_embedding_to_pgvector(emb) for emb in embs]
    for db_column in EMPTY_DB_COLUMNS:
        embedding_columns[db_column] = None

    db_records = pl.DataFrame(
        {
            "id": candidate_deduped["candidate_seer_gr_id"],
            "project_id": candidate_deduped["project_id"],
            "hash": candidate_deduped["candidate_hash"],
            "error_type": candidate_deduped["candidate_error_type"],
            **embedding_columns,
        }
    )
    db_records = db_records.select(
        [
            "id",
            "project_id",
            "hash",
            "error_type",
            "stacktrace_embedding",
            "stacktrace_embedding_v2",
            "stacktrace_embedding_v2_short",
        ]
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


def main(similarities_dir: str) -> None:
    export_dir = f"{similarities_dir}/export_for_db"
    print(f"Loading from {similarities_dir}/")
    print(f"Writing to {export_dir}/")

    df = pl.read_csv(f"{similarities_dir}/similarities.csv")

    with open(f"{similarities_dir}/model_configs.json") as f:
        model_configs = json.load(f)
    model_names = [mc["name"] for mc in model_configs["model_configs"]]

    unknown_models = set(model_names) - set(MODEL_NAME_TO_DB_COLUMN)
    if unknown_models:
        raise ValueError(f"Unknown model names {unknown_models}, add them to MODEL_NAME_TO_DB_COLUMN")

    candidate_embs_by_model = {}
    for name in model_names:
        path = f"{similarities_dir}/{name}_candidate_embeddings.npy"
        candidate_embs_by_model[name] = np.load(path)
        print(f"  {name} candidate embeddings: {candidate_embs_by_model[name].shape}")

    print(f"  similarities.csv: {len(df)} rows")

    # Add row index to track back into .npy files
    df = df.with_row_index("_row_idx")

    _export(df, candidate_embs_by_model, export_dir)


def export_for_load_test(
    similarities_dir: str,
    keep_fraction_candidates: float = 0.5,
    keep_fraction_queries: float = 1.0,
) -> None:
    """Like main(), but randomly drops candidates and optionally queries.

    Writes to <similarities_dir>/export_for_load_test/.
    """
    export_dir = f"{similarities_dir}/export_for_load_test"
    print(f"Loading from {similarities_dir}/")
    print(f"Writing to {export_dir}/")

    df = pl.read_csv(f"{similarities_dir}/similarities.csv")

    with open(f"{similarities_dir}/model_configs.json") as f:
        model_configs = json.load(f)
    model_names = [mc["name"] for mc in model_configs["model_configs"]]

    unknown_models = set(model_names) - set(MODEL_NAME_TO_DB_COLUMN)
    if unknown_models:
        raise ValueError(f"Unknown model names {unknown_models}, add them to MODEL_NAME_TO_DB_COLUMN")

    candidate_embs_by_model = {}
    for name in model_names:
        path = f"{similarities_dir}/{name}_candidate_embeddings.npy"
        candidate_embs_by_model[name] = np.load(path)
        print(f"  {name} candidate embeddings: {candidate_embs_by_model[name].shape}")

    print(f"  similarities.csv: {len(df)} rows")

    df = df.with_row_index("_row_idx")
    thinned_df = thin_candidates(df, keep_fraction_candidates=keep_fraction_candidates)

    _export(thinned_df, candidate_embs_by_model, export_dir, full_df=df, keep_fraction_queries=keep_fraction_queries)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) >= 3 and sys.argv[2] == "--load-test":
        kwargs: dict[str, float] = {}
        for arg in sys.argv[3:]:
            key, _, val = arg.partition("=")
            kwargs[key] = float(val)
        export_for_load_test(sys.argv[1], **kwargs)
    else:
        print(
            f"Usage: python {sys.argv[0]} <similarities_dir> [--load-test [keep_fraction_candidates=X] [keep_fraction_queries=X]]"
        )
        sys.exit(1)
