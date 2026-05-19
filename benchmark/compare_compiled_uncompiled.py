"""
Verify that compiled and uncompiled SentenceTransformer embeddings can be arbitrarily mixed in the prod vector DB. W/
v2.1 this is almost certainly fine b/c its distances live on a wider scale.

For each test pair we have incoming query embeddings and candidate DB embeddings from a compiled model (C) and from the
uncompiled model (U). Mid-migration, any of these four sims could be the prod similarity op:

    S_CC = sim(C_q, C_c)   S_CU = sim(C_q, U_c)
    S_UC = sim(U_q, C_c)   S_UU = sim(U_q, U_c)

The interesting failure case is pairs where S_CC and S_UU agree on the decision but a mixed configuration disagrees.

Reads `similarities/{dataset_name}/` (compiled) and `similarities-uncompiled/{dataset_name}/` (uncompiled) from a
run dir on GCS, computes all four sims at the prod truncation dim, and writes a local markdown report.

Example:

python benchmark/compare_compiled_uncompiled.py \
    --run_gcs_dir gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix
"""

import logging
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers.util import pairwise_cos_sim
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)

_COLUMNS_PAIR = ("query_stacktrace_string", "candidate_stacktrace_string")


def _sync_gcs(gcs_dir: str, dir_local: Path) -> None:
    """Sync a GCS similarities directory to a local cache."""
    dir_local.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing {gcs_dir} -> {dir_local}")
    subprocess.run(["gcloud", "storage", "rsync", "-r", gcs_dir, str(dir_local)], check=True)


def _load_side(dir_local: Path) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    df = pl.read_csv(dir_local / "similarities.csv")
    emb_query = np.load(dir_local / "query_embeddings.npy")
    emb_candidate = np.load(dir_local / "candidate_embeddings.npy")
    assert len(df) == len(emb_query) == len(emb_candidate), (
        f"Row count mismatch in {dir_local}: csv={len(df)} q={len(emb_query)} c={len(emb_candidate)}"
    )
    return df, emb_query, emb_candidate


def _cos_sim_truncated(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    return pairwise_cos_sim(torch.as_tensor(a[..., :dim]), torch.as_tensor(b[..., :dim])).detach().cpu().numpy()


def _percentiles(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def _df_to_markdown(df: pl.DataFrame) -> str:
    """Render a Polars DataFrame as a GitHub-flavored markdown table."""
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=10000,
        fmt_str_lengths=1000,
    ):
        return str(df)


def main(
    run_gcs_dir: str,
    dataset_name: str = "test_full3",
    dim: int = 64,
    thresholds: tuple[float, ...] = (0.06, 0.08),
    report_path: str = "/tmp/compare_compiled_uncompiled.md",
    include_outliers: bool = False,
    top_n_outliers: int = 20,
):
    """
    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory, e.g.,
        gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix.
        Must contain both `similarities/{dataset_name}/` and `similarities-uncompiled/{dataset_name}/`.
    dataset_name
        Subdirectory name under similarities/ and similarities-uncompiled/, e.g., "test_full3".
    dim
        Truncated embedding dimension. Defaults to 64 (V2.1 prod pgvector width).
    thresholds
        Cosine distance thresholds at which to evaluate decision agreement. Defaults to (0.06, 0.08), the V2.1
        platform-STRICT and STRICT thresholds. RELAXED (0.2) is excluded as out of scope.
    report_path
        Local path to write the markdown report.
    include_outliers
        If True, also compute and include a table of the top-N pairs by max sim shift vs S_CC. Off by default
        because shift outliers usually live far from the decision threshold and don't change grouping.
    top_n_outliers
        Number of worst-case pairs to include in the outlier table (only used if include_outliers=True).
    """
    gt.logging.configure_logging(process_type="compare_compiled_uncompiled")

    run_gcs_dir = run_gcs_dir.rstrip("/")
    run_name = run_gcs_dir.rsplit("/", 1)[-1]
    src_compiled = f"{run_gcs_dir}/similarities/{dataset_name}"
    src_uncompiled = f"{run_gcs_dir}/similarities-uncompiled/{dataset_name}"
    dir_compiled = Path("eval/similarities") / run_name / dataset_name
    dir_uncompiled = Path("eval/similarities-uncompiled") / run_name / dataset_name
    _sync_gcs(src_compiled, dir_compiled)
    _sync_gcs(src_uncompiled, dir_uncompiled)

    df_c, eq_c, ec_c = _load_side(dir_compiled)
    df_u, eq_u, ec_u = _load_side(dir_uncompiled)

    assert eq_c.shape[-1] >= dim and eq_u.shape[-1] >= dim, (
        f"Embeddings are narrower than dim={dim}: compiled={eq_c.shape[-1]} uncompiled={eq_u.shape[-1]}"
    )
    if len(df_c) != len(df_u):
        raise ValueError(f"Row count mismatch: compiled={len(df_c)} uncompiled={len(df_u)}")

    # Align by pair-key (defensive — both should be in the same order from load_val_df, but join makes that explicit).
    cols_needed = list(_COLUMNS_PAIR) + ["project_id", "platform", "label", f"cos_sim_{dim}"]
    df_c_small = df_c.select(cols_needed).with_row_index("idx_c")
    df_u_small = df_u.select(cols_needed).with_row_index("idx_u")
    df_joined = df_c_small.join(df_u_small, on=list(_COLUMNS_PAIR), how="inner", suffix="_u")
    if len(df_joined) != len(df_c):
        raise ValueError(
            f"Pair-key join shrank rows: compiled={len(df_c)} uncompiled={len(df_u)} joined={len(df_joined)}. "
            "Embeddings cannot be aligned safely."
        )

    idx_c = df_joined["idx_c"].to_numpy()
    idx_u = df_joined["idx_u"].to_numpy()
    eq_c = eq_c[idx_c]
    ec_c = ec_c[idx_c]
    eq_u = eq_u[idx_u]
    ec_u = ec_u[idx_u]

    logger.info(f"Computing 4 cosine-sim arrays at dim={dim} over {len(df_joined)} pairs")
    s_cc = _cos_sim_truncated(eq_c, ec_c, dim)
    s_uu = _cos_sim_truncated(eq_u, ec_u, dim)
    s_cu = _cos_sim_truncated(eq_c, ec_u, dim)
    s_uc = _cos_sim_truncated(eq_u, ec_c, dim)

    # Sanity: recomputed self-sims should match the CSV cos_sim_{dim} columns within float tolerance.
    csv_cc = df_joined[f"cos_sim_{dim}"].to_numpy()
    csv_uu = df_joined[f"cos_sim_{dim}_u"].to_numpy()
    sanity_diff_cc = float(np.max(np.abs(csv_cc - s_cc)))
    sanity_diff_uu = float(np.max(np.abs(csv_uu - s_uu)))
    sanity_ok = sanity_diff_cc < 1e-5 and sanity_diff_uu < 1e-5
    logger.info(
        f"Sanity recomputed-vs-CSV self-sim diffs: compiled={sanity_diff_cc:.2e} uncompiled={sanity_diff_uu:.2e} "
        f"(ok={sanity_ok})"
    )
    if not sanity_ok:
        raise RuntimeError(
            f"Recomputed self-sims don't match CSV cos_sim_{dim} columns within 1e-5: "
            f"compiled diff={sanity_diff_cc:.2e}, uncompiled diff={sanity_diff_uu:.2e}. "
            "Embedding alignment or truncation logic may be wrong."
        )

    # Distance and decision matrices, shape (N, 4) for [CC, UU, CU, UC].
    sims = np.stack([s_cc, s_uu, s_cu, s_uc], axis=-1)
    dists = 1 - sims

    diff_cc_uu = np.abs(s_cc - s_uu)
    mix_spread = np.ptp(sims, axis=-1)  # max - min across the 4 configs, per pair

    rows_threshold = []
    for t in thresholds:
        decisions = dists <= t  # (N, 4)
        d_cc = decisions[:, 0]
        d_uu = decisions[:, 1]
        all_agree = np.all(decisions == d_cc[:, None], axis=-1)
        any_disagreement = ~all_agree
        endpoint_flip = d_cc != d_uu
        mix_only = (d_cc == d_uu) & any_disagreement
        rows_threshold.append(
            {
                "threshold": t,
                "all_agree": int(all_agree.sum()),
                "any_disagreement": int(any_disagreement.sum()),
                "endpoint_flip": int(endpoint_flip.sum()),
                "mix_flip": int(mix_only.sum()),
            }
        )

    df_outliers: pl.DataFrame | None = None
    n_outliers = 0
    if include_outliers:
        n_outliers = min(top_n_outliers, len(df_joined))
        rank = np.argsort(-mix_spread)[:n_outliers]
        df_outliers = (
            df_joined.with_columns(
                [
                    pl.Series("S_CC", s_cc),
                    pl.Series("S_UU", s_uu),
                    pl.Series("S_CU", s_cu),
                    pl.Series("S_UC", s_uc),
                    pl.Series("mix_spread", mix_spread),
                ]
            )
            .with_row_index("idx_joined")
            .filter(pl.col("idx_joined").is_in(rank.tolist()))
            .with_columns(
                [
                    pl.col("query_stacktrace_string").str.slice(0, 80).alias("query_prefix"),
                    pl.col("candidate_stacktrace_string").str.slice(0, 80).alias("candidate_prefix"),
                ]
            )
            .sort("mix_spread", descending=True)
            .select(
                [
                    "project_id",
                    "platform",
                    "label",
                    "S_CC",
                    "S_UU",
                    "S_CU",
                    "S_UC",
                    "mix_spread",
                    "query_prefix",
                    "candidate_prefix",
                ]
            )
        )

    stats_cc_uu = _percentiles(diff_cc_uu)
    stats_spread = _percentiles(mix_spread)
    cols_stats = ("mean", "median", "p95", "p99", "max")
    df_shift = pl.DataFrame(
        {
            "metric": ["|S_CC - S_UU| (endpoint)", "max - min across 4 configs (worst-case mixed)"],
            **{col: [stats_cc_uu[col], stats_spread[col]] for col in cols_stats},
        }
    )

    n_pairs = len(df_joined)

    def _fmt_count_pct(n: int | float) -> str:
        return f"{int(n):,} ({n / n_pairs * 100:.3g}%)"

    df_threshold = pl.DataFrame(
        {
            "threshold": [r["threshold"] for r in rows_threshold],
            "all_4_agree": [_fmt_count_pct(r["all_agree"]) for r in rows_threshold],
            "any_disagreement": [_fmt_count_pct(r["any_disagreement"]) for r in rows_threshold],
            "endpoint_flip (CC vs UU)": [_fmt_count_pct(r["endpoint_flip"]) for r in rows_threshold],
            "mix_flip (endpoints agree, mixed differs)": [_fmt_count_pct(r["mix_flip"]) for r in rows_threshold],
        }
    )

    lines: list[str] = []
    lines.append("# Compiled vs Uncompiled Embedding Mix Analysis\n")
    lines.append(f"- **Run**: `{run_gcs_dir}`")
    lines.append(f"- **Dataset**: `{dataset_name}`")
    lines.append(f"- **Truncation dim**: `{dim}` (prod V2.1 pgvector width)")
    lines.append(f"- **Pairs analyzed**: `{len(df_joined):,}`\n")

    lines.append("## Sanity check\n")
    lines.append(
        f"Max |CSV `cos_sim_{dim}` − recomputed| should be tiny (<1e-5). "
        f"compiled={sanity_diff_cc:.2e}, uncompiled={sanity_diff_uu:.2e}. "
        f"**{'OK' if sanity_ok else 'FAIL'}**\n"
    )

    lines.append("## Similarity-shift distribution\n")
    lines.append(_df_to_markdown(df_shift))
    lines.append("")

    lines.append("## Decision agreement at thresholds\n")
    lines.append(
        "A pair is `grouped` if `1 - sim <= threshold`. Column meanings:\n\n"
        "- **any_disagreement** = pairs where at least one of the 4 configs disagrees with another. "
        'Upper bound on "a migration could change this pair\'s decision."\n'
        "- **endpoint_flip** = subset of any_disagreement where the two clean states (CC, UU) themselves "
        "disagree. Risk of toggling the flag globally.\n"
        "- **mix_flip** = subset of any_disagreement where (CC, UU) agree but at least one mixed config "
        "(CU, UC) disagrees. Pairs uniquely affected by mid-migration mixing.\n\n"
        "`any_disagreement = endpoint_flip + mix_flip` by construction.\n"
    )
    lines.append(_df_to_markdown(df_threshold))
    lines.append("")

    if df_outliers is not None:
        lines.append(f"## Top {n_outliers} outliers by mix_spread (max − min across 4 configs)\n")
        lines.append(_df_to_markdown(df_outliers))
        lines.append("")

    report = "\n".join(lines)
    Path(report_path).write_text(report)
    logger.info(f"Wrote report to {report_path}")
    print(report)


if __name__ == "__main__":
    tapify(main, description=__doc__)
