"""Gap tables: to optimum and to each baseline (SPEC 12.2), and `summary.md`.

Statistics are taken over successful plans only, and every table carries the success rate next
to them, so a producer cannot look cheap by failing the hard instances. Each summary states its
benchmark tier (SPEC 8.4, 12.3): a number answers only its tier's question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from meda.analysis.evaluate import relative_gap
from meda.core.composition import Rational

# Name prefix of a frozen set -> its tier (SPEC 8.4).
TIERS = {
    "tiny_exact": "0 Correctness",
    "val_stage": "1 In-distribution",
    "dmfb_sweep": "2 Engine check (DMFB)",
    "literature_": "3 Literature",
    "unseen_ratios": "4 Generalisation",
    "larger": "4 Generalisation",
    "frontier": "5 Frontier",
    "assays": "6 Assays",
}

SUMMARY_COLUMNS = (
    "cost",
    "gap_to_optimal",
    "gap_to_best_known",
    "gap_to_lower_bound",
    "mix_ops",
    "total_reagent",
    "sample_volume",
    "waste_volume",
    "distinct_intermediates",
    "sharing_count",
    "makespan",
    "critical_path",
    "peak_droplets",
    "storage_slots",
    "wall_time",
)


def tier_of(benchmark: str) -> str:
    for prefix, tier in TIERS.items():
        if benchmark.startswith(prefix):
            return tier
    raise ValueError(f"benchmark {benchmark!r} has no tier in SPEC 8.4; add it to TIERS")


def add_baseline_gaps(results: pd.DataFrame, baselines: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Add `gap_to_<name>` for each baseline: exact, where both plans succeeded."""
    out = results.copy()
    for name, base in baselines.items():
        ok = base[base["success"]].set_index("instance_id")["cost_exact"]
        gaps: list[float | None] = []
        for iid, success, cost in zip(
            out["instance_id"], out["success"], out["cost_exact"], strict=True
        ):
            if not success or iid not in ok.index:
                gaps.append(None)
            else:
                gaps.append(relative_gap(Rational.parse(cost), Rational.parse(ok[iid]), iid))
        out[f"gap_to_{name}"] = gaps
    return out


def _columns(results: pd.DataFrame) -> list[str]:
    extra = sorted(
        c for c in results.columns if c.startswith("gap_to_") and c not in SUMMARY_COLUMNS
    )
    cols = [c for c in SUMMARY_COLUMNS if c in results.columns]
    return cols[:4] + extra + cols[4:]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (producer, metric) with n, mean, median, p90 over successes."""
    rows: list[dict[str, Any]] = []
    for producer, group in results.groupby("producer", sort=False):
        ok = group[group["success"]]
        for col in _columns(results):
            values = pd.to_numeric(ok[col], errors="coerce").dropna()
            rows.append(
                {
                    "producer": producer,
                    "metric": col,
                    "n": len(values),
                    "mean": values.mean() if len(values) else None,
                    "median": values.median() if len(values) else None,
                    "p90": values.quantile(0.9) if len(values) else None,
                }
            )
    return pd.DataFrame(rows)


def _cols(df: pd.DataFrame, *names: str) -> list[list[Any]]:
    return [df[n].tolist() for n in names]


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_summary_md(path: str | Path, results: pd.DataFrame) -> None:
    """`summary.md` for one benchmark and any number of producers."""
    (benchmark,) = results["benchmark"].unique()
    producers = list(results["producer"].unique())
    parts = [
        f"# Evaluation on `{benchmark}`",
        "",
        f"Benchmark tier: **{tier_of(benchmark)}**. Statistics are over successful plans only.",
        "",
        "## Overview",
        "",
    ]
    overview = []
    for p in producers:
        g = results[results["producer"] == p]
        overview.append(
            [
                p,
                len(g),
                f"{g['success'].mean():.0%}",
                int((g["gap_to_optimal"] == 0).sum()),
                int(g["gap_to_optimal"].notna().sum()),
                g["gap_to_optimal"].mean(),
                g["gap_to_best_known"].mean(),
            ]
        )
    parts.append(
        markdown_table(
            [
                "producer",
                "instances",
                "success",
                "at optimum",
                "with optimum",
                "mean gap to optimal",
                "mean gap to best known",
            ],
            overview,
        )
    )
    summary = summarize(results)
    parts += ["", "## Metrics (n, mean, median, p90)", ""]
    parts.append(
        markdown_table(
            ["producer", "metric", "n", "mean", "median", "p90"],
            summary[["producer", "metric", "n", "mean", "median", "p90"]].values.tolist(),
        )
    )
    cost, success, status = _cols(results, "cost_exact", "success", "status")
    shown = results.assign(
        shown=[c if ok else st for c, ok, st in zip(cost, success, status, strict=True)]
    )
    costs = shown.pivot(index="instance_id", columns="producer", values="shown")
    ref = results.drop_duplicates("instance_id").set_index("instance_id")
    parts += ["", "## Exact cost per instance", ""]
    parts.append(
        markdown_table(
            ["instance", "reference", "ref status", *producers],
            [
                [iid, ref.at[iid, "reference_cost"], ref.at[iid, "reference_status"]]
                + [costs.at[iid, p] for p in producers]
                for iid in ref.index
            ],
        )
    )
    Path(path).write_text("\n".join(parts) + "\n")
