"""Cost vs baseline scatter, gap histogram, cost vs number of targets (SPEC 12.2).

Static PNGs (matplotlib, Agg backend) over successful plans only. Each producer has a fixed
colour slot, so a colour means the same producer in every figure of the project, whichever
producers a run includes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Validated categorical palette (light surface), in fixed slot order.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PRODUCER_SLOTS = ("naive", "exact_search", "random", "bitscan", "tabular_q", "ppo", "mcts", "tree")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def colour(producer: str) -> str:
    if producer not in PRODUCER_SLOTS:
        raise ValueError(f"producer {producer!r} has no colour slot; add it to PRODUCER_SLOTS")
    return PALETTE[PRODUCER_SLOTS.index(producer)]


def _axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def _figure(ncols: int = 1, sharex: bool = False) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols + 1.2, 4), squeeze=False, sharex=sharex)
    fig.patch.set_facecolor(SURFACE)
    return fig, list(axes[0])


def _save(fig: plt.Figure, path: Path, title: str) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def cost_vs_baseline(results: pd.DataFrame, baseline: str, path: Path) -> Path:
    ok = results[results["success"]]
    base = ok[ok["producer"] == baseline].set_index("instance_id")["cost"]
    fig, (ax,) = _figure()
    hi = 0.0
    for producer, g in ok[ok["producer"] != baseline].groupby("producer", sort=False):
        g = g[g["instance_id"].isin(base.index)]
        x = base.loc[g["instance_id"]].to_numpy()
        ax.scatter(
            x,
            g["cost"],
            s=36,
            color=colour(str(producer)),
            label=producer,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
        hi = max([hi, *x, *g["cost"]])
    ax.plot(
        [0, hi * 1.05],
        [0, hi * 1.05],
        color=MUTED,
        linewidth=1,
        linestyle="--",
        label=f"= {baseline}",
        zorder=2,
    )
    _axes(ax, f"{baseline} cost", "producer cost")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path, f"Plan cost against {baseline} (below the line = cheaper)")


def gap_histogram(results: pd.DataFrame, path: Path) -> Path:
    ok = results[results["success"] & results["gap_to_best_known"].notna()]
    producers = list(ok["producer"].unique()) or ["(none)"]
    fig, axes = _figure(len(producers), sharex=True)  # one x range, so panels compare
    edges = list(np.histogram_bin_edges(ok["gap_to_best_known"], bins=16)) if len(ok) else 12
    for ax, producer in zip(axes, producers, strict=True):
        gaps = ok.loc[ok["producer"] == producer, "gap_to_best_known"]
        if len(gaps):
            ax.hist(gaps, bins=edges, color=colour(producer), edgecolor=SURFACE, linewidth=2)
        _axes(ax, "gap to best known", "instances")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(producer, color=INK, fontsize=10, loc="left")
    return _save(fig, path, "Gap to the stored best known cost (0 = matches it)")


def cost_vs_targets(results: pd.DataFrame, path: Path) -> Path:
    ok = results[results["success"]]
    producers = list(ok["producer"].unique())
    fig, (ax,) = _figure()
    for i, producer in enumerate(producers):
        g = ok[ok["producer"] == producer]
        offset = (i - (len(producers) - 1) / 2) * 0.08  # side by side, not on top
        ax.scatter(
            g["num_targets"] + offset,
            g["cost"],
            s=36,
            color=colour(producer),
            label=producer,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
    _axes(ax, "number of targets", "plan cost")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path, "Plan cost by number of targets")


def write_plots(results: pd.DataFrame, out_dir: str | Path, baseline: str | None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        gap_histogram(results, out / "gap_histogram.png"),
        cost_vs_targets(results, out / "cost_vs_targets.png"),
    ]
    if baseline is not None and (results["producer"] == baseline).any():
        paths.append(cost_vs_baseline(results, baseline, out / f"cost_vs_{baseline}.png"))
    return paths
