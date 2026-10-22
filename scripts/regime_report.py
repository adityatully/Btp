"""Compare training regimes from their run directories and apply the decision rule (SPEC 8.3).

    uv run python scripts/regime_report.py multirun/2026-10-23/22-00-00 [more dirs...] \
        --out notes/track2/regime_pilot.md

Each argument is a run directory of `meda.train train.source=regime` or a directory holding
several (a Hydra multirun). Per run it reads `.hydra/config.yaml` (regime, seed, design),
`results.parquet` (final grading on every stage's set and the held-out set), `metrics.jsonl`
(the validation rounds, for forgetting) and `regime.json` (promotions).

Decision rule, fixed before the runs (SPEC 8.3):
1. Reject a regime whose mean final greedy success over seeds is below `--min-success` on any
   stage's validation set.
2. Among the rest, pick the lowest weighted mean gap to the best known plan (stage weights =
   the design's final weights).
3. If two regimes' seed ranges of that gap overlap, prefer the simpler one (mixed, then
   expanding, then sequential), then the one that forgets less.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from meda.analysis.compare import markdown_table

SIMPLICITY = {"mixed": 0, "expanding": 1, "sequential": 2}


def run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out += sorted(d.parent for d in p.glob("**/regime.json"))
    if not out:
        raise SystemExit(f"no finished regime runs (regime.json) under {paths}")
    return out


def read_run(run: Path) -> dict[str, Any]:
    cfg: Any = OmegaConf.to_container(OmegaConf.load(run / ".hydra" / "config.yaml"))
    results = pd.read_parquet(run / "results.parquet")
    regime = json.loads((run / "regime.json").read_text())
    stages = list(regime["stages"])
    weights = {s: float(cfg["stages"]["final_weights"][s]) for s in stages}
    total = sum(weights.values())
    final = {}
    for stage, g in results.groupby("stage"):
        final[stage] = (float(g["success"].mean()), float(g["gap_to_best_known"].mean()))
    rounds: dict[str, list[float]] = {s: [] for s in stages}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        prefix = row["prefix"]
        if prefix.startswith("eval/") and prefix[5:] in rounds:
            rounds[prefix[5:]].append(row["success_rate"])
    forgetting = {s: (max(rounds[s]) - final[s][0]) if rounds[s] else float("nan") for s in stages}
    gap = sum(weights[s] * final[s][1] for s in stages) / total
    return {
        "run": str(run),
        "regime": regime["regime"],
        "seed": int(cfg["seed"]),
        "env_steps": int(results["env_steps"].iloc[0]),
        "stages": stages,
        "final": final,
        "weighted_gap": gap,
        "max_forgetting": max(forgetting.values()),
        "forgetting": forgetting,
        "events": regime["events"],
    }


def decide(runs: list[dict[str, Any]], min_success: float) -> tuple[pd.DataFrame, str]:
    rows = []
    for name, group in pd.DataFrame(runs).groupby("regime"):
        stages = group["stages"].iloc[0]
        success = {s: group["final"].map(lambda f, s=s: f[s][0]).mean() for s in stages}
        rows.append(
            {
                "regime": name,
                "seeds": len(group),
                "worst_stage_success": min(success.values()),
                "gap_mean": group["weighted_gap"].mean(),
                "gap_min": group["weighted_gap"].min(),
                "gap_max": group["weighted_gap"].max(),
                "max_forgetting": group["max_forgetting"].mean(),
                "forced_promotions": sum(sum(e["forced"] for e in ev) for ev in group["events"]),
            }
        )
    # No successful plan on a stage leaves its gap NaN; rank that as the worst possible gap.
    table = pd.DataFrame(rows).fillna(
        {"gap_mean": float("inf"), "gap_min": float("inf"), "gap_max": float("inf")}
    )
    table = table.sort_values("gap_mean").reset_index(drop=True)
    ok = table[table["worst_stage_success"] >= min_success]
    if ok.empty:
        return table, f"no regime reaches {min_success:.0%} success on every stage: none chosen"
    best = ok.iloc[0]
    tied = ok[(ok["gap_min"] <= best["gap_max"]) & (ok["gap_max"] >= best["gap_min"])]
    pick = min(
        tied.to_dict("records"), key=lambda r: (SIMPLICITY[r["regime"]], r["max_forgetting"])
    )
    others = sorted(set(tied["regime"]) - {pick["regime"]})
    why = (
        f"'{pick['regime']}': lowest weighted gap among regimes passing every stage"
        if not others
        else f"'{pick['regime']}': gap ranges overlap with {others}, so the simpler regime wins"
    )
    return table, why


def report(runs: list[dict[str, Any]], min_success: float) -> str:
    table, decision = decide(runs, min_success)
    lines = ["# Training regime comparison", "", f"**Decision:** {decision}", ""]
    lines += [markdown_table(list(table.columns), table.values.tolist()), "", "## Runs", ""]
    for r in sorted(runs, key=lambda r: (r["regime"], r["seed"])):
        lines.append(f"### {r['regime']}, seed {r['seed']} ({r['env_steps']:,} env steps)")
        lines.append(f"`{r['run']}`")
        lines.append("")
        lines.append("| stage | final success | mean gap to best known | forgetting |")
        lines.append("|---|---|---|---|")
        for s, (succ, gap) in r["final"].items():
            forget = r["forgetting"].get(s)
            lines.append(
                f"| {s} | {succ:.0%} | {gap:.3g} | {'-' if forget is None else f'{forget:.0%}'} |"
            )
        for e in r["events"]:
            mark = " (forced)" if e["forced"] else ""
            lines.append(f"- step {e['env_step']:,}: {e['stage']} {e['event']}{mark}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--min-success", type=float, default=0.95)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = report([read_run(d) for d in run_dirs(args.paths)], args.min_success)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
