"""Compare training regimes from their run directories and apply the decision rule (SPEC 8.3).

    uv run python scripts/regime_report.py multirun/2026-10-23/22-00-00 [more dirs...] \
        --out notes/track2/regime_pilot.md

Each argument is a run directory of `meda.train train.source=regime` or a directory holding
several (a Hydra multirun). Per run it reads `.hydra/config.yaml` (regime, seed, design),
`metrics.jsonl` (the validation rounds, for forgetting) and the regime state (promotions) from
`regime.json`, or from `checkpoints/last.pt` while the run has not finished. Finished runs are
graded on `results.parquet` (the final grading of every stage's set).

`--at-step N` grades every run at its last validation round at or before env step N instead, so
unfinished or stopped runs compare at equal steps; `--at-step common` takes the latest round
every run has reached. Runs graded at different rounds are an error, never compared.

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


def _config(run: Path) -> Any:
    return OmegaConf.to_container(OmegaConf.load(run / ".hydra" / "config.yaml"))


def run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        found = (d.parent for d in p.glob("**/metrics.jsonl"))
        out += sorted(
            d
            for d in found
            if (d / ".hydra" / "config.yaml").exists() and _config(d)["train"]["source"] == "regime"
        )
    if not out:
        raise SystemExit(f"no regime runs (metrics.jsonl with train.source=regime) under {paths}")
    return out


def regime_state(run: Path) -> tuple[dict[str, Any], float]:
    """The regime's state and the env step it is complete up to: `regime.json` at the end of a
    run, else the last checkpoint (written right after a validation round)."""
    if (run / "regime.json").exists():
        return json.loads((run / "regime.json").read_text()), float("inf")
    ckpt = run / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise SystemExit(f"{run}: no regime.json and no checkpoint yet")
    import torch

    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    return data["extra"]["regime"], float(data["global_step"])


def validation_rounds(run: Path, stages: list[str]) -> dict[str, list[tuple[int, float, float]]]:
    """Per stage, the (env step, success rate, mean gap to best known) of every round."""
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    env_step = {r["step"]: int(r["global_step"]) for r in rows if r["prefix"] == "train"}
    rounds: dict[str, list[tuple[int, float, float]]] = {s: [] for s in stages}
    for r in rows:
        stage = r["prefix"][5:]
        if r["prefix"].startswith("eval/") and stage in rounds:
            gap = r["gap_to_best_known"]
            rounds[stage].append(
                (env_step[r["step"]], r["success_rate"], float("nan") if gap is None else gap)
            )
    return rounds


def latest_round(run: Path) -> int:
    state, _ = regime_state(run)
    steps = [s for rs in validation_rounds(run, list(state["stages"])).values() for s, _, _ in rs]
    if not steps:
        raise SystemExit(f"{run}: no validation round yet")
    return max(steps)


def read_run(run: Path, at_step: int | None = None) -> dict[str, Any]:
    cfg = _config(run)
    regime, complete_to = regime_state(run)
    stages = list(regime["stages"])
    weights = {s: float(cfg["stages"]["final_weights"][s]) for s in stages}
    total = sum(weights.values())
    rounds = validation_rounds(run, stages)
    events = regime["events"]
    if at_step is None:
        if not (run / "results.parquet").exists():
            raise SystemExit(f"{run}: unfinished (no results.parquet); compare with --at-step")
        results = pd.read_parquet(run / "results.parquet")
        env_steps = int(results["env_steps"].iloc[0])
        final = {}
        for stage, g in results.groupby("stage"):
            final[stage] = (float(g["success"].mean()), float(g["gap_to_best_known"].mean()))
    else:
        rounds = {s: [r for r in rs if r[0] <= at_step] for s, rs in rounds.items()}
        if not all(rounds.values()):
            raise SystemExit(f"{run}: no validation round at or before env step {at_step:,}")
        env_steps = min(rs[-1][0] for rs in rounds.values())
        if env_steps > complete_to:
            raise SystemExit(
                f"{run}: regime state covers env steps up to {complete_to:,.0f}, "
                f"not the round at {env_steps:,}"
            )
        final = {s: (rs[-1][1], rs[-1][2]) for s, rs in rounds.items()}
        events = [e for e in events if e["env_step"] <= env_steps]
    successes = {s: [r[1] for r in rs] for s, rs in rounds.items()}
    forgetting = {
        s: (max(successes[s]) - final[s][0]) if successes[s] else float("nan") for s in stages
    }
    gap = sum(weights[s] * final[s][1] for s in stages) / total
    return {
        "run": str(run),
        "regime": regime["regime"],
        "seed": int(cfg["seed"]),
        "env_steps": env_steps,
        "stages": stages,
        "final": final,
        "weighted_gap": gap,
        "max_forgetting": max(forgetting.values()),
        "forgetting": forgetting,
        "events": events,
    }


def read_runs(dirs: list[Path], at_step: str | None) -> list[dict[str, Any]]:
    """Every run, graded at its end or at one shared validation round (`--at-step`)."""
    if at_step is None:
        return [read_run(d) for d in dirs]
    step = min(latest_round(d) for d in dirs) if at_step == "common" else int(at_step)
    runs = [read_run(d, step) for d in dirs]
    graded = {r["run"]: r["env_steps"] for r in runs}
    if len(set(graded.values())) > 1:
        listing = ", ".join(f"{run} at {s:,}" for run, s in graded.items())
        raise SystemExit(f"runs reach different validation rounds by step {step:,}: {listing}")
    return runs


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


def report(runs: list[dict[str, Any]], min_success: float, at_step: str | None = None) -> str:
    table, decision = decide(runs, min_success)
    graded = (
        "each run's final grading"
        if at_step is None
        else f"the validation round at env step {runs[0]['env_steps']:,} (runs not finished)"
    )
    lines = ["# Training regime comparison", "", f"Graded on {graded}.", ""]
    lines += [f"**Decision:** {decision}", ""]
    lines += [markdown_table(list(table.columns), table.values.tolist()), "", "## Runs", ""]
    for r in sorted(runs, key=lambda r: (r["regime"], r["seed"])):
        lines.append(f"### {r['regime']}, seed {r['seed']} ({r['env_steps']:,} env steps)")
        lines.append(f"`{r['run']}`")
        lines.append("")
        lines.append("| stage | success | mean gap to best known | forgetting |")
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
    parser.add_argument("--at-step", help="an env step, or 'common': grade at that round")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = report(read_runs(run_dirs(args.paths), args.at_step), args.min_success, args.at_step)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
