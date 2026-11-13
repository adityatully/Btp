"""Plan 8.2 / D-85 / Q-147: how far is list scheduling from the optimal schedule?

Per problem: the sharing pass's plan (fluid goal, `configs/costs/geometric.yaml`), its
operation graph on the default platform with `changeover_ticks = 0` (the ILP has no
changeover), list scheduling under `critical_path` and `plan_order`, and the ILP optimum
(`exact_makespan`, horizon = the better list makespan). An ILP that finds no schedule within
the limit is kept and counted (`ilp_no_answer`), with its proved lower bound; ratios use the
plans the ILP answered (its best, optimal or not).

Reading (D-85, fixed before the run): scheduling is done if the better list schedule averages
<= 1.05 x the optimum and is within 10% on >= 90% of plans. Otherwise: search over priorities.

Arguments are key=value: `sets` (comma list, default tiny_exact_v1 and val_s1_v2..val_s5_v2),
`limit` per set, `ilp_seconds` (60), `out` (records, jsonl). Summary only:
`summary=<records.jsonl>`.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.opgraph import build_opgraph
from meda.pipeline.platform import Platform
from meda.pipeline.schedule import ListScheduler, schedule_metrics, validate_schedule
from meda.pipeline.schedule_ilp import exact_makespan
from meda.solvers.baselines.share import ShareSolver

ROOT = Path(__file__).resolve().parents[1]
SETS = ["tiny_exact_v1", *(f"val_s{k}_v2" for k in range(1, 6))]


def weights() -> CostWeights:
    data = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/costs/geometric.yaml"))
    assert isinstance(data, dict)
    return CostWeights.from_dict({str(k): v for k, v in data.items()})


def record(
    name: str, inst: Instance, w: CostWeights, platform: Platform, ilp_seconds: float
) -> dict[str, Any]:
    """One problem's makespans: both list priorities and the ILP."""
    dag = ShareSolver(w).solve(inst).dag
    if dag is None:
        return {"set": name, "id": inst.instance_id, "status": "no plan"}
    g = build_opgraph(dag, inst, platform)
    out: dict[str, Any] = {"set": name, "id": inst.instance_id, "ops": len(g.ops)}
    for priority in ListScheduler.PRIORITIES:
        s = ListScheduler(priority).schedule(g, platform)
        validate_schedule(s, g, platform)
        out[priority] = schedule_metrics(s, g).makespan
    out["list"] = min(out[p] for p in ListScheduler.PRIORITIES)
    t0 = time.process_time()
    res = exact_makespan(g, platform, horizon=out["list"], time_limit=ilp_seconds)
    out["ilp"] = res.makespan
    out["optimal"] = res.optimal
    out["ilp_bound"] = res.lower_bound
    out["ilp_seconds"] = round(time.process_time() - t0, 3)
    out["status"] = "ok"
    return out


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The D-85 reading, overall and per set."""

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        timed_out = [r for r in rs if r["status"] == "ok" and r["ilp"] is None]
        ok = [r for r in rs if r["status"] == "ok" and r["ilp"] is not None]
        if not ok:
            return {"n": 0, "ilp_no_answer": len(timed_out)}
        ratio = [r["list"] / r["ilp"] for r in ok]
        cp_ratio = [r["critical_path"] / r["ilp"] for r in ok]
        po_ratio = [r["plan_order"] / r["ilp"] for r in ok]
        return {
            "n": len(ok),
            "no_plan": sum(r["status"] != "ok" for r in rs),
            "ilp_no_answer": len(timed_out),
            "ilp_no_answer_max_ratio_to_bound": max(
                (r["list"] / max(r["ilp_bound"], 1) for r in timed_out), default=None
            ),
            "not_optimal": sum(not r["optimal"] for r in ok),
            "mean_ratio": round(sum(ratio) / len(ok), 4),
            "within_10pct": round(sum(x <= 1.10 for x in ratio) / len(ok), 4),
            "at_optimum": sum(x == 1 for x in ratio),
            "max_ratio": round(max(ratio), 4),
            "mean_ratio_critical_path": round(sum(cp_ratio) / len(ok), 4),
            "mean_ratio_plan_order": round(sum(po_ratio) / len(ok), 4),
            "critical_path_better": sum(r["critical_path"] < r["plan_order"] for r in ok),
            "plan_order_better": sum(r["plan_order"] < r["critical_path"] for r in ok),
            "ilp_seconds_max": max(r["ilp_seconds"] for r in ok),
        }

    overall = block(records)
    passed = overall.get("n", 0) > 0 and (
        overall["mean_ratio"] <= 1.05 and overall["within_10pct"] >= 0.90
    )
    by_set = {
        name: block([r for r in records if r["set"] == name])
        for name in dict.fromkeys(r["set"] for r in records)
    }
    return {"overall": overall, "by_set": by_set, "scheduling_done": passed}


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        with open(args["summary"]) as fh:
            records = [json.loads(line) for line in fh]
        print(json.dumps(summarise(records), indent=2))
        return
    sets = args["sets"].split(",") if "sets" in args else SETS
    limit = int(args["limit"]) if "limit" in args else None
    ilp_seconds = float(args.get("ilp_seconds", 60))
    out = Path(args.get("out", "outputs/schedule_gap/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    platform = replace(Platform.load(), changeover_ticks=0)
    w = weights()
    records = []
    with open(out, "w") as fh:
        for name in sets:
            for inst in BenchmarkSet.read(name, BENCHMARKS_DIR).instances[:limit]:
                r = record(name, inst, w, platform, ilp_seconds)
                records.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
    print(json.dumps(summarise(records), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
