"""Plan 8.3 / D-86 / Q-145: does sharing intermediates cost run time on the chip?

Per problem of the paper test sets: each producer's plan is replayed, validated, turned into
chip steps (`meda.pipeline.build_opgraph`) and scheduled by list scheduling, the better of the
two priorities (E-87), on three platforms: the default (`configs/platform/meda_default.yaml`),
`slow_mix` (mix 4 ticks) and `slow_detect` (detect 8, changeover 4). Per plan and platform:
makespan, storage_ticks, storage_peak, detector_wait; per plan: fluid cost (fluid goal),
sample units, mixes, sharing count (SPEC 12.1).

Producers: `construction` (the cheaper of naive and split tree: no droplet shared between
targets), `share` (the plain sharing pass), `share_mrcm_look3` (our
strongest planner variant), the fast published heuristics (`methods`, default DFM, FacDA,
Generic, DFMM, MRCM, e-MRCM), and with `method=1` our method (method-v3 search, `seconds` of
CPU, default 60: the paper's short column).

Reading (D-86, fixed before the run): per comparator, paired on the problems both solve, the
makespan ratio ours / comparator on the default platform, stages 3-5 + held-out: *costs time*
if mean > 1.05 with the 95% CI above 1.00; *no cost* if the CI lies within [0.95, 1.05];
*saves time* if mean < 0.95 with the CI below 1.00; else *unclear*. "Ours" is `method` when
present, else `share_mrcm_look3`.

    OMP_NUM_THREADS=1 uv run python scripts/sharing_time.py workers=4 out=outputs/st/r.jsonl
    uv run python scripts/sharing_time.py summary=outputs/st/r.jsonl

Arguments are key=value: `sets` (comma list, default the six test sets), `limit` per set,
`methods` (comma list of published methods), `method` (0/1), `seconds`, `sims` (256),
`workers`, `out`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.analysis.dag_metrics import _sharing_count
from meda.core.actions import Action
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import FluidKind
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.pipeline.opgraph import build_opgraph
from meda.pipeline.platform import Platform
from meda.pipeline.schedule import ListScheduler, schedule_metrics, validate_schedule
from meda.solvers.baselines import share
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from meda.solvers.fallback import cheapest_construction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import lit_bench as lb  # noqa: E402
import lit_bench_full as lbf  # noqa: E402
import search_check as sc  # noqa: E402

SETS = ("test_s1_v1", "test_s2_v1", "test_s3_v1", "test_s4_v1", "test_s5_v1", "test_heldout_v1")
MANY = ("test_s3_v1", "test_s4_v1", "test_s5_v1", "test_heldout_v1")  # many targets
FAST = ("dfm", "facda", "generic", "dfmm", "mrcm", "emrcm")
PUBLISHED = {**lb.LITERATURE, **lbf.METHODS}
COMPARATORS = ("construction", "share", *FAST)
_BASE = Platform.load()
PLATFORMS = {
    "default": _BASE,
    "slow_mix": replace(_BASE, name="slow_mix", mix_ticks=4),
    "slow_detect": replace(_BASE, name="slow_detect", detect_ticks=8, changeover_ticks=4),
}
BOOT, CI = 10_000, 0.95


def timing(inst: Instance, name: str, plan: tuple[Action, ...]) -> dict[str, Any]:
    """A validated plan's fluid totals and its schedule on each platform."""
    _, totals, deltas = replay(inst, plan)
    dag = MixingDAG.from_deltas(inst, deltas, totals, name)
    validate(dag, inst).raise_if_invalid()
    w = gc.use_goal("fluid")
    out: dict[str, Any] = {
        "cost": float(w.scalar(totals, inst.fluids)),
        "sample": sum(
            totals.reagent_by_fluid[f.id] for f in inst.fluids if f.kind is FluidKind.SAMPLE
        ),
        "mixes": totals.mix_ops,
        "sharing": _sharing_count(dag),
    }
    for pname, platform in PLATFORMS.items():
        g = build_opgraph(dag, inst, platform)
        best = None
        for priority in ListScheduler.PRIORITIES:
            s = ListScheduler(priority).schedule(g, platform)
            validate_schedule(s, g, platform)
            m = schedule_metrics(s, g)
            if best is None or (m.makespan, m.storage_ticks) < (best.makespan, best.storage_ticks):
                best = m
        assert best is not None
        out[pname] = {
            "makespan": best.makespan,
            "storage_ticks": best.storage_ticks,
            "storage_peak": best.storage_peak,
            "detector_wait": best.detector_wait,
        }
    return out


def plans(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, tuple[Action, ...] | str]:
    """Each producer's plan, or why there is none."""
    inst = Instance.from_json(task["instance"])
    w = gc.use_goal("fluid")
    out: dict[str, tuple[Action, ...] | str] = {}
    made_c = cheapest_construction(inst, w)
    out["construction"] = made_c[0] if made_c is not None else "no plan"
    made = share.share_plan(inst, w)
    out["share"] = made if made is not None else "no plan"
    made = share.share_plan(inst, w, lookahead=3, mrcm=True)
    out["share_mrcm_look3"] = made if made is not None else "no plan"
    for name in opts["methods"]:
        try:
            made = emit(inst, PUBLISHED[name](inst))
            out[name] = made if made is not None else "no plan"
        except NotApplicable as e:
            out[name] = f"na: {e}"[:200]
        except (IllegalActionError, ValueError, AssertionError) as e:
            out[name] = f"error: {type(e).__name__}: {e}"[:200]
    if opts["method"]:
        found = [
            float(w.scalar(replay(inst, p)[1], inst.fluids))
            for p in out.values()
            if not isinstance(p, str)
        ]
        task_ = {
            "set": task["set"],
            "id": inst.instance_id,
            "cell": task["cell"],
            "instance": task["instance"],
            "optimum": None,
            "planner": min(found),
            "mrcm": None,
            "greedy": None,
        }
        sopts = {
            "agent": None,
            "seconds": opts["seconds"],
            "hints": "planner",
            "reward_scale": 0.01,
            "keep_plan": True,
            "switch_seconds": sc.AUTO_LIMIT,
            "root_switch_seconds": sc.AUTO_ROOT_LIMIT,
            "jumps": 0,
            "jump_share": 0.5,
            "tier_share": sc.TIER_SHARE,
            "switch_mean": sc.SWITCH_MEAN,
            "switch_min_leaves": sc.SWITCH_MIN_LEAVES,
            "first_plan": True,
        }
        rec = sc.run_task(task_, "uniform", opts["sims"], sopts, "switch")
        out["method"] = tuple(rec["plan"]) if rec["plan"] else "no plan"
    return out


def run(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    rec: dict[str, Any] = {
        "set": task["set"],
        "id": inst.instance_id,
        "T": len(inst.targets),
        "S": inst.chip.storage_capacity,
        "producers": {},
    }
    for name, plan in plans(task, opts).items():
        rec["producers"][name] = (
            {"none": plan} if isinstance(plan, str) else timing(inst, name, plan)
        )
    return rec


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run(*args)


def boot_ci(x: list[float], seed: int = 0) -> tuple[float, float]:
    a = np.array(x)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (BOOT, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - CI) / 2, (1 + CI) / 2])
    return float(lo), float(hi)


def verdict(mean: float, lo: float, hi: float) -> str:
    """D-86's rule."""
    if mean > 1.05 and lo > 1.00:
        return "costs time"
    if lo >= 0.95 and hi <= 1.05:
        return "no cost"
    if mean < 0.95 and hi < 1.00:
        return "saves time"
    return "unclear"


def compare(
    recs: list[dict[str, Any]], ours: str, other: str, platform: str = "default"
) -> dict[str, Any]:
    """Paired ratios ours / other over the problems both solve."""
    t, c, st = [], [], []
    for r in recs:
        a, b = r["producers"].get(ours, {}), r["producers"].get(other, {})
        if platform not in a or platform not in b:
            continue
        t.append(a[platform]["makespan"] / b[platform]["makespan"])
        c.append(a["cost"] / b["cost"])
        st.append((a[platform]["storage_ticks"] + 1) / (b[platform]["storage_ticks"] + 1))
    if not t:
        return {"n": 0}
    lo, hi = boot_ci(t)
    mean = sum(t) / len(t)
    return {
        "n": len(t),
        "time_ratio": round(mean, 4),
        "time_ci": [round(lo, 4), round(hi, 4)],
        "verdict": verdict(mean, lo, hi),
        "cost_ratio": round(sum(c) / len(c), 4),
        "storage_ratio": round(sum(st) / len(st), 4),
        "slower": sum(x > 1 for x in t),
        "faster": sum(x < 1 for x in t),
    }


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ours = "method" if any("method" in r["producers"] for r in recs) else "share_mrcm_look3"
    groups = {
        "many_targets": [r for r in recs if r["set"] in MANY],
        "s1_s2": [r for r in recs if r["set"] not in MANY],
    }
    out: dict[str, Any] = {"ours": ours}
    for gname, rs in groups.items():
        out[gname] = {
            other: {p: compare(rs, ours, other, p) for p in PLATFORMS}
            for other in COMPARATORS
            if other != ours
        }
    default = {
        o: v["default"]["verdict"] for o, v in out["many_targets"].items() if v["default"]["n"]
    }
    robust = {
        o: all(out["many_targets"][o][p]["verdict"] == default[o] for p in PLATFORMS)
        for o in default
    }
    out["reading"] = {"verdict": default, "robust": robust}
    return out


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        with open(args["summary"]) as fh:
            recs = [json.loads(line) for line in fh]
        print(json.dumps(summarise(recs), indent=2))
        return
    sets = args["sets"].split(",") if "sets" in args else list(SETS)
    limit = int(args["limit"]) if "limit" in args else None
    opts = {
        "methods": args["methods"].split(",") if "methods" in args else list(FAST),
        "method": args.get("method", "0") == "1",
        "seconds": float(args.get("seconds", 60)),
        "sims": int(args.get("sims", 256)),
    }
    out = Path(args.get("out", "outputs/sharing_time/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "set": s,
            "cell": i.metadata.get("cell") or i.metadata.get("stage", ""),
            "instance": i.to_json(),
        }
        for s in sets
        for i in BenchmarkSet.read(s).instances[:limit]
    ]
    recs = []
    with open(out, "w") as fh, Pool(int(args.get("workers", 1))) as pool:
        for r in pool.imap_unordered(_star, [(t, opts) for t in tasks]):
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{len(recs)}/{len(tasks)} {r['id']}", flush=True)
    print(json.dumps(summarise(recs), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
