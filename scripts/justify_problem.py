"""J8 of the justification sweeps (D-98): do the final runs' verdicts hinge on the cost weights
or the step limit (E2, E3 in `notes/justification_audit.md`)?

The search runs and our 4 planner runs come from `scripts/ablation.py runs=J8` (method-v3 and
the planners under each profile: `configs/costs/sensitivity/*.yaml`, or the step limit scaled by
0.75 or 2, never below the witness plan). A published method's plan does not depend on the
weights, so each is made once per subset problem (`published=build`) and its totals (units
per fluid, mixes, waste, steps) are costed under every profile; under a scaled step limit a
plan longer than the limit does not count.

**Reading, fixed in advance (D-98):** per profile, our cost over the best of the 4 planner runs
and over each published method (on the problems it accepts), paired, with D-69's verdict
(beats: 95% bootstrap CI below 1 and losses <= 10%; loses: CI above 1; else level). A verdict
that differs from the base profile's (`j8_base`) is listed. The verdicts hold if none changes
from beats to level or loses.

    OMP_NUM_THREADS=1 uv run python scripts/justify_problem.py published=build workers=16 \\
        out=outputs/ablation/j8_published.jsonl
    uv run python scripts/justify_problem.py summary=outputs/ablation/j8.jsonl \\
        published=outputs/ablation/j8_published.jsonl
"""

from __future__ import annotations

import json
import signal
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ablation as ab  # noqa: E402
import bench  # noqa: E402
from meda.core.costs import CostWeights, StepCost  # noqa: E402
from meda.core.rules import IllegalActionError, replay  # noqa: E402
from meda.generators.benchmark_io import BenchmarkSet  # noqa: E402
from meda.solvers.baselines.paper_graph import NotApplicable, emit  # noqa: E402

METHOD_CAP = 900.0  # CPU s per published method and problem (NFSP's ILP: 600 s per solve)


def make_published(entry: dict[str, Any]) -> dict[str, Any]:
    """Every published method on one subset problem: its plan's totals, or why not."""
    bench.nfsp_module.TIME_LIMIT = 600.0
    inst = BenchmarkSet.read(entry["set"]).by_id(entry["id"])
    out: dict[str, Any] = {"id": entry["id"], "set": entry["set"], "methods": {}}
    for name, make in bench.PUBLISHED.items():
        signal.signal(signal.SIGPROF, ab.jp._alarm)
        signal.setitimer(signal.ITIMER_PROF, METHOD_CAP)
        try:
            plan = emit(inst, make(inst))
            _, total, _ = replay(inst, plan)
            out["methods"][name] = {
                "reagent_by_fluid": list(total.reagent_by_fluid),
                "mix_ops": total.mix_ops,
                "waste_volume": total.waste_volume,
                "outputs": total.outputs,
                "steps": len(plan),
            }
        except NotApplicable as e:
            out["methods"][name] = {"na": str(e)[:200]}
        except ab.jp._Cap:
            out["methods"][name] = {"na": f"over {METHOD_CAP:g} CPU s"}
        except (IllegalActionError, ValueError, AssertionError) as e:
            out["methods"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
    return out


def profile_weights(cfg: dict[str, Any]) -> CostWeights:
    if "costs" not in cfg:
        return ab._BASE["weights"]
    return CostWeights.from_dict(yaml.safe_load((ROOT / "configs" / cfg["costs"]).read_text()))


def method_cost(totals: dict[str, Any], w: CostWeights, inst: Any, limit: int) -> float | None:
    if "reagent_by_fluid" not in totals or totals["steps"] > limit:
        return None
    cost = StepCost(
        tuple(totals["reagent_by_fluid"]),
        totals["mix_ops"],
        totals["waste_volume"],
        totals["outputs"],
    )
    return float(w.scalar(cost, inst.fluids))


def summarise(runs: Path, published: Path) -> str:
    pub = {}
    for line in published.read_text().splitlines():
        r = json.loads(line)
        pub[r["id"]] = r["methods"]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in runs.read_text().splitlines():
        r = json.loads(line)
        if r["config"] in ab.J8 and r["cost"] is not None:
            by[r["config"]].append(r)
    verdicts: dict[str, dict[str, str]] = {}
    lines = [
        "| profile | against | n | w / t / l | mean ours / theirs [95% CI] | median | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for prof in ab.J8:
        rs = by.get(prof, [])
        if not rs:
            continue
        w = profile_weights(ab.J8[prof])
        verdicts[prof] = {}
        planner = [
            (r["cost"], min(b["cost"] for b in r["baselines"].values() if b["cost"] is not None))
            for r in rs
            if r["baselines"] and any(b["cost"] is not None for b in r["baselines"].values())
        ]
        rows = [("best planner run", planner)]
        for m in bench.PUBLISHED:
            pairs = []
            for r in rs:
                totals = pub.get(r["id"], {}).get(m)
                if totals is None:
                    continue
                inst = BenchmarkSet.read(r["set"]).by_id(r["id"])
                theirs = method_cost(totals, w, inst, r["step_limit"])
                if theirs:
                    pairs.append((r["cost"], theirs))
            rows.append((m, pairs))
        for name, pairs in rows:
            if len(pairs) < 10:
                continue
            p = bench.paired([a for a, _ in pairs], [b for _, b in pairs])
            v = bench.verdict(p)
            verdicts[prof][name] = v
            lines.append(f"| {prof} | {name} | {bench._fmt(p)} | {v} |")
    base = verdicts.get("j8_base", {})
    changed = [
        f"- {prof}, {name}: {base[name]} -> {v}"
        for prof, vs in verdicts.items()
        if prof != "j8_base"
        for name, v in vs.items()
        if name in base and v != base[name]
    ]
    lines += ["", "Verdicts that differ from the base profile (D-98):", *(changed or ["- none"])]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"]), Path(args["published"])))
        return
    if args.get("published") != "build":
        raise SystemExit("published=build out=... or summary=... published=...")
    entries = json.loads(ab.SUBSET.read_text())
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text().splitlines()}
    todo = [e for e in entries if e["id"] not in done]
    if "limit" in args:
        todo = todo[: int(args["limit"])]
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems, {workers} workers", flush=True)
    with out.open("a") as fh:
        pool = Pool(workers) if workers > 1 else None
        results = pool.imap_unordered(make_published, todo) if pool else map(make_published, todo)
        for i, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ok = sum("reagent_by_fluid" in m for m in rec["methods"].values())
            print(f"[{i}/{len(todo)}] {rec['id']}: {ok} methods planned", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
