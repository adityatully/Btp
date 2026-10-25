"""Teacher study (paper log E-15): how far each plan producer is from the optimum or a bound.

Three kinds of evidence, each graded against the best reference it has:
- `e2`: the small Track 2 problems of `notes/track2/e2_headroom.jsonl`, regenerated from their
  seeds by `scripts/headroom.py`; only the 65 with a proved optimum are graded, against it.
- `e3`: the 45 medium problems of `e3_grading.jsonl` (E-6), graded against the larger of the
  bound exact search proved there and today's root bound (SPEC 0.21); both are admissible on
  these exact targets.
- `val`: the frozen `val_s1_v2` ... `val_s5_v2` and `heldout_combo_v2` sets, graded against
  their stored lower bound (the cost itself where it is proved optimal).

Every plan is replayed by the engine and its DAG validated before its cost counts. Each
regenerated problem is checked against the record (M, storage, targets) and skipped if it does
not match, so a changed generator cannot slip in a different problem.

    uv run python scripts/teacher_study.py which=e2,e3,val producers=teacher,share \
        out=notes/teacher_study/records.jsonl
    uv run python scripts/teacher_study.py summary=notes/teacher_study/records.jsonl

Arguments are key=value. `workers` (default 1) runs problems in parallel; on the Mac keep 1.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.actions import Action
from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.exact_search import fluid_bound_heuristic
from meda.solvers.fallback import cheapest_construction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom  # noqa: E402

WEIGHTS: CostWeights = headroom.WEIGHTS
VAL_SETS = ("val_s1_v2", "val_s2_v2", "val_s3_v2", "val_s4_v2", "val_s5_v2", "heldout_combo_v2")


def _teacher(inst: Instance) -> tuple[Action, ...] | None:
    best = cheapest_construction(inst, WEIGHTS)
    return None if best is None else best[0]


def _share(inst: Instance) -> tuple[Action, ...] | None:
    from meda.solvers.baselines.share import share_plan

    return share_plan(inst, WEIGHTS)


def _share_look(inst: Instance) -> tuple[Action, ...] | None:
    from meda.solvers.baselines.share import share_plan

    return share_plan(inst, WEIGHTS, lookahead=3)


PRODUCERS: dict[str, Callable[[Instance], tuple[Action, ...] | None]] = {
    "teacher": _teacher,
    "share": _share,
    "share_look3": _share_look,
}


def e2_stage(cell: str) -> str:
    """The stage of `track2_v1` whose cells an E2 cell resembles (SPEC 8.3)."""
    kind, targets, demand, storage = cell.split("-")
    if storage == "tight":
        return (
            "heldout"
            if kind == "dilgrad" or (kind == "dil" and targets == "many" and demand == "dN")
            else "s5"
        )
    if targets == "many":
        return "s4"
    if demand == "dN":
        return "s3"
    return "s1" if kind == "dil" else "s2"


def _matches(inst: Instance, rec: dict[str, Any]) -> bool:
    d = headroom.describe(inst)
    keys = ("M", "storage", "F", "T", "dens", "volumes", "demands")
    return all(d[k] == rec[k] for k in keys)


def _regenerate(rec: dict[str, Any], size: str) -> Instance | None:
    cfgs = {name: (cfg, grad) for name, cfg, grad in headroom.cells(size)}
    cfg, grad = cfgs[rec["cell"]]
    import zlib

    from meda.generators.sampler import GeneratorConfig, sample_instance

    rng = np.random.default_rng([zlib.crc32(rec["cell"].encode()), rec["index"]])
    inst = sample_instance(GeneratorConfig.from_dict(cfg), headroom.CHIP, rng, WEIGHTS)
    if grad:
        inst = headroom.as_gradient(inst, rng)
    inst = Instance(
        f"{rec['mode']}-{rec['cell']}-{rec['index']}",
        inst.fluids,
        inst.targets,
        inst.chip,
        inst.step_limit,
        inst.metadata,
    )
    return inst if _matches(inst, rec) else None


def tasks(which: set[str]) -> list[dict[str, Any]]:
    """(instance, reference) tasks; the instance travels as JSON so workers can rebuild it."""
    out: list[dict[str, Any]] = []
    if "e2" in which:
        for line in (ROOT / "notes/track2/e2_headroom.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if "skipped" in rec or rec["exact"]["status"] != "optimal":
                continue
            out.append({"source": "e2", "rec": rec, "stage": e2_stage(rec["cell"])})
    if "e3" in which:
        for line in (ROOT / "notes/track2/e3_grading.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if "skipped" in rec:
                continue
            out.append({"source": "e3", "rec": rec, "stage": e2_stage(rec["cell"])})
    if "val" in which:
        for name in VAL_SETS:
            bench = BenchmarkSet.read(name)
            stage = "heldout" if name.startswith("heldout") else name.split("_")[1]
            for inst in bench.instances:
                ref = bench.reference[inst.instance_id]
                lb = ref.lower_bound if ref.lower_bound is not None else ref.cost
                out.append(
                    {
                        "source": name,
                        "instance": inst.to_json(),
                        "stage": stage,
                        "cell": inst.metadata.get("cell", ""),
                        "ref_kind": "optimal" if ref.status == "optimal" else "lower_bound",
                        "ref": str(lb),
                        "best_known": str(ref.cost),
                    }
                )
    return out


def run_task(task: dict[str, Any], producers: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"source": task["source"], "stage": task["stage"]}
    if task["source"] in ("e2", "e3"):
        rec = task["rec"]
        out["cell"], out["id"] = rec["cell"], f"{rec['cell']}-{rec['index']}"
        size = "small" if task["source"] == "e2" else "medium"
        inst = _regenerate(rec, size)
        if inst is None:
            out["skipped"] = "regenerated problem does not match the record"
            return out
        if task["source"] == "e2":
            ref = Rational.parse(str(rec["exact"]["cost"]))
            out["ref_kind"] = "optimal"
        else:
            root = fluid_bound_heuristic(WEIGHTS)(PrepState.initial(inst), inst)
            proved = [e["lower_bound"] for e in rec["exact"] if e["lower_bound"] is not None]
            searched = max((Rational.parse(str(x)) for x in proved), default=root)
            ref = max(root, searched)
            out["ref_kind"] = "lower_bound"
    else:
        inst = Instance.from_json(task["instance"])
        out["cell"], out["id"] = task["cell"], inst.instance_id
        ref = Rational.parse(task["ref"])
        out["ref_kind"] = task["ref_kind"]
    out["ref"] = float(ref)
    out["deliveries"] = sum(t.demand for t in inst.targets)
    out["T"], out["M"], out["S"] = (
        len(inst.targets),
        inst.chip.max_mix_volume,
        (inst.chip.storage_capacity),
    )
    for name in producers:
        t0 = time.perf_counter()
        plan = PRODUCERS[name](inst)
        secs = time.perf_counter() - t0
        if plan is None:
            out[name] = {"cost": None, "seconds": round(secs, 3)}
            continue
        _, totals, deltas = replay(inst, plan)
        report = validate(MixingDAG.from_deltas(inst, deltas, totals, name), inst)
        if not report.ok:
            raise RuntimeError(f"{out['id']}: {name} plan fails validation: {report}")
        cost = WEIGHTS.scalar(totals, inst.fluids)
        sample = sum(totals.reagent_by_fluid[f.id] for f in inst.fluids if f.kind.name == "SAMPLE")
        out[name] = {
            "cost": float(cost),
            "ratio": float(cost) / float(ref),
            "sample": sample,
            "mixes": totals.mix_ops,
            "waste": totals.waste_volume,
            "steps": len(plan),
            "seconds": round(secs, 3),
        }
    return out


def _star(args: tuple[dict[str, Any], list[str]]) -> dict[str, Any]:
    return run_task(*args)


def _q(xs: list[float], q: float) -> float:
    return float(np.quantile(np.array(xs), q))


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    producers = sorted({k for r in recs for k, v in r.items() if isinstance(v, dict)})
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if "skipped" in r:
            continue
        groups[(r["source"], r["stage"])].append(r)
        groups[(r["source"] + " by cell", r["cell"])].append(r)
    lines = [f"| set | group | n | ref | {' | '.join(p + ' med / p90 / max' for p in producers)} |"]
    lines.append("|---" * (4 + len(producers)) + "|")
    for (src, grp), rs in sorted(groups.items()):
        cells = []
        for p in producers:
            xs = [r[p]["ratio"] for r in rs if p in r and r[p].get("ratio") is not None]
            missing = sum(1 for r in rs if p in r and r[p].get("ratio") is None)
            txt = (f"{_q(xs, 0.5):.2f} / {_q(xs, 0.9):.2f} / {max(xs):.2f}" if xs else "-") + (
                f" ({missing} none)" if missing else ""
            )
            cells.append(txt)
        lines.append(f"| {src} | {grp} | {len(rs)} | {rs[0]['ref_kind']} | {' | '.join(cells)} |")
    skipped = sum(1 for r in recs if "skipped" in r)
    lines.append(f"\nskipped (regenerated problem differs from the record): {skipped}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    which = set(args.get("which", "e2,e3,val").split(","))
    producers = args.get("producers", "teacher").split(",")
    out = Path(args.get("out", "notes/teacher_study/records.jsonl"))
    workers = int(args.get("workers", 1))
    todo = [(t, producers) for t in tasks(which)]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    print(f"wrote {done} records to {out}")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
