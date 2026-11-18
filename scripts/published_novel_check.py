"""D-94: are the search's novel intermediates (D-76, E-78) made by any published method?

E-78 and E-80 called an intermediate of the search's plan **novel** when none of our own 4
planner runs, the search's root finish, nor any one target's own plan (our sharing pass, plain
and with MRCM's trees) makes its composition. This check asks the same of the 16 published
methods (`bench.PUBLISHED`), so the claim can read "no planner and no published method makes
them".

Per problem with novel intermediates in the given bench records (method-v3, `ours.novel`), every
published method is run twice:
- **whole:** on the problem itself, where the method takes it (most take one target only);
- **per target:** on each target alone, demand 1, on the problem's chip (as E-80 (a)), so every
  single-target method gets its say on every target.
A composition counts as **made** by a method if any `Mix` in its plan produces a droplet of that
exact composition (delivered or not: stricter than E-78, which compared intermediates only).
A novel intermediate **stays novel** if no published method makes it, whole or per target.
Published plans do not depend on the goal, so each problem is run once and every goal's record
of it is read against the same plans.

Reading (D-94, fixed before the run): on the records where the search is cheaper than the best
of the 4 planner runs and holds a novel intermediate (E-78's wins), the claim "no published
method makes them" holds if at least 90% of those novel intermediates stay novel and at least
90% of those problems keep at least one. Otherwise the doc says which methods make them and how
often.

    OMP_NUM_THREADS=1 uv run python scripts/published_novel_check.py \
        records=outputs/bench/b1_v3_all.jsonl workers=30 out=outputs/novel/published.jsonl
    uv run python scripts/published_novel_check.py summary=outputs/novel/published.jsonl \
        records=outputs/bench/b1_v3_all.jsonl

Arguments are key=value: `records` (bench records, comma-separated), `methods` (default all
of `bench.PUBLISHED`), `nfsp_seconds` (per ILP solve, default 600 as in `bench.py`),
`per_target` (1 default, 0 to skip), `limit` (problems), `workers`, `out`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from meda.core.actions import Action, Mix
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, apply
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import nfsp as nfsp_module
from meda.solvers.baselines.paper_graph import NotApplicable, emit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench  # noqa: E402
import novel_check as nc  # noqa: E402
import search_check as sc  # noqa: E402

READ_SHARE = 0.90  # D-94, fixed before the run


def mixed(inst: Instance, plan: tuple[Action, ...]) -> set[str]:
    """The composition (as `novel_check.show`) of every droplet a `Mix` in `plan` makes."""
    state = PrepState.initial(inst)
    out: set[str] = set()
    for act in plan:
        before = {d.uid for d in state.droplets}
        state, _, _ = apply(state, act, inst)
        if isinstance(act, Mix):
            out |= {nc.show(d.composition) for d in state.droplets if d.uid not in before}
    return out


def run_method(inst: Instance, name: str) -> dict[str, Any]:
    """One published method on `inst`: the compositions its plan mixes, or why it has none."""
    t0 = sc.CLOCK()
    try:
        plan = emit(inst, bench.PUBLISHED[name](inst))
    except NotApplicable as e:
        return {"na": str(e)[:200], "seconds": round(sc.CLOCK() - t0, 3)}
    except (IllegalActionError, ValueError, AssertionError) as e:  # recorded, never hidden
        return {"error": f"{type(e).__name__}: {e}"[:200], "seconds": round(sc.CLOCK() - t0, 3)}
    return {"comps": sorted(mixed(inst, plan)), "seconds": round(sc.CLOCK() - t0, 3)}


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    nfsp_module.TIME_LIMIT = opts["nfsp_seconds"]
    inst = Instance.from_json(task["instance"])
    out: dict[str, Any] = {k: v for k, v in task.items() if k != "instance"}
    out["whole"] = {m: run_method(inst, m) for m in opts["methods"]}
    if opts["per_target"]:
        out["per_target"] = {}
        for t in inst.targets:
            one = replace(inst, targets=(replace(t, id=0, demand=1),))
            out["per_target"][str(t.id)] = {m: run_method(one, m) for m in opts["methods"]}
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def load(paths: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for p in paths for line in Path(p).read_text().splitlines()]


def tasks(recs: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """One task per problem holding a novel intermediate under any goal."""
    want = sorted({(r["set"], r["id"]) for r in recs if r.get("ours", {}).get("novel")})
    sets: dict[str, dict[str, Instance]] = {}
    out = []
    for s, i in want[:limit]:
        if s not in sets:
            sets[s] = {x.instance_id: x for x in BenchmarkSet.read(s).instances}
        out.append({"set": s, "id": i, "instance": sets[s][i].to_json()})
    return out


def makers(res: dict[str, Any]) -> dict[str, set[str]]:
    """Composition -> the published methods that mix it (`m` whole, `m/t` on one target)."""
    by: dict[str, set[str]] = defaultdict(set)
    for m, st in res["whole"].items():
        for c in st.get("comps", []):
            by[c].add(m)
    for st_t in res.get("per_target", {}).values():
        for m, st in st_t.items():
            for c in st.get("comps", []):
                by[c].add(f"{m}/t")
    return by


def summarise(results: list[dict[str, Any]], recs: list[dict[str, Any]]) -> list[str]:
    res = {(r["set"], r["id"]): r for r in results}
    ran = sorted({m for r in results for m in r["whole"]})
    lines = ["Coverage: problems each method took (whole / every target alone)", ""]
    lines += ["| method | whole | per target | errors |", "|---|---|---|---|"]
    for m in ran:
        whole = sum("comps" in r["whole"][m] for r in results)
        per = sum(
            bool(r.get("per_target")) and all("comps" in st[m] for st in r["per_target"].values())
            for r in results
        )
        err = sum("error" in r["whole"][m] for r in results) + sum(
            "error" in st[m] for r in results for st in r.get("per_target", {}).values()
        )
        lines.append(f"| {m} | {whole}/{len(results)} | {per}/{len(results)} | {err} |")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        ours, best = r.get("ours", {}), r.get("best_planner")
        cost = (ours.get("by_goal") or {}).get(r["goal"])
        if not ours.get("novel") or cost is None or best is None or (r["set"], r["id"]) not in res:
            continue
        if cost < best - 1e-9:
            groups[(r["set"], r["goal"])].append(r)
            groups[("all", r["goal"])].append(r)
            groups[("all", "all")].append(r)
    lines += [
        "",
        "Wins over the best planner run with a novel intermediate: how many stay novel against "
        "every published method (whole and per target)",
        "",
        "| set | goal | wins | novel intermediates | stay novel | wins keeping one |",
        "|---|---|---|---|---|---|",
    ]
    hits: dict[str, int] = defaultdict(int)
    verdict = "no wins with novel intermediates in these records"
    for key in sorted(groups, key=lambda k: (k[0] == "all", k)):
        rs = groups[key]
        total = stay = keep = 0
        for r in rs:
            by = makers(res[(r["set"], r["id"])])
            left = [m for m in r["ours"]["novel"] if not by.get(m["comp"])]
            total += len(r["ours"]["novel"])
            stay += len(left)
            keep += bool(left)
            if key == ("all", "all"):
                for m in r["ours"]["novel"]:
                    for who in by.get(m["comp"], ()):
                        hits[who] += 1
        lines.append(
            f"| {key[0]} | {key[1]} | {len(rs)} | {total} | {stay} ({stay / total:.0%}) | "
            f"{keep} ({keep / len(rs):.0%}) |"
        )
        if key == ("all", "all"):
            holds = stay / total >= READ_SHARE and keep / len(rs) >= READ_SHARE
            verdict = (
                f"Reading (D-94): {stay} of {total} novel intermediates ({stay / total:.0%}) stay "
                f"novel; {keep} of {len(rs)} wins ({keep / len(rs):.0%}) keep one: "
                + (
                    "no published method makes them"
                    if holds
                    else "published methods make a share of them (see below)"
                )
            )
    lines += ["", verdict]
    if hits:
        lines += ["", "Novel intermediates (all wins) a published method makes, by method:", ""]
        lines += [f"- {who}: {n}" for who, n in sorted(hits.items(), key=lambda x: -x[1])]
    return lines


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    recs = load(args["records"].split(","))
    if "summary" in args:
        print("\n".join(summarise(load([args["summary"]]), recs)))
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    opts = {
        "methods": args.get("methods", ",".join(bench.PUBLISHED)).split(","),
        "nfsp_seconds": float(args.get("nfsp_seconds", 600)),
        "per_target": args.get("per_target", "1") == "1",
    }
    unknown = set(opts["methods"]) - set(bench.PUBLISHED)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}")
    todo = [
        ({**t, "commit": commit}, opts)
        for t in tasks(recs, int(args["limit"]) if "limit" in args else None)
    ]
    out = Path(args.get("out", "outputs/novel/published.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems, {workers} workers, commit {commit}", flush=True)
    t0 = time.perf_counter()
    results = []
    with out.open("w") as fh:
        stream = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for done, rec in enumerate(stream, 1):
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{done}/{len(todo)} {rec['id']}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {len(results)} records to {out}\n")
    print("\n".join(summarise(results, recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
