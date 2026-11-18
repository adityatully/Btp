"""Justification sweeps of method-v3's search (D-98, `notes/justification_audit.md` Part 2).

Method-v3 (SPEC 0.44) on a fixed dev subset, one setting changed at a time; the test sets stay
untouched. This script only imports `search_check` and `bench`: every variant is a subclass of
`search_check.Search` or a module setting, so the method's own code is unchanged.

**Subset** (`subset=build`, written once to `notes/ablation/subset_v1.json`): 110 problems from
the dev sets, drawn with seed `SUBSET_SEED` and spread evenly over each set's cells: val s1 10,
s2 10, s3 20, s4 20, s5 20 (`_v2`), `val_emrcm_v1` 30 (over its 8 cells: N = 3 8, N = 4 12,
N = 5 10). Fluid goal (`costs/geometric`), as the finals.

**Runs** (`runs=`, comma-separated groups or configs; each is (config, repeat)):
- `J1`, the noise floor: method-v3 three times (`v3` repeats 0-2) and with random tie-breaking
  under two seeds (`seed1`, `seed2`).
- `J2`, simulations x exploration: sims {16, 64, 256, 1024} x C_PUCT {0.5, 1.5, 4}; `v3` repeat 0
  is the (256, 1.5) cell.
- `J3`, search mechanics one at a time: best-leaf backup, first-play value {0, -1}, value
  scale from the first plan, no floor, dead-end value {-2, -0.5}, play the move with the
  cheapest plan, never pick a Discard. Join with J2's chosen point by '+'
  (`backup_max+sims64_c0.5`).
- `J5`, the finisher switch, only on problems where it fires (val s5, `val_emrcm_v1` N 4-5):
  switch mean {5, 10, 40} s, leaves before switching {1, 10}, MRCM's time share {0.25, 0.75}.

Each record: config, repeat, problem, final cost, trace (CPU s, cost), leaves, moves the tree
committed (`depth`, E-81) and plan length, capped, finisher mode and switch, seconds; and,
measured without changing anything, leaf finishes longer than the step limit (S14), leaves
that repeat an earlier leaf's state (S21), and how often the split tree's and MRCM's search
budgets ran out (P15, P16). Resumes: a (config, repeat, problem) already in `out` is skipped.

**Reading, fixed in advance (D-98):** per config, the paired cost ratio to `v3` repeat 0 per
problem, its mean and 95% bootstrap CI, losses and wins. The **noise band** is the largest
|mean ratio - 1| among J1's other runs (v3 repeats 1-2, seed1, seed2) against v3 repeat 0, per
stage group (s1-s2, s3, s4-s5, emrcm). A config is **level** when its CI contains 1 or its mean
lies inside the band; **better / worse** when the CI excludes 1 and the mean is outside it.
Method-v3's value is justified when its neighbours on each side are level or worse.

    OMP_NUM_THREADS=1 uv run python scripts/ablation.py subset=build
    OMP_NUM_THREADS=1 uv run python scripts/ablation.py runs=J1 workers=16 \\
        out=outputs/ablation/j1.jsonl
    uv run python scripts/ablation.py summary=outputs/ablation/j1.jsonl
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import justify_planner as jp  # noqa: E402  (counts the tree's and MRCM's budget runs)
import search_check as sc  # noqa: E402
from meda.core.actions import Discard  # noqa: E402
from meda.generators.benchmark_io import BenchmarkSet  # noqa: E402

SUBSET = ROOT / "notes/ablation/subset_v1.json"
SUBSET_SEED = 20261118
# (set, count): the dev sets the subset is drawn from, spread evenly over each set's cells
SUBSET_SETS = (
    ("val_s1_v2", 10),
    ("val_s2_v2", 10),
    ("val_s3_v2", 20),
    ("val_s4_v2", 20),
    ("val_s5_v2", 20),
    ("val_emrcm_v1", 30),
)
SECONDS, SIMS = 900.0, 256  # method-v3 (bench.METHOD)
SIMS_GRID, CPUCT_GRID = (16, 64, 256, 1024), (0.5, 1.5, 4.0)


J3 = {  # search mechanics, one at a time (S6-S10, S12, S20)
    "backup_max": {"backup": "max"},  # a move's value = its best leaf, not the mean
    "fpu0": {"fpu": 0.0},  # an unvisited move's value (method-v3: the parent's)
    "fpu_m1": {"fpu": -1.0},
    "scale_first": {"scale": "first"},  # c0 = the first plan's cost, not the root's finish
    "nofloor": {"floor": None},  # no floor at -1 on a leaf's value
    "dead_m2": {"dead": -2.0},  # a dead end's value (method-v3: -1)
    "dead_m05": {"dead": -0.5},
    "move_cheapest": {"move": "cheapest"},  # play the move with the cheapest plan through it
    "no_discard": {"discard": False},  # the search never picks a Discard
}
J5 = {  # the finisher switch (F3-F5), only where it fires
    "switch5": {"switch_mean": 5.0, "only": "switch"},
    "switch10": {"switch_mean": 10.0, "only": "switch"},
    "switch40": {"switch_mean": 40.0, "only": "switch"},
    "leaves1": {"switch_min_leaves": 1, "only": "switch"},
    "leaves10": {"switch_min_leaves": 10, "only": "switch"},
    "tier25": {"tier_share": 0.25, "only": "switch"},
    "tier75": {"tier_share": 0.75, "only": "switch"},
}


def configs() -> dict[str, dict[str, Any]]:
    """Every named config: changes against method-v3 (none = method-v3 itself). A run may join
    configs with '+' (e.g. `backup_max+sims64_c0.5`, J3 at J2's chosen point)."""
    out: dict[str, dict[str, Any]] = {"v3": {}, "seed1": {"seed": 1}, "seed2": {"seed": 2}}
    for s in SIMS_GRID:
        for c in CPUCT_GRID:
            if (s, c) != (SIMS, sc.C_PUCT):
                out[f"sims{s}_c{c:g}"] = {"sims": s, "cpuct": c}
    return out | J3 | J5


def config(name: str) -> dict[str, Any]:
    known = configs()
    merged: dict[str, Any] = {}
    for part in name.split("+"):
        if part not in known:
            raise ValueError(f"unknown config '{part}' (groups {sorted(GROUPS)}, {sorted(known)})")
        merged |= known[part]
    return merged


GROUPS: dict[str, list[tuple[str, int]]] = {
    "J1": [("v3", 0), ("v3", 1), ("v3", 2), ("seed1", 0), ("seed2", 0)],
    "J2": [("v3", 0)] + [(name, 0) for name in configs() if name.startswith("sims")],
    "J3": [("v3", 0)] + [(name, 0) for name in J3],
    "J5": [("v3", 0)] + [(name, 0) for name in J5],
}


def switch_problem(entry: dict[str, Any]) -> bool:
    """Where the finisher switch fires (E-69, E-73): 4-5 fluid mixtures and stage 5."""
    return entry["set"] == "val_s5_v2" or (
        entry["set"] == "val_emrcm_v1" and entry["cell"].startswith(("N4", "N5"))
    )


# --------------------------------------------------------------------------- subset


def build_subset() -> list[dict[str, Any]]:
    """The fixed subset: per set, `count` problems spread evenly over its cells (a remainder
    goes to the first cells in sorted order), drawn with `SUBSET_SEED`."""
    rng = np.random.default_rng(SUBSET_SEED)
    picked: list[dict[str, Any]] = []
    for name, count in SUBSET_SETS:
        bench = BenchmarkSet.read(name)
        cells: dict[str, list[str]] = defaultdict(list)
        for inst in bench.instances:
            cells[str(inst.metadata.get("cell", "-"))].append(inst.instance_id)
        names = sorted(cells)
        share = [count // len(names) + (i < count % len(names)) for i in range(len(names))]
        for cell, n in zip(names, share, strict=True):
            ids = sorted(cells[cell])
            if n > len(ids):
                raise ValueError(f"{name}: cell {cell} has {len(ids)} problems, {n} wanted")
            for i in sorted(rng.choice(len(ids), size=n, replace=False)):
                picked.append({"set": name, "cell": cell, "id": ids[int(i)]})
    return picked


def stage(entry: dict[str, Any]) -> str:
    """The stage group a problem is read in."""
    s = entry["set"]
    if s.startswith("val_emrcm"):
        return "emrcm"
    if s.startswith(("val_s1", "val_s2")):
        return "s1-s2"
    return "s3" if s.startswith("val_s3") else "s4-s5"


# --------------------------------------------------------------------------- search variants

_SETTINGS: dict[str, Any] = {}  # this worker's config (set before each task)
_LAST: list[Any] = []  # the last search built in this worker


class AblationSearch(sc.Search):
    """`search_check.Search` with the config's settings. With no setting every method is
    `search_check`'s own, so method-v3 runs unchanged; each J3 variant changes one line of a
    copied method. Keeps a handle on itself so the record can read the moves it committed, and
    counts (always, without changing anything): leaf finishes longer than the step limit
    (S14) and leaves that repeat an earlier leaf's state (S21)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._c0 = 1.0
        super().__init__(*args, **kwargs)
        if _SETTINGS.get("seed") is not None:
            self.rng = np.random.default_rng(int(_SETTINGS["seed"]))
        self.over_limit = 0
        self.finishes = 0
        self.seen: set[Any] = set()
        self.repeats = 0
        inner = self.finish

        def counted(state: Any) -> Any:
            found = inner(state)
            self.finishes += 1
            if found is not None and state.step + len(found[0]) > self.inst.step_limit:
                self.over_limit += 1
            return found

        self.finish = counted
        _LAST[:] = [self]

    # S6: the value's scale
    @property
    def c0(self) -> float:
        if _SETTINGS.get("scale") == "first" and self.trace:
            return max(self.trace[0][1], 1.0)  # the plain pass's plan, offered before the root
        return self._c0

    @c0.setter
    def c0(self, value: float) -> None:
        self._c0 = value

    def score(self, cost: float) -> float:
        if "floor" in _SETTINGS and _SETTINGS["floor"] is None:  # S7
            return (self.c0 - cost) / self.c0
        return super().score(cost)

    def leaf(self, obs: dict[str, Any], actions: tuple[Any, ...], info: dict | None) -> Any:
        node = super().leaf(obs, actions, info)
        state = self.env.state
        if state in self.seen:
            self.repeats += 1
        else:
            self.seen.add(state)
        if "dead" in _SETTINGS and node.value == -1.0 and not np.isfinite(node.best):  # S8
            node.value = float(_SETTINGS["dead"])
        if _SETTINGS.get("discard") is False and not node.terminal:  # S20
            node.discard = np.array([isinstance(a, Discard) for a in self.env.legal], dtype=bool)
        return node

    def simulate(self, root: Any) -> None:
        """`search_check.Search.simulate` (no planner jumps, as in method-v3), with S9's
        backup, S10's first-play value and S20's Discard mask as options."""
        if not ({"backup", "fpu", "discard"} & set(_SETTINGS)):
            return super().simulate(root)
        best_backup = _SETTINGS.get("backup") == "max"
        path: list[tuple[Any, int]] = []
        node = root
        while True:
            if node.jumps is None:
                self.add_jumps(node)
            n = node.visits.sum()
            if best_backup:
                if not hasattr(node, "top"):
                    node.top = np.full(len(node.prior), -np.inf)
                seen = node.top
            else:
                seen = node.total / np.maximum(node.visits, 1)
            unvisited = float(_SETTINGS.get("fpu", node.value))
            q = np.where(node.visits > 0, seen, unvisited)
            u = sc.C_PUCT * node.prior * math.sqrt(n + 1) / (1 + node.visits)
            score = q + u
            mask = getattr(node, "discard", None)
            if mask is not None and len(mask) == len(score) and not mask.all():
                score = np.where(mask, -np.inf, score)
            if self.rng is not None:
                score = score + 1e-9 * self.rng.random(len(score))
            i = int(np.argmax(score))
            path.append((node, i))
            if i not in node.children:
                node.children[i] = self.child(node, i)
                end = node.children[i]
                break
            node = node.children[i]
            if node.terminal:
                end = node
                break
        for nd, i in path:
            nd.visits[i] += 1
            nd.total[i] += end.value
            if best_backup:
                if not hasattr(nd, "top"):
                    nd.top = np.full(len(nd.prior), -np.inf)
                nd.top[i] = max(nd.top[i], end.value)
            nd.best = min(nd.best, end.best)

    def _descend(self, root: Any, sims: int) -> dict[str, Any]:
        """`search_check.Search._descend` (no jumps, no sample collection, as in method-v3),
        with S12's move choice: the move with the cheapest whole plan seen through it."""
        if _SETTINGS.get("move") != "cheapest":
            return super()._descend(root, sims)
        capped = False
        while not root.terminal:
            for _ in range(sims):
                if sc.CLOCK() > self.deadline:
                    capped = True
                    break
                self.simulate(root)
            if capped or root.visits.sum() == 0:
                break
            visited = [j for j in root.children if root.visits[j] > 0]
            i = min(visited, key=lambda j: (root.children[j].best, -root.visits[j], j))
            root = root.children[i]
        self.path = root.actions
        out = {"capped": capped, "played_to_end": root.terminal}
        if self.best is None:
            return {"plan": None, "tail": 0, **out}
        return {"plan": self.best[1], "tail": self.best[2], **out}


sc.Search = AblationSearch  # type: ignore[misc]  # run_task builds the search by this name


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    """One (config, repeat, problem): method-v3 with the config's changes."""
    cfg = config(task["config"])
    _SETTINGS.clear()
    _SETTINGS.update(cfg)
    sc.C_PUCT = float(cfg.get("cpuct", 1.5))
    inst = BenchmarkSet.read(task["set"]).by_id(task["id"])
    search_task = {
        "set": "J",
        "id": task["id"],
        "cell": task["cell"],
        "instance": inst.to_json(),
        "optimum": None,
        "planner": None,  # the plain pass from the empty chip, off the clock (search_check)
        "mrcm": None,
        "greedy": None,
    }
    opts = {
        "agent": None,
        "seconds": float(task.get("seconds", SECONDS)),
        "hints": "planner",
        "reward_scale": 0.01,
        "tier_share": float(cfg.get("tier_share", sc.TIER_SHARE)),
        "switch_mean": float(cfg.get("switch_mean", sc.SWITCH_MEAN)),
        "switch_min_leaves": int(cfg.get("switch_min_leaves", sc.SWITCH_MIN_LEAVES)),
        "first_plan": True,
    }
    sims = int(cfg.get("sims", SIMS))
    jp.COUNTS.update(tree_budget=0, mrcm_budget=0)
    rec = sc.run_task(search_task, "uniform", sims, opts, "switch")
    search = _LAST[0]
    return {
        "config": task["config"],
        "repeat": task["repeat"],
        "set": task["set"],
        "cell": task["cell"],
        "id": task["id"],
        "commit": task["commit"],
        "settings": cfg,
        "sims": sims,
        "cpuct": sc.C_PUCT,
        "cost": rec["search_cost"],
        "trace": rec["trace"],
        "leaves": rec["leaves"],
        "depth": len(search.path),
        "plan_length": rec.get("search_steps"),
        "capped": rec["capped"],
        "played_to_end": rec["played_to_end"],
        "finish_mode": rec["finish_mode"],
        "switched_at": rec["switched_at"],
        "seconds": rec.get("search_seconds"),
        "plain_pass": rec["planner"],
        "finishes": search.finishes,
        "over_limit": search.over_limit,  # S14: leaf finishes longer than the step limit
        "repeats": search.repeats,  # S21: leaves whose state an earlier leaf had
        "tree_budget": jp.COUNTS["tree_budget"],  # P15
        "mrcm_budget": jp.COUNTS["mrcm_budget"],  # P16
    }


# --------------------------------------------------------------------------- summary


def boot_ci(xs: list[float], seed: int = 0, n: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(xs)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cost_at(rec: dict[str, Any], t: float) -> float | None:
    """The cheapest whole plan in the trace by CPU second t."""
    found = [c for s, c in rec["trace"] if s <= t]
    return min(found) if found else None


def compare(base: dict[str, dict], other: dict[str, dict]) -> dict[str, Any] | None:
    """Paired ratios other / base over the problems both finished."""
    ids = [i for i in base if i in other and base[i]["cost"] and other[i]["cost"] is not None]
    if not ids:
        return None
    r = [other[i]["cost"] / base[i]["cost"] for i in ids]
    lo, hi = boot_ci(r)
    return {
        "n": len(ids),
        "mean": float(np.mean(r)),
        "lo": lo,
        "hi": hi,
        "worse": sum(x > 1 + 1e-9 for x in r),
        "better": sum(x < 1 - 1e-9 for x in r),
    }


def noise_band(by: dict[tuple[str, int], dict[str, dict]], group: str) -> float | None:
    """The largest |mean ratio - 1| of J1's other runs against v3 repeat 0, in `group`."""
    base = by.get(("v3", 0))
    if base is None:
        return None
    gaps = []
    for key in GROUPS["J1"][1:]:
        if key in by:
            c = compare(
                {i: r for i, r in base.items() if r["group"] == group},
                {i: r for i, r in by[key].items() if r["group"] == group},
            )
            if c:
                gaps.append(abs(c["mean"] - 1))
    return max(gaps) if gaps else None


def verdict(c: dict[str, Any], band: float | None) -> str:
    inside = band is not None and abs(c["mean"] - 1) <= band
    if (c["lo"] <= 1 <= c["hi"]) or inside:
        return "level"
    return "worse" if c["mean"] > 1 else "better"


def summarise(paths: list[Path]) -> str:
    entries = {e["id"]: e for e in json.loads(SUBSET.read_text())}
    by: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r["group"] = stage(entries[r["id"]])
            by[(r["config"], r["repeat"])][r["id"]] = r
    groups = ("s1-s2", "s3", "s4-s5", "emrcm", "all")
    bands = {g: noise_band(by, g) for g in groups[:-1]}
    lines = [
        "| config | rep | group | n | mean ratio to v3 [95% CI] | worse / better | verdict |"
        " depth (median) | leaves (median) | capped |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = by.get(("v3", 0), {})
    for key in sorted(by):
        for g in groups:
            rs = {i: r for i, r in by[key].items() if g == "all" or r["group"] == g}
            if not rs:
                continue
            mine = {i: r for i, r in base.items() if i in rs}
            c = compare(mine, rs) if key != ("v3", 0) else None
            known = [b for b in bands.values() if b is not None]
            band = bands.get(g) if g != "all" else max(known, default=None)
            ratio = f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c else "reference"
            wb = f"{c['worse']} / {c['better']}" if c else "-"
            v = verdict(c, band) if c else "-"
            depth = int(np.median([r["depth"] for r in rs.values()]))
            leaves = int(np.median([r["leaves"] for r in rs.values()]))
            capped = sum(bool(r["capped"]) for r in rs.values())
            lines.append(
                f"| {key[0]} | {key[1]} | {g} | {len(rs)} | {ratio} | {wb} | {v} | {depth} |"
                f" {leaves} | {capped} |"
            )
    lines += [
        "",
        "Noise band (D-98): "
        + ", ".join(f"{g} {b:.4f}" if b is not None else f"{g} -" for g, b in bands.items()),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def expand(runs: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    names = configs()
    for item in runs.split(","):
        if item in GROUPS:
            out += GROUPS[item]
        elif all(part in names for part in item.split("+")):
            out.append((item, 0))
        else:
            raise ValueError(
                f"unknown run '{item}' (groups {sorted(GROUPS)}, configs {sorted(names)})"
            )
    return list(dict.fromkeys(out))


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if args.get("subset") == "build":
        if SUBSET.exists():
            raise FileExistsError(f"{SUBSET} is frozen; delete it by hand to rebuild")
        SUBSET.parent.mkdir(parents=True, exist_ok=True)
        picked = build_subset()
        SUBSET.write_text(json.dumps(picked, indent=1) + "\n")
        print(f"{len(picked)} problems -> {SUBSET}")
        return
    if "summary" in args:
        print(summarise([Path(p) for p in args["summary"].split(",")]))
        return
    entries = json.loads(SUBSET.read_text())
    if "limit" in args:  # light tests: the first n problems of each set
        per: dict[str, int] = defaultdict(int)
        kept = []
        for e in entries:
            if per[e["set"]] < int(args["limit"]):
                kept.append(e)
                per[e["set"]] += 1
        entries = kept
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["repeat"], r["id"]))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    todo = [
        {
            **e,
            "config": c,
            "repeat": rep,
            "commit": commit,
            **({"seconds": float(args["seconds"])} if "seconds" in args else {}),
        }
        for c, rep in expand(args["runs"])
        for e in entries
        if (c, rep, e["id"]) not in done
        and (config(c).get("only") != "switch" or switch_problem(e))
    ]
    # the slow problems first, so the last workers are not left with an s5 at the end
    order = {"s4-s5": 0, "emrcm": 1, "s3": 2, "s1-s2": 3}
    todo.sort(key=lambda t: order[stage(t)])
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} runs, {workers} workers, commit {commit}", flush=True)
    with out.open("a") as fh:
        pool = Pool(workers) if workers > 1 else None
        results = pool.imap_unordered(run_one, todo) if pool else map(run_one, todo)
        for i, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {rec['config']}#{rec['repeat']} {rec['id']} cost {rec['cost']}"
                f" depth {rec['depth']} leaves {rec['leaves']} {rec['seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
