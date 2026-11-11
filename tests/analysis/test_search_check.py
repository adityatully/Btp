"""`scripts/search_check.py` (D-37): the search's answer is a valid whole plan, never dearer
than the planner's finish from the start, never below a proved optimum; the reading follows
the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from meda.core.state import PrepState
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def search_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("search_check", ROOT / "scripts/search_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["search_check"] = module  # its dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


SC = search_module()
OPTS = {"agent": None, "seconds": 60.0, "hints": "planner", "reward_scale": 0.01}


@pytest.mark.parametrize("instance", TINY.instances[:4], ids=lambda i: i.instance_id)
def test_flat_prior_search_returns_a_valid_plan_no_dearer_than_the_planner(instance: Any) -> None:
    start = finish_from(PrepState.initial(instance), instance, SC.WEIGHTS)
    assert start is not None
    task = {
        "set": "A",
        "id": instance.instance_id,
        "cell": "",
        "instance": instance.to_json(),
        "optimum": instance.metadata.get("optimum"),
        "planner": float(start[1]),
        "mrcm": None,
        "greedy": None,
    }
    rec = SC.run_task(task, "uniform", 8, OPTS)  # replays and validates the plan
    assert rec["search_cost"] is not None
    assert rec["search_cost"] <= float(start[1]) + 1e-9
    assert not rec["capped"]


def stats(n: int, below: int, ratio_opt: float = 1.1) -> dict[str, Any]:
    return {"n": n, "below": below, "ratio_opt": ratio_opt}


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    assert SC.verdict(stats(9, 3), stats(9, 2), stats(9, 3)).endswith("without the planner")
    found = SC.verdict(stats(9, 3), stats(9, 3, 1.05), stats(9, 2))
    assert found.startswith("(1) yes, (2) no, (3) no") and found.endswith("not from learning")
    assert SC.verdict(stats(9, 2), stats(9, 2), stats(9, 0)).endswith("at this budget")
    # ties on "below" go to the lower mean ratio to the optimum
    assert "(2) yes" in SC.verdict(stats(9, 3, 1.02), stats(9, 3, 1.05), None)
    # without optima (set C) ties go to the lower mean ratio to the planner
    no_opt = {"ratio_opt": None}
    net = stats(9, 3) | no_opt | {"ratio_plan": 0.95}
    assert "(2) yes" in SC.verdict(net, stats(9, 3) | no_opt | {"ratio_plan": 0.97}, None)


def test_set_c_is_full_size_stages_3_to_5_where_the_planner_is_not_known_optimal() -> None:
    found = SC.tasks({"C"}, None)
    assert {t["set"] for t in found} == {"C3", "C4", "C5"}
    assert len(found) == len({t["id"] for t in found}) == 99  # 122 less 23 at the bound
    assert all(t["optimum"] is None and t["planner"] > 0 for t in found)


@pytest.mark.parametrize("instance", TINY.instances[:3], ids=lambda i: i.instance_id)
def test_recording_samples_does_not_change_the_search(instance: Any) -> None:
    import time

    import numpy as np

    from meda.analysis.evaluate import build_env

    cfg = {"costs": SC._load("costs/geometric.yaml"), "env": SC._load("env/sample_prep.yaml")}
    found = []
    for collect in (None, 2):
        search = SC.Search(
            build_env(cfg, [instance]),
            instance,
            None,
            time.perf_counter() + 60,
            rng=np.random.default_rng(0),
            collect=collect,
        )
        found.append(search.run(8)["plan"])
    assert found[0] == found[1]
    root = [s for s in search.samples(4, np.random.default_rng(0)) if s[0] == ()]
    assert len(root) == 1 and root[0][1] is not None


@pytest.mark.parametrize("instance", TINY.instances[:3], ids=lambda i: i.instance_id)
def test_trace_ends_at_the_answer_and_only_improves(instance: Any) -> None:
    start = finish_from(PrepState.initial(instance), instance, SC.WEIGHTS)
    assert start is not None
    task = {
        "set": "C3",
        "id": instance.instance_id,
        "cell": "",
        "instance": instance.to_json(),
        "optimum": None,
        "planner": float(start[1]),
        "mrcm": None,
        "greedy": None,
    }
    rec = SC.run_task(task, "uniform", 8, OPTS)
    costs = [c for _, c in rec["trace"]]
    times = [s for s, _ in rec["trace"]]
    assert costs and costs[-1] == pytest.approx(rec["search_cost"])
    assert costs == sorted(costs, reverse=True) and times == sorted(times)
    assert SC.cost_at(rec, 1e9) == pytest.approx(min(rec["search_cost"], task["planner"]))
    assert SC.cost_at(rec, -1) == task["planner"]


def test_race_reading_follows_the_rule_fixed_in_advance() -> None:
    def rec(i: int, trace: list[tuple[float, float]]) -> dict[str, Any]:
        return {"id": str(i), "planner": 100.0, "trace": trace}

    fast = [rec(i, [(5.0, 90.0)]) for i in range(10)]
    slow = [rec(i, [(100.0, 90.0)]) for i in range(10)]
    assert SC.race_verdict(fast, slow).endswith("the network makes search faster")
    assert SC.race_verdict(slow, fast).endswith("no speed gain at these budgets")
    near = [rec(i, [(5.0, 99.0)]) for i in range(10)]  # cheaper, but by less than 0.02
    assert SC.race_verdict(near, slow).endswith("no speed gain at these budgets")


def test_search_with_mrcm_finishes_is_never_dearer_than_mrcm() -> None:
    """D-55: the root's first leaf is MRCM's own plan, so the answer is at most MRCM's cost."""
    task = next(t for t in SC.tasks({"B"}, None) if t["id"] == "val_s2-036")
    rec = SC.run_task(task, "uniform", 8, OPTS, "mrcm")
    assert rec["finisher"] == "mrcm"
    assert rec["search_cost"] is not None and rec["search_cost"] <= task["mrcm"] + 1e-9


def test_finisher_reading_follows_the_rule_fixed_in_advance() -> None:
    assert SC.finisher_verdict({"n": 13, "below_mrcm": 5}).endswith("improves MRCM")
    assert SC.finisher_verdict({"n": 13, "below_mrcm": 4}).endswith("a weak gain")
    assert SC.finisher_verdict({"n": 13, "below_mrcm": 0}).endswith("its own finishes")


def test_set_d_is_full_size_where_a_published_method_beats_the_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-56: D keeps the s3-s5 problems where e-MRCM or NFSP is cheaper than the best planner;
    the cheaper of the two is the reference."""
    c = {t["id"]: t for t in SC.tasks({"C"}, None)}
    a, b, d = sorted(c)[:3]
    lines = [
        {
            "source": c[a]["id"].split("-")[0] + "_v2",
            "id": a,
            "emrcm": {"cost": c[a]["planner"] - 5},
            "nfsp": {"cost": c[a]["planner"] - 9},
        },
        {
            "source": c[b]["id"].split("-")[0] + "_v2",
            "id": b,
            "emrcm": {"cost": c[b]["planner"] + 5},
            "nfsp": {"cost": None, "na": "over"},
        },
        {
            "source": c[d]["id"].split("-")[0] + "_v2",
            "id": d,
            "emrcm": {"cost": None, "na": "x"},
            "nfsp": {"cost": c[d]["planner"] - 1},
        },
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    monkeypatch.setattr(SC, "LIT_FULL", path)
    found = {t["id"]: t for t in SC.tasks({"D"}, None)}
    assert set(found) == {a, d}
    assert all(t["set"] == "D" for t in found.values())
    assert found[a]["mrcm"] == c[a]["planner"] - 9
    assert found[a]["planner"] == c[a]["planner"]


def test_race_on_d_is_reported_only_below_ten_problems() -> None:
    def rec(i: int, trace: list[tuple[float, float]], kind: str) -> dict[str, Any]:
        return {"id": str(i), "planner": 100.0, "trace": trace}

    for n, few in ((9, True), (10, False)):
        groups = {
            ("D", "net", 256): [rec(i, [(5.0, 90.0)], "net") for i in range(n)],
            ("D", "uniform", 256): [rec(i, [(100.0, 90.0)], "uniform") for i in range(n)],
        }
        line = next(x for x in SC.race(groups) if x.startswith("Race reading on D"))
        assert ("reported only" in line) == few
        assert "D-56" in line and line.endswith("the network makes search faster")


def test_set_t_is_the_forty_target_probes_with_the_planner_computed_in_the_run() -> None:
    """D-59: T has no planner records; the reference is the sharing pass, computed before the
    search's clock starts."""
    found = SC.tasks({"T"}, None)
    assert [t["id"] for t in found] == [f"t40-{i:02d}" for i in range(10)]
    assert all(t["planner"] is None and t["set"] == "T" for t in found)
    instance = TINY.instances[0]
    start = finish_from(PrepState.initial(instance), instance, SC.WEIGHTS)
    assert start is not None
    task = {"set": "T", "id": instance.instance_id, "cell": "", "instance": instance.to_json()}
    task |= {"optimum": None, "planner": None, "mrcm": None, "greedy": None}
    rec = SC.run_task(task, "uniform", 4, OPTS)
    assert rec["planner"] == float(start[1]) and rec["planner_seconds"] >= 0
    assert rec["search_cost"] <= rec["planner"] + 1e-9


def test_race_on_t_counts_net_or_zero_at_any_budget() -> None:
    def rec(i: int, trace: list[tuple[float, float]]) -> dict[str, Any]:
        return {"id": str(i), "planner": 100.0, "trace": trace}

    slow = [rec(i, [(100.0, 90.0)]) for i in range(10)]
    fast = [rec(i, [(5.0, 90.0)]) for i in range(10)]
    groups = {("T", "uniform", 256): slow, ("T", "net", 256): slow, ("T", "zero", 256): slow}
    assert SC.race(groups)[-1].endswith("D-59): no speed gain")
    groups[("T", "zero", 1)] = fast
    lines = SC.race(groups)
    assert lines[-1].endswith("the network makes search faster at 40 targets")
    assert any(x.startswith("Race on T, zero at 1 sims") for x in lines)


def test_set_f_is_each_methods_best_thirteen_with_its_cost_as_reference() -> None:
    """D-60: per method, the 13 problems it takes where it does best against our planner."""
    found = SC.tasks({"F"}, None)
    by = {m: [t for t in found if t["set"] == f"F-{m}"] for m in SC.F_METHODS}
    assert all(len(ts) == SC.F_N for ts in by.values())
    for m, ts in by.items():
        assert all(t["method"] == m and t["mrcm"] is not None for t in ts)
        ratios = [t["mrcm"] / t["planner"] for t in ts]
        assert ratios == sorted(ratios)


def test_search_with_a_methods_finishes_is_never_dearer_than_the_method() -> None:
    """D-60: the root's first leaf is the method's own plan."""
    task = next(t for t in SC.tasks({"F"}, None) if t["set"] == "F-facda")
    rec = SC.run_task(task, "uniform", 8, OPTS, "paper")
    assert rec["finisher"] == "facda"
    assert rec["search_cost"] is not None and rec["search_cost"] <= task["mrcm"] + 1e-9


def test_reading_on_f_counts_methods_improved() -> None:
    def recs(below: int) -> list[dict[str, Any]]:
        return [
            {
                "planner": 100.0,
                "optimum": None,
                "mrcm": 50.0,
                "search_cost": 40.0 if i < below else 50.0,
                "planner_tail": 1,
                "capped": False,
                "search_seconds": 1.0,
            }
            for i in range(13)
        ]

    groups = {(f"F-{m}", f"uniform/{m}", 256): recs(5) for m in SC.F_METHODS}
    assert SC.f_reading(groups)[-1].endswith(
        "4 of 4 improved: the search improves most published methods (with MRCM, E-55)"
    )
    groups[("F-dfm", "uniform/dfm", 256)] = recs(4)
    groups[("F-facda", "uniform/facda", 256)] = recs(0)
    assert SC.f_reading(groups)[-1].endswith(
        "2 of 4 improved: the search improves some of these methods"
    )
    assert SC.f_reading({}) == []


def test_search_with_strong_finishes_is_never_dearer_than_its_finisher() -> None:
    """D-64: the root's first leaf is the strong finisher's own plan from the empty chip."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    root = SC.StrongFinish(inst)(SC.PrepState.initial(inst))
    assert root is not None
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0}, "strong")
    assert rec["finisher"] == "strong"
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(root[1]) + 1e-9


def test_search_with_greedy_mrcm_finishes_is_never_dearer_than_its_finisher() -> None:
    """D-65: the root's first leaf is the greedy MRCM pass's own plan from the empty chip."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    root = SC.StrongFinish(inst, lookahead=0)(SC.PrepState.initial(inst))
    assert root is not None
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0}, "share_mrcm")
    assert rec["finisher"] == "share_mrcm"
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(root[1]) + 1e-9


def test_set_w_is_each_problem_a_published_method_beats_the_old_planner_on() -> None:
    """D-66: the reference is the cheapest published method, below the old planner."""
    tasks = SC.w_tasks(2)
    assert tasks
    for t in tasks:
        assert t["set"] == "W" and t["method"] in SC.W_METHODS
        assert t["mrcm"] == min(t["published"].values()) == t["published"][t["method"]]
        assert t["mrcm"] < t["planner"] - 1e-9


def test_set_w_skips_a_search_where_the_new_planner_closes_the_gap() -> None:
    """D-66: the new planner runs first; the search runs only where a method still wins."""
    for t in SC.w_tasks(3):
        rec = SC.run_task(t, "uniform", 2, OPTS | {"seconds": 10.0}, "share_mrcm")
        assert rec["planner"] <= rec["old_planner"] + 1e-9
        if rec["mrcm"] >= rec["planner"] - 1e-9:
            assert rec["closed_by_planner"] and rec["search_cost"] is None
        else:
            assert rec["search_cost"] is not None
            assert rec["search_cost"] <= rec["new_planner"] + 1e-9


def _w(source: str, method: str, pub: float, old: float, new: float, search: float | None) -> dict:
    closed = pub >= new
    return {
        "set": "W", "source": source, "id": f"{source}-{pub}", "method": method,
        "published": {method: pub}, "mrcm": pub, "old_planner": old, "planner": min(old, new),
        "search_cost": None if closed else search, "closed_by_planner": closed or None,
    }  # fmt: skip


def test_reading_on_w_counts_the_published_wins_left() -> None:
    rs = [
        _w("val_s1_v2", "facda", 77, 144, 142, 76),  # search catches it
        _w("val_s1_v2", "mrcm", 203, 402, 203, None),  # planner closes it
        _w("val_s2_v2", "dfmm", 100, 120, 110, 105),  # left
    ]
    out = SC.w_reading(rs)
    assert "| facda | 1 | 1 | 0 |" in out
    assert "| dfmm | 1 | 1 | 1 |" in out
    assert out[-1].endswith(
        "of 3 problems, the new planner closes 1; on the 2 left the search is at or below the best "
        "published method on 1: published methods keep real wins; look at the ones left"
    )
    assert SC.w_reading(rs[:2])[-1].endswith("no published method beats our method on these sets")
    assert SC.w_reading([]) == []


def test_auto_finisher_keeps_greedy_mrcm_while_it_is_fast() -> None:
    """D-72 (pilot A): with generous switches the auto finisher is v1's greedy MRCM finisher."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    root = SC.StrongFinish(inst, lookahead=0)(SC.PrepState.initial(inst))
    assert root is not None
    slow = {"switch_seconds": 1e9, "root_switch_seconds": 1e9}
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0} | slow, "auto")
    assert rec["finish_mode"] == "share_mrcm" and rec["switched_at"] is None
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(root[1]) + 1e-9


@pytest.mark.parametrize(("limit", "root_limit", "at"), [(1e9, 0.0, 1), (0.0, 1e9, 2)])
def test_auto_finisher_switches_to_the_plain_pass_when_mrcm_is_slow(
    limit: float, root_limit: float, at: int
) -> None:
    """D-72: past a switch the leaves use the plain sharing pass (the root has its own switch,
    as it fills MRCM's cache); the switching leaf keeps the cheaper of the two finishes."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    finish = SC.AutoFinish(inst, limit, root_limit)
    for _ in range(at):
        got = finish(start)
    assert finish.mode == "share" and finish.switched_at == at
    both = [
        f
        for f in (SC.StrongFinish(inst, lookahead=0)(start), finish_from(start, inst, SC.WEIGHTS))
        if f
    ]
    assert got is not None and got[1] == min(f[1] for f in both)
    assert finish(start) == finish_from(start, inst, SC.WEIGHTS)
    opts = OPTS | {"seconds": 10.0, "switch_seconds": limit, "root_switch_seconds": root_limit}
    rec = SC.run_task(task, "uniform", 2, opts, "auto")
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(got[1]) + 1e-9


def test_tiered_finisher_checks_the_root_with_both_and_keeps_the_cheaper() -> None:
    """D-74 (pilot A2): the root gets the plain pass and greedy MRCM, and the cheaper answer."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    finish = SC.TieredFinish(inst, 0.5)
    got = finish(start)
    both = [
        f
        for f in (SC.StrongFinish(inst, lookahead=0)(start), finish_from(start, inst, SC.WEIGHTS))
        if f
    ]
    assert got is not None and got[1] == min(f[1] for f in both)
    assert finish.mrcm_calls == 1 and finish.root_seconds is not None


def test_tiered_finisher_skips_mrcm_past_its_share_of_the_time() -> None:
    """D-74: once MRCM has used more than its share of the time, leaves get the plain pass."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    plain = finish_from(start, inst, SC.WEIGHTS)
    assert plain is not None
    finish = SC.TieredFinish(inst, 0.0)
    finish.calls, finish.mrcm_seconds = 1, 1.0  # the root is done, over its share
    assert finish(start) == plain and finish.mrcm_calls == 0
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0}, "tiered")
    assert rec["finish_mode"] == "tiered" and rec["mrcm_calls"] >= 1
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(plain[1]) + 1e-9


def test_switch_finisher_is_v1_while_mrcm_is_fast_on_average() -> None:
    """D-75 (pilot A3): with a generous mean the switch finisher is v1's greedy MRCM finisher."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    finish = SC.SwitchFinish(inst, 1e9, 3, 0.5)
    for _ in range(3):
        assert finish(start) == SC.StrongFinish(inst, lookahead=0)(start)
    assert finish.mode == "share_mrcm" and finish.switched_at is None
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0, "switch_mean": 1e9}, "switch")
    assert rec["finish_mode"] == "share_mrcm" and rec["switched_at"] is None
    assert rec["mrcm_calls"] is None


def test_switch_finisher_leaves_out_the_root_and_switches_on_the_average() -> None:
    """D-75: the root's finish never counts; the first leaf switches only if it alone is over
    `mean * min_leaves`; once switched, leaves go to A2's tiered finisher with MRCM's cache."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    finish = SC.SwitchFinish(inst, 0.0, 3, 0.0)
    finish(start)  # the root: never switches
    assert finish.mode == "share_mrcm" and finish.root_seconds is not None
    finish(start)  # a leaf slower than 0 s on average: switch
    assert finish.mode == "tiered" and finish.switched_at == 2
    assert finish.tiered is not None and finish.tiered.mrcm is finish.mrcm
    finish.tiered.mrcm_seconds = 1.0  # over its share: the plain pass only
    assert finish(start) == finish_from(start, inst, SC.WEIGHTS)
    slow = SC.SwitchFinish(inst, 1.0, 3, 0.5)
    slow.calls, slow.leaf_seconds = 1, 3.5  # root done; as if the first leaf took 3.5 s
    slow(start)  # 3.5 s and more over 1 leaf > 1 s * max(1, 3): switch at once
    assert slow.mode == "tiered" and slow.switched_at == 2
    rec = SC.run_task(task, "uniform", 2, OPTS | {"seconds": 10.0, "switch_mean": 0.0}, "switch")
    assert rec["finish_mode"] == "tiered" and rec["switched_at"] == 2
    assert rec["search_cost"] is not None


def test_the_search_budget_is_cpu_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-78: the search's budget runs on `CLOCK` (this process's CPU time). With a clock that
    jumps 100 s a reading, a 10 s search stops after its first leaf, however long the wall
    clock says it took."""
    assert SC.CLOCK is SC.time.process_time
    ticks = iter(range(0, 10**9, 100))
    monkeypatch.setattr(SC, "CLOCK", lambda: float(next(ticks)))
    task = SC.tasks({"C"}, 1)[0]
    rec = SC.run_task(task, "uniform", 256, OPTS | {"seconds": 10.0}, "share_mrcm")
    assert rec["leaves"] <= 2 and rec["search_cost"] is not None
    assert all(s % 100 == 0 for s, _ in rec["trace"])


def test_planner_jumps_hold_the_planners_first_decision() -> None:
    """D-73: from the empty chip the jumps include the plain planner's first delivery group,
    one run of actions per (target, candidate), each longer than one action."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    start = SC.PrepState.initial(inst)
    jumps = SC.PlannerJumps(inst, 3, None)(start)
    assert jumps and all(len(j) > 1 for j in jumps) and len(set(jumps)) == len(jumps)
    plan = finish_from(start, inst, SC.WEIGHTS)
    assert plan is not None
    assert any(plan[0][: len(j)] == j for j in jumps)


def test_search_with_jumps_is_never_dearer_than_its_finisher() -> None:
    """D-73: jumps are extra moves; the root's first leaf is still the finisher's own plan."""
    task = SC.tasks({"C"}, 1)[0]
    inst = SC.Instance.from_json(task["instance"])
    root = SC.StrongFinish(inst, lookahead=0)(SC.PrepState.initial(inst))
    assert root is not None
    opts = OPTS | {"seconds": 20.0, "jumps": 3, "jump_share": 0.5}
    rec = SC.run_task(task, "uniform", 4, opts, "share_mrcm")
    assert rec["jumps"] == 3 and rec["leaves"] > 1
    assert rec["search_cost"] is not None and rec["search_cost"] <= float(root[1]) + 1e-9


def test_jumps_take_their_share_of_the_prior() -> None:
    search = SC.Search.__new__(SC.Search)
    search.jump_share = 0.25
    p = search._with_jumps(SC.np.full(6, 1 / 6), 2)
    assert p.sum() == pytest.approx(1.0) and p[-1] == pytest.approx(0.125)
    assert search._with_jumps(SC.np.zeros(0), 4).tolist() == [0.25] * 4
