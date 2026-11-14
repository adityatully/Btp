"""Execution harness, part 2: the grid (SPEC 14.6-14.10, milestone M11).

Expected numbers are worked out by hand from the grids, not read off the code.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from meda.core.actions import Mix, Output, Source
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.executor import ExecutionDeadlock, GridExecutor, Run, grid_opgraph
from meda.pipeline.geometry import Chip, GridPlatform, Rect, SizeRule, layout, window_any
from meda.pipeline.opgraph import OpKind
from meda.pipeline.router import INF, AStarRouter, Body, ChipState, RouteRequest
from meda.pipeline.run_check import RunError, RunRule, check_run, run_metrics, validate_run
from meda.pipeline.schedule import critical_path
from meda.solvers.baselines.share import ShareSolver
from tests.core.helpers import chip, dilution
from tests.pipeline.test_schedule import SHARED, SHARED_PLAN, _dag, _weights

GP = GridPlatform.load()
S, B = Source.reservoir(0), Source.reservoir(1)


def _small(rows: int, cols: int) -> GridPlatform:
    """A grid for hand cases: droplets up to 4 units, one 3 x 3 mixer, one detector, two ports
    per fluid (so dispensing never makes two mixes queue)."""
    return replace(
        GP,
        grid=(rows, cols),
        detectors=1,
        ports_per_fluid=2,
        footprints=(SizeRule(1, (1, 1)), SizeRule(2, (1, 2)), SizeRule(4, (2, 2))),
        mixer_library=(SizeRule(4, (3, 3), 3),),
    )


def _bare(rows: int, cols: int) -> ChipState:
    gp = _small(rows, cols)
    return ChipState(Chip("hand", rows, cols, {}, (), (), frozenset(), Rect(0, 0, rows, cols)), gp)


def _gaps_ok(a: Body, b: Body, spacing: int) -> bool:
    for t in range(min(a.start, b.start), max(a.arrival, b.arrival) + 2):
        ra = a.rect(t)
        for u in (t - 1, t, t + 1):
            rb = b.rect(u)
            if ra is not None and rb is not None and ra.gap(rb) < spacing:
                return False
    return True


# ---------------------------------------------------------------- geometry


def test_grid_platform_loads_and_checks_its_tables() -> None:
    assert GP.grid == (60, 60) and GP.mixer(4) == ((3, 3), 15) and GP.mixer(8) == ((4, 4), 20)
    assert GP.footprint(1) == (1, 1) and GP.footprint(10) == (3, 4)
    with pytest.raises(ValueError, match="no mixer for volume 11"):
        GP.mixer(11)
    with pytest.raises(ValueError, match="does not fit its mixer"):
        replace(GP, mixer_library=(SizeRule(12, (2, 2), 1),))


def test_layout_puts_ports_detectors_and_waste_on_the_edges() -> None:
    c = layout(SHARED, GP)
    (west,), (north,) = c.ports[0], c.ports[1]
    assert west[1] == 0 and north[0] == 0
    assert all(d.c2 == GP.grid[1] - 1 for d in c.detectors) and len(c.detectors) == 2
    assert all(w[0] == GP.grid[0] - 1 for w in c.waste)
    assert c.interior == Rect(4, 4, 52, 52)
    with pytest.raises(ValueError, match="do not fit"):
        layout(SHARED, replace(GP, grid=(8, 60), detectors=4))


def test_window_any_grows_by_the_pad() -> None:
    import numpy as np

    g = np.zeros((5, 5), dtype=bool)
    g[2, 2] = True
    assert window_any(g, (1, 1), 0).sum() == 1
    assert window_any(g, (1, 1), 1).sum() == 9  # every anchor within one cell
    assert window_any(g, (2, 2), 0).sum() == 4


# ---------------------------------------------------------------- router (accept a)


def test_router_takes_the_chebyshev_distance_on_an_empty_grid() -> None:
    st = _bare(10, 10)
    res = AStarRouter().route([RouteRequest(1, (1, 1), 1, (0, 0), (5, 3), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 5 and 1 in st.bodies


def test_router_detours_around_a_stored_droplet() -> None:
    """A 1 x 1 droplet at (2, 2) with spacing 1 blocks rows 1-3 x cols 1-3. From (2, 0) to
    (2, 4): down to row 0 by col 1 (2 moves), along to col 3 (2), back to (2, 4) (2) = 6."""
    st = _bare(10, 10)
    st.put(Body(9, (1, 1), 1, 0, ((2, 2),), INF, "stored"))
    res = AStarRouter().route([RouteRequest(1, (1, 1), 1, (2, 0), (2, 4), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 6


def test_two_crossing_droplets_keep_their_spacing() -> None:
    st = _bare(5, 9)
    reqs = [
        RouteRequest(1, (1, 2), 2, (2, 0), (2, 7), 0, to_waste=True),
        RouteRequest(2, (1, 1), 1, (2, 8), (2, 0), 0, to_waste=True),
    ]
    assert AStarRouter().route(reqs, st).ok
    assert _gaps_ok(st.bodies[1], st.bodies[2], 1)


def test_impossible_request_fails_and_reserves_nothing() -> None:
    st = _bare(7, 7)
    st.put(Body(9, (1, 1), 1, 0, ((5, 5),), INF, "stored"))  # parked next to the goal corner
    reqs = [
        RouteRequest(1, (1, 1), 1, (0, 0), (0, 3), 0, to_waste=True),
        RouteRequest(2, (1, 1), 1, (0, 6), (6, 6), 0),
    ]
    res = AStarRouter().route(reqs, st)
    assert not res.ok and res.failed == 1 and set(st.bodies) == {9}


# ---------------------------------------------------------------- area is the resource (accept b)

TWO = dilution([(1, 2), (1, 2)], chip_limits=chip("dmfb_m2", max_denominator=2))
TWO_PLAN = [
    Mix(S, 1, B, 1),
    Mix(S, 1, B, 1),
    Output(Source.droplet(2), 0),
    Output(Source.droplet(3), 1),
]


def _mix_times(gp: GridPlatform) -> tuple[Run, list[tuple[int, int]]]:
    dag = _dag(TWO, TWO_PLAN)
    g = grid_opgraph(dag, TWO, gp)
    c = layout(TWO, gp)
    run = GridExecutor(gp).run(g, c)
    validate_run(run, g, c, gp)
    pairs = zip(g.ops, run.records, strict=True)
    mixes = [(r.start, r.end) for o, r in pairs if o.kind is OpKind.MIX]
    return run, mixes


def test_one_module_of_room_runs_two_mixes_one_after_the_other() -> None:
    """Interior 6 x 6 (14 x 14, band 4): one 3 x 3 module with its ring fits, not two."""
    run, ((s1, e1), (s2, e2)) = _mix_times(_small(14, 14))
    assert e1 <= s2 or e2 <= s1
    assert run.waits["space"] > 0


def test_two_modules_of_room_run_the_mixes_together() -> None:
    """Interior 6 x 10 (14 x 18): two modules side by side."""
    _, ((s1, e1), (s2, e2)) = _mix_times(_small(14, 18))
    assert s1 < e2 and s2 < e1


# ---------------------------------------------------------------- val plans (accept c-f)


def _val(stage: int, n: int | None):  # type: ignore[no-untyped-def]
    w = _weights()
    for inst in BenchmarkSet.read(f"val_s{stage}_v2", BENCHMARKS_DIR).instances[:n]:
        dag = ShareSolver(w).solve(inst).dag
        assert dag is not None
        g = grid_opgraph(dag, inst, GP)
        yield inst, g, layout(inst, GP)


def _runs_validate(stages: tuple[int, ...], n: int | None, may_deadlock: bool) -> None:
    for stage in stages:
        for inst, g, c in _val(stage, n):
            try:
                run = GridExecutor(GP).run(g, c)
            except ExecutionDeadlock as e:
                assert may_deadlock, str(e)
                assert inst.instance_id in str(e)
                continue
            validate_run(run, g, c, GP)
            m = run_metrics(run, g, c, GP)
            cp = max(critical_path(g).values())
            assert m.makespan >= cp and run.schedule_makespan >= cp
            assert m.storage_peak <= g.storage_capacity


def test_val_plans_run_and_validate_sample() -> None:
    _runs_validate((1, 2, 3), 2, may_deadlock=False)
    _runs_validate((4, 5), 2, may_deadlock=True)


@pytest.mark.slow
def test_val_plans_run_and_validate_all() -> None:
    """Acceptance M11 (c, e): every sharing-pass plan of val_s1_v2 ... val_s5_v2."""
    _runs_validate((1, 2, 3), None, may_deadlock=False)
    _runs_validate((4, 5), None, may_deadlock=True)


def _a_run() -> tuple[Run, object, Chip]:
    dag = _dag(SHARED, SHARED_PLAN)
    g = grid_opgraph(dag, SHARED, GP)
    c = layout(SHARED, GP)
    return GridExecutor(GP).run(g, c), g, c


def test_shared_plan_runs_and_is_deterministic() -> None:
    run, g, c = _a_run()
    validate_run(run, g, c, GP)  # type: ignore[arg-type]
    assert run == _a_run()[0]
    m = run_metrics(run, g, c, GP)  # type: ignore[arg-type]
    # the 1/2 droplet waits for its second user; the run is longer than the no-travel schedule
    assert m.storage_ticks > 0 and m.makespan > m.schedule_makespan and m.travel_cells > 0


def _tampered(run: Run, body_id: int, **changes: object) -> Run:
    bodies = tuple(replace(b, **changes) if b.id == body_id else b for b in run.bodies)  # type: ignore[arg-type]
    return replace(run, bodies=bodies)


def test_tampering_breaks_the_right_rule() -> None:
    run, g, c = _a_run()
    mover = next(b for b in run.bodies if len(b.path) > 3)
    jump = (mover.path[0], (mover.path[0][0] + 2, mover.path[0][1]), *mover.path[2:])
    bad = check_run(_tampered(run, mover.id, path=jump), g, c, GP)  # type: ignore[arg-type]
    assert RunRule.CONTINUITY in {i.rule for i in bad}

    # park a droplet on top of another (different droplet, alive at the same time)
    a, b = next(
        (a, b)
        for a in run.bodies
        for b in run.bodies
        if a.droplet != b.droplet
        and a.dest != b.dest
        and a.start <= b.start < a.end
        and b.end != INF
        and a.shape == b.shape
    )
    onto = (a.pos(b.start),) * len(b.path)
    with pytest.raises(RunError) as e:
        validate_run(_tampered(run, b.id, path=onto), g, c, GP)  # type: ignore[arg-type]
    assert RunRule.SPACING in e.value.rules

    mix = next(i for i, o in enumerate(g.ops) if o.kind is OpKind.MIX)  # type: ignore[attr-defined]
    moved = replace(run.records[mix], module=Rect(0, 0, 3, 3))
    recs = tuple(moved if i == mix else r for i, r in enumerate(run.records))
    with pytest.raises(RunError) as e:
        validate_run(replace(run, records=recs), g, c, GP)  # type: ignore[arg-type]
    assert RunRule.OPERATIONS in e.value.rules
