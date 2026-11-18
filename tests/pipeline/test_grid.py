"""Execution harness, part 2: the grid (SPEC 14.6-14.10, milestone M11).

Expected numbers are worked out by hand from the grids, not read off the code.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from meda.core.actions import Mix, Output, Source
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.executor import ExecutionDeadlock, GridExecutor, Run, grid_opgraph
from meda.pipeline.geometry import Chip, GridPlatform, Rect, SizeRule, layout, window_any
from meda.pipeline.opgraph import OpKind
from meda.pipeline.router import (
    INF,
    AStarRouter,
    Body,
    ChipState,
    FreeRouter,
    RouteRequest,
    exempt_until,
    free_distances,
    turn_distances,
)
from meda.pipeline.run_check import (
    RunError,
    RunIssue,
    RunRule,
    _check_motion,
    check_run,
    run_metrics,
    validate_run,
)
from meda.pipeline.schedule import critical_path
from meda.solvers.baselines.share import ShareSolver
from tests.core.helpers import chip, dilution
from tests.pipeline import nakamura_reference
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


# ---------------------------------------------------------------- routing gap (D-89)


def test_free_router_ignores_droplets_and_never_fails() -> None:
    """The detour case above: the free router goes straight through, 4 moves not 6; and the
    impossible case succeeds."""
    st = _bare(10, 10)
    st.put(Body(9, (1, 1), 1, 0, ((2, 2),), INF, "stored"))
    res = FreeRouter().route([RouteRequest(1, (1, 1), 1, (2, 0), (2, 4), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 4 and st.bodies[1].end == 5
    st = _bare(7, 7)
    st.put(Body(9, (1, 1), 1, 0, ((5, 5),), INF, "stored"))
    reqs = [
        RouteRequest(1, (1, 1), 1, (0, 0), (0, 3), 0, to_waste=True),
        RouteRequest(2, (1, 1), 1, (0, 6), (6, 6), 0),
    ]
    res = FreeRouter().route(reqs, st)
    assert res.ok and [len(p) - 1 for p in res.paths] == [3, 6] and st.bodies[2].end == INF


def test_free_distances_go_round_dead_cells() -> None:
    """Dead cells at col 2, rows 0-3, on a 5 x 5 chip: a 1 x 1 droplet from (0, 0) to (0, 4)
    must pass (4, 2): 4 moves there, 4 back up = 8."""
    c = Chip("hand", 5, 5, {}, (), (), frozenset((r, 2) for r in range(4)), Rect(0, 0, 5, 5))
    assert int(free_distances(c, GP, (1, 1), (0, 4))[0, 0]) == 8
    assert int(free_distances(c, GP, (1, 1), (0, 1))[0, 0]) == 1


def test_trip_floor_bounds_transit_and_free_runs_have_no_gap() -> None:
    dag = _dag(SHARED, SHARED_PLAN)
    g = grid_opgraph(dag, SHARED, GP)
    c = layout(SHARED, GP)
    run = GridExecutor(GP).run(g, c)
    m = run_metrics(run, g, c, GP)
    assert 0 < m.trip_floor <= m.transit_ticks
    free = GridExecutor(GP, router=FreeRouter()).run(g, c)
    fm = run_metrics(free, g, c, GP)
    assert fm.trip_floor == fm.transit_ticks == fm.travel_cells and fm.route_wait == 0


# ---------------------------------------------------------------- shape-dependent speed (L-21)


def test_move_ticks_follow_width_height_and_the_diagonal_rule() -> None:
    """Nakamura et al. 2026: 1 x 1 one unit any way; a 2 x 1 (two rows) one unit sideways, two
    up or down or diagonally. Shiro et al. 2022: diagonal = width + height."""
    assert {GP.move_ticks((1, 1), dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} == {1}
    assert (GP.move_ticks((2, 1), 0, 1), GP.move_ticks((2, 1), 1, 0)) == (1, 2)
    assert GP.move_ticks((2, 1), 1, 1) == 2 and GP.move_ticks((3, 4), 0, -1) == 4
    assert replace(GP, diagonal_rule="sum").move_ticks((2, 1), -1, 1) == 3
    assert replace(GP, move_rule="uniform").move_ticks((3, 4), 1, 0) == 1
    with pytest.raises(ValueError, match="move_rule"):
        replace(GP, move_rule="fast")


def test_wide_droplet_is_slow_sideways_and_fast_up_and_down() -> None:
    """A 1 x 2 droplet (one row, two columns): 3 cells sideways = 6 ticks, 3 cells down = 3;
    the path holds each anchor for the move's ticks. (Turns off: turned, the sideways trip
    also takes 6, 2 + 2 + 2, by another path.)"""
    st = _bare(10, 10)
    st.gp = replace(st.gp, turns=False)
    res = AStarRouter().route([RouteRequest(1, (1, 2), 2, (0, 0), (0, 3), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 6
    assert res.paths[0] == ((0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2), (0, 3))
    st = _bare(10, 10)
    res = AStarRouter().route([RouteRequest(1, (1, 2), 2, (0, 0), (3, 0), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 3


def test_free_distances_weigh_moves_by_shape() -> None:
    """A 1 x 2 droplet on an empty 6 x 6 chip from (0, 0) to (3, 2): two diagonals (2 ticks
    each) and one vertical move (1) = 5, against 3 moves."""
    c = Chip("hand", 6, 6, {}, (), (), frozenset(), Rect(0, 0, 6, 6))
    assert int(free_distances(c, GP, (1, 2), (3, 2))[0, 0]) == 5
    free = FreeRouter().route(
        [RouteRequest(1, (1, 2), 2, (0, 0), (3, 2), 0, to_waste=True)], _bare(6, 6)
    )
    assert len(free.paths[0]) - 1 == 5


def test_turning_pays_off_on_long_trips_along_the_slow_side() -> None:
    """A 1 x 2 droplet 6 cells to the right: 12 ticks as it is. Turned about its right half to
    2 x 1 one cell on (2 ticks), 5 moves at 1, turned back about its top cell (2) = 9. Over 4
    cells: 8 as it is, 2 + 3 + 2 = 7 turned (pivots, 0.51)."""
    c = Chip("hand", 8, 12, {}, (), (), frozenset(), Rect(0, 0, 8, 12))
    assert int(free_distances(c, GP, (1, 2), (3, 6))[3, 0]) == 12
    assert int(turn_distances(c, GP, (1, 2), (3, 6))[3, 0]) == 9
    assert int(turn_distances(c, GP, (1, 2), (3, 4))[3, 0]) == 7


def test_free_router_options_take_the_fastest_trip() -> None:
    """A 3 x 3 droplet (volume 9) 10 cells to the right: 30 ticks as it is; split into 1 x 1
    pieces: 4 rounds of splits and 4 of merges (ceil(log2 9)) + 10 moves = 18."""
    req = RouteRequest(1, (3, 3), 9, (2, 0), (2, 10), 0, to_waste=True)
    plain = FreeRouter().route([req], _bare(8, 16))
    split = FreeRouter(turn=True, split=True).route([req], _bare(8, 16))
    assert len(plain.paths[0]) - 1 == 30 and len(split.paths[0]) - 1 == 18
    assert split.paths[0][0] == (2, 0) and split.paths[0][-1] == (2, 10)


# ---------------------------------------------------------------- turning (0.50, D-92)


def test_astar_turns_on_a_long_trip_along_the_slow_side() -> None:
    """The free case above, now with A*: 9 ticks with turns, 12 without. Each turn, like a
    move, holds the old shape and anchor for a tick, then lands about a corner; it ends in its
    own shape."""
    st = _bare(8, 12)
    req = RouteRequest(1, (1, 2), 2, (3, 0), (3, 6), 0, to_waste=True)
    res = AStarRouter().route([req], st)
    assert res.ok and len(res.paths[0]) - 1 == 9
    shapes = res.shapes[0]
    assert shapes == st.bodies[1].shapes and len(shapes) == len(res.paths[0])
    assert shapes[0] == shapes[1] == shapes[-1] == (1, 2) and shapes.count((2, 1)) == 7
    assert res.paths[0][2] == (3, 1)  # pivoted about its right half
    st = _bare(8, 12)
    st.gp = replace(st.gp, turns=False)
    res = AStarRouter().route([req], st)
    assert res.ok and len(res.paths[0]) - 1 == 12 and res.shapes == ((),)


def test_astar_does_not_turn_when_it_does_not_pay() -> None:
    """2 cells sideways: 4 ticks as it is; turned, 2 + 1 + 2 = 5."""
    st = _bare(10, 10)
    res = AStarRouter().route([RouteRequest(1, (1, 2), 2, (0, 0), (0, 2), 0, to_waste=True)], st)
    assert res.ok and len(res.paths[0]) - 1 == 4 and res.shapes == ((),)


def test_turning_gets_a_droplet_through_a_one_cell_gap() -> None:
    """A wall of dead cells along row 3 with one gap at col 3: a 1 x 2 droplet cannot pass as
    it is, but can turned to 2 x 1. Every footprint keeps off the dead cells."""
    dead = frozenset((3, c) for c in range(7) if c != 3)
    gp = _small(7, 7)
    chip_ = Chip("hand", 7, 7, {}, (), (), dead, Rect(0, 0, 7, 7))
    req = RouteRequest(1, (1, 2), 2, (0, 0), (5, 0), 0, to_waste=True)
    st = ChipState(chip_, replace(gp, turns=False))
    assert not AStarRouter().route([req], st).ok
    st = ChipState(chip_, gp)
    res = AStarRouter().route([req], st)
    assert res.ok and res.shapes[0]
    body = st.bodies[1]
    for t in range(body.start, body.arrival + 1):
        rect = body.rect(t)
        assert rect is not None and not any(cell in dead for cell in rect.cells())
    issues: list[str] = []
    _check_motion(body, gp, lambda rule, msg: issues.append(msg))
    assert not issues


def test_checker_rejects_bad_turns_and_diagonals_beside_dead_cells() -> None:
    """A 1 x 2 turn takes 2 ticks and pivots about a corner (0.51); a droplet ends in its own
    shape; it may not move diagonally from beside a dead cell (Nakamura et al. 2026)."""
    gp = _small(8, 8)

    def issues(
        path: tuple[tuple[int, int], ...],
        shapes: tuple[tuple[int, int], ...] = (),
        shape: tuple[int, int] = (1, 2),
        dead: frozenset[tuple[int, int]] = frozenset(),
    ) -> str:
        out: list[str] = []
        body = Body(1, shape, 2, 0, path, INF, "move", shapes=shapes)
        _check_motion(body, gp, lambda rule, msg: out.append(msg), dead)
        return "; ".join(out)

    own, turned = (1, 2), (2, 1)
    a, b = (2, 2), (2, 3)  # turned about the right half: the anchor moves one cell right
    assert issues((a, a, b, b, a), (own, own, turned, turned, own)) == ""
    assert "needs 2" in issues((a, b, b, a), (own, turned, turned, own))
    off = (3, 3)  # not one of the four corner pivots of a 1 x 2 at (2, 2)
    assert "not about a corner" in issues((a, a, off, off, a), (own, own, turned, turned, own))
    assert "own shape" in issues((a, a, b), (own, own, turned))
    assert "may not turn" in issues((a, a), ((2, 2), (2, 2)), shape=(2, 2))
    diag = ((2, 2), (2, 2), (3, 3))
    assert issues(diag) == ""
    assert "dead cell" in issues(diag, dead=frozenset({(1, 3)}))  # above the right half
    assert issues(diag, dead=frozenset({(0, 0)})) == ""


# ---------------------------------------------------------------- the Dijkstra check (E-99)


def _astar_time(grid: list[str], turns: bool = True) -> int | None:
    h, w, dead, start, target = nakamura_reference.parse(grid)
    chip_ = Chip("nakamura", h, w, {}, (), (), dead, Rect(0, 0, h, w))
    st = ChipState(chip_, replace(GP, grid=(h, w), turns=turns))
    res = AStarRouter().route([RouteRequest(1, (1, 2), 2, start, target, 0, to_waste=True)], st)
    if res.ok:
        body = st.bodies[1]
        issues: list[str] = []
        _check_motion(body, st.gp, lambda rule, msg: issues.append(msg), dead)
        assert not issues, issues
    return len(res.paths[0]) - 1 if res.ok else None


def test_astar_matches_the_dijkstra_paper_on_its_five_chips() -> None:
    """Nakamura et al. 2026, Figs. 8-12: our A* with turns (no splitting) finds the optimum of
    the paper's graph on every chip; without turns three are blocked. With the published 2 ->
    1 + 1 split, the paper's model is faster still (Q-153)."""
    expect = {"exp1": 10, "exp2": 11, "exp3": 14, "exp4": 16, "exp5": 24}
    for name, grid in nakamura_reference.PAPER_GRIDS.items():
        assert nakamura_reference.route_time(grid) == expect[name], name
        assert _astar_time(grid) == expect[name], name
    assert [_astar_time(g, turns=False) for g in nakamura_reference.PAPER_GRIDS.values()] == [
        10,
        None,
        None,
        16,
        None,
    ]
    grids = nakamura_reference.PAPER_GRIDS.values()
    split = [nakamura_reference.route_time(g, split=True) for g in grids]
    assert split == [8, 7, 9, 10, 15]


def test_astar_matches_the_dijkstra_paper_on_random_chips() -> None:
    """As the paper's large experiments: random dead cells (5-30%), start bottom-left, target
    top-right; 40 chips of 5 x 5 to 10 x 10, including unroutable ones."""
    rng = np.random.default_rng(99)
    for _ in range(40):
        n = int(rng.integers(5, 11))
        frac = float(rng.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
        rows = [["X" if rng.random() < frac else "." for _ in range(n)] for _ in range(n)]
        rows[n - 1][0] = rows[n - 1][1] = "S"
        rows[0][n - 2] = rows[0][n - 1] = "T"
        grid = ["".join(r) for r in rows]
        assert _astar_time(grid) == nakamura_reference.route_time(grid), grid


def test_checker_rejects_a_droplet_moving_too_fast() -> None:
    """Every moving droplet of this run is 1 x 1; claim one is 2 x 2 (2 ticks per move)
    and its one-tick moves break continuity."""
    run, g, c = _a_run()
    assert not any(i.rule is RunRule.CONTINUITY for i in check_run(run, g, c, GP))  # type: ignore[arg-type]
    moved = next(b for b in run.bodies if b.role == "move" and len(set(b.path)) > 1)
    bad = _tampered(run, moved.id, shape=(2, 2))
    assert any(
        i.rule is RunRule.CONTINUITY and "needs 2" in i.message
        for i in check_run(bad, g, c, GP)  # type: ignore[arg-type]
    )


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


def test_run_error_survives_pickling() -> None:
    # Workers send errors back by pickling; a bad round trip hangs the pool.
    import pickle

    err = RunError("x-1", [RunIssue(RunRule.SPACING, "too close")])
    back = pickle.loads(pickle.dumps(err))
    assert str(back) == str(err)
    assert back.rules == {RunRule.SPACING}


def test_remainder_keeps_its_parents_exemption() -> None:
    # val_s5-040 (2026-11-18): a 2 x 2 droplet gives a 1 x 1 part (window from the 1 x 2
    # remainder), then the remainder gives another part and shrinks to 1 x 1 in place. The
    # shrunk remainder is the same liquid: the first part keeps its longer window.
    gp = GP
    parent = Body(1, (2, 2), 4, 0, ((5, 5),), 1, "stored", droplet=0)
    part = Body(2, (1, 1), 1, 1, ((5, 5), (4, 4)), INF, "mix", dest=7, parent=1, droplet=0)
    rem = Body(3, (1, 2), 3, 1, ((5, 5),), 2, "stored", parent=1, droplet=0)
    rem2 = Body(4, (1, 1), 1, 2, ((5, 5),), INF, "stored", parent=3, droplet=0)
    bodies = {b.id: b for b in (parent, part, rem, rem2)}
    before = exempt_until(part, rem, gp)
    assert exempt_until(part, rem2, gp) < before  # counted afresh from the second take
    assert exempt_until(part, rem2, gp, bodies) == before
    assert exempt_until(rem2, part, gp, bodies) == before
    # a part that moves off is not a remainder: it keeps its own window only
    other = Body(5, (1, 1), 1, 2, ((5, 5), (6, 6)), INF, "mix", dest=8, parent=3, droplet=0)
    assert exempt_until(part, other, gp, bodies | {5: other}) == exempt_until(part, other, gp)
