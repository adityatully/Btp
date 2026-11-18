"""Router calls saved as a benchmark (plan 8.6 step 4)."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import replace

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.executor import ExecutionDeadlock, GridExecutor, grid_opgraph
from meda.pipeline.geometry import GridPlatform, layout
from meda.pipeline.route_bench import RecordingRouter, RouteCall, gp_to_dict
from meda.pipeline.router import AStarRouter
from meda.solvers.baselines.share import ShareSolver
from tests.pipeline.test_schedule import SHARED, SHARED_PLAN, _dag, _weights

GP = GridPlatform.load()


def _replays(gp: GridPlatform, chip, calls: list[RouteCall]) -> None:  # type: ignore[no-untyped-def]
    for call in calls:
        back = RouteCall.from_json(json.loads(json.dumps(call.to_json())))
        assert back.requests == call.requests and back.bodies == call.bodies
        assert back.regions == call.regions
        got = AStarRouter().route(list(back.requests), back.state(chip, gp))
        assert got == back.result == call.result


def test_recording_does_not_change_the_run() -> None:
    dag = _dag(SHARED, SHARED_PLAN)
    g, c = grid_opgraph(dag, SHARED, GP), layout(SHARED, GP)
    rec = RecordingRouter(AStarRouter())
    assert GridExecutor(GP, router=rec).run(g, c) == GridExecutor(GP).run(g, c)
    assert rec.calls


def test_saved_calls_replay_to_the_same_paths() -> None:
    """A* on each saved call (after a JSON round trip, state pruned) finds the run's paths:
    nothing that matters to a call was dropped. Turning on, crowded 30 x 30, a stage 5 plan."""
    w = _weights()
    gp = replace(GP, grid=(30, 30), turns=True)
    for inst in BenchmarkSet.read("val_s5_v2", BENCHMARKS_DIR).instances[:2]:
        dag = ShareSolver(w).solve(inst).dag
        assert dag is not None
        g, c = grid_opgraph(dag, inst, gp), layout(inst, gp)
        rec = RecordingRouter(AStarRouter())
        with suppress(ExecutionDeadlock):
            GridExecutor(gp, router=rec).run(g, c)
        assert any(len(x.requests) > 1 for x in rec.calls)
        _replays(gp, c, rec.calls)


def test_platform_round_trips() -> None:
    gp = replace(GP, grid=(30, 30), turns=True)
    assert GridPlatform.from_dict(json.loads(json.dumps(gp_to_dict(gp)))) == gp
