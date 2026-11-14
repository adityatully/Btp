"""Protocols of the execution harness (SPEC 14.5). Every classical or learned version of a
stage implements the same protocol, so stages swap without touching each other."""

from __future__ import annotations

from typing import Any, Protocol

from meda.core.dag import MixingDAG
from meda.pipeline.geometry import Cell, Rect, Shape
from meda.pipeline.opgraph import OpGraph
from meda.pipeline.platform import Platform
from meda.pipeline.router import ChipState, RouteRequest, RouteResult
from meda.pipeline.schedule import Schedule


class Scheduler(Protocol):
    """Start tick and resource instance for every operation (14.4)."""

    name: str

    def schedule(self, graph: OpGraph, platform: Platform) -> Schedule: ...


class Router(Protocol):
    """Paths for a list of moves (14.8, the doc's section 8): reserves a path for every request
    in `state`, or nothing; reports success, the first failed request and the ticks taken."""

    name: str

    def route(self, requests: list[RouteRequest], state: ChipState) -> RouteResult: ...


class Placer(Protocol):
    """A module rectangle for a mix served from tick `t`, or None if no room (14.7)."""

    name: str

    def place(self, shape: Shape, sources: list[Cell], state: ChipState, t: int) -> Rect | None: ...


class CostOracle(Protocol):
    """M12: what a DAG costs to execute (completion time, washes, ...); cache by DAG hash."""

    def evaluate(self, dag: MixingDAG) -> Any: ...
