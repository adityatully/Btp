"""Protocols of the execution harness (SPEC 14.5). Every classical or learned version of a
stage implements the same protocol, so stages swap without touching each other."""

from __future__ import annotations

from typing import Any, Protocol

from meda.core.dag import MixingDAG
from meda.pipeline.opgraph import OpGraph
from meda.pipeline.platform import Platform
from meda.pipeline.schedule import Schedule


class Scheduler(Protocol):
    """Start tick and resource instance for every operation (14.4)."""

    name: str

    def schedule(self, graph: OpGraph, platform: Platform) -> Schedule: ...


class Router(Protocol):
    """M11: paths for a list of (droplet, source, target) moves before a deadline, or failure;
    reports time taken, cells newly contaminated, success (doc section 8)."""

    def route(self, requests: list[Any], chip_state: Any) -> Any: ...


class CostOracle(Protocol):
    """M12: what a DAG costs to execute (completion time, washes, ...); cache by DAG hash."""

    def evaluate(self, dag: MixingDAG) -> Any: ...
