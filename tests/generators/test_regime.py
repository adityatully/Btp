"""Training regimes: stage shares and promotion from validation rounds (SPEC 8.3)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from meda.generators.regime import Regime, RegimeConfig
from meda.generators.stages import StageDesign
from tests.core.helpers import load_yaml

NAN = math.nan
TOTAL = 1_000_000


def regime(kind: str, **overrides: Any) -> Regime:
    cfg = RegimeConfig.from_dict({**load_yaml(f"regime/{kind}.yaml"), **overrides})
    return Regime(cfg, StageDesign.from_dict(load_yaml("stages/track2_v1.yaml")), TOTAL)


def rounds(
    r: Regime, stage: str, gaps: list[float], success: float = 1.0, step: int = 0
) -> list[bool]:
    """Feed one validation round per gap for `stage` (other stages get dummy numbers)."""
    out = []
    for g in gaps:
        results = {s: (0.0, NAN) for s in r.names}
        results[stage] = (success, g)
        out.append(r.observe(step, results))
    return out


def test_the_three_configs_load_and_bad_ones_raise() -> None:
    for name in ("sequential", "expanding", "mixed"):
        assert regime(name).cfg.name == name
    with pytest.raises(ValueError, match="regime must be"):
        regime("expanding", name="annealed")
    with pytest.raises(ValueError, match="keys mismatch"):
        RegimeConfig.from_dict({**load_yaml("regime/mixed.yaml"), "extra": 1})
    with pytest.raises(ValueError, match="fractions"):
        regime("mixed", newest_share=0.0)


def test_a_stage_passes_on_success_and_a_gap_plateau_only() -> None:
    r = regime("sequential")
    # Still improving: no promotion however high the success.
    assert rounds(r, "s1_dilution", [0.5, 0.4, 0.3, 0.2]) == [False] * 4
    # Three rounds no better than 0.2 - 0.01: plateau, and success is high: promoted.
    assert rounds(r, "s1_dilution", [0.2, 0.21]) == [False, False]
    assert rounds(r, "s1_dilution", [0.195]) == [True]
    assert r.newest == 1 and not r.events[-1]["forced"]
    # A plateau below the success threshold does not promote.
    assert rounds(r, "s2_mixture", [0.3] * 6, success=0.9) == [False] * 6


def test_sequential_draws_only_the_newest_stage() -> None:
    r = regime("sequential")
    assert r.stage_weights() == {"s1_dilution": 1.0}
    rounds(r, "s1_dilution", [0.1] * 4)
    assert r.stage_weights() == {"s2_mixture": 1.0}


def test_expanding_adds_stages_then_moves_to_the_final_weights() -> None:
    r = regime("expanding", num_stages=3)
    assert r.stage_weights() == {"s1_dilution": 1.0}
    assert rounds(r, "s1_dilution", [0.1] * 4)[-1]
    assert r.stage_weights() == {"s1_dilution": 0.5, "s2_mixture": 0.5}
    rounds(r, "s2_mixture", [0.1] * 4)
    assert r.stage_weights() == {"s1_dilution": 0.25, "s2_mixture": 0.25, "s3_demand": 0.5}
    assert rounds(r, "s3_demand", [0.1] * 4)[-1]  # the last stage passed: shares change
    assert r.finished and r.events[-1]["event"] == "finished"
    assert r.stage_weights() == pytest.approx(
        {"s1_dilution": 0.05 / 0.3, "s2_mixture": 0.05 / 0.3, "s3_demand": 0.2 / 0.3}
    )
    assert not rounds(r, "s3_demand", [0.0] * 4)[-1]  # nothing changes after that


def test_sequential_keeps_the_last_stage_when_it_finishes() -> None:
    r = regime("sequential", num_stages=2)
    rounds(r, "s1_dilution", [0.1] * 4)
    assert rounds(r, "s2_mixture", [0.1] * 4)[-1] is False  # finished, shares unchanged
    assert r.finished and r.stage_weights() == {"s2_mixture": 1.0}


def test_mixed_uses_the_final_weights_and_never_promotes() -> None:
    r = regime("mixed")
    assert r.stage_weights() == load_yaml("stages/track2_v1.yaml")["final_weights"]
    assert rounds(r, "s1_dilution", [0.1] * 8) == [False] * 8 and not r.events


def test_a_stuck_stage_is_promoted_after_its_share_of_steps_and_marked_forced() -> None:
    r = regime("expanding")
    assert rounds(r, "s1_dilution", [NAN], success=0.0, step=199_999) == [False]
    assert rounds(r, "s1_dilution", [NAN], success=0.0, step=200_000) == [True]
    assert r.events[-1]["forced"] and r.events[-1]["gap"] is None
    assert r.stage_start == 200_000


@pytest.mark.parametrize("kind", ["sequential", "expanding"])
def test_a_cap_that_could_starve_the_last_stage_is_refused(kind: str) -> None:
    """5 stages x 0.3 would force stage 5 in at 120% of the budget, i.e. never."""
    with pytest.raises(ValueError, match="max_stage_fraction"):
        regime(kind, max_stage_fraction=0.3)
    assert regime(kind, max_stage_fraction=0.3, num_stages=3).names[-1] == "s3_demand"
    assert regime("mixed", max_stage_fraction=0.3).names  # mixed never promotes


def test_state_round_trips_and_a_mismatched_checkpoint_raises() -> None:
    r = regime("expanding")
    rounds(r, "s1_dilution", [0.1] * 4)
    rounds(r, "s2_mixture", [0.3, 0.2])
    again = regime("expanding")
    again.load_state_dict(r.state_dict())
    assert again.state_dict() == r.state_dict()
    assert again.stage_weights() == r.stage_weights()
    with pytest.raises(ValueError, match="does not match"):
        regime("sequential").load_state_dict(r.state_dict())
    with pytest.raises(ValueError, match="does not match"):
        regime("expanding", num_stages=3).load_state_dict(r.state_dict())
