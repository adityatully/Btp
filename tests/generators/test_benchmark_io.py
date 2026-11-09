"""Frozen benchmark sets, canonical hashing and the overlap check (SPEC 8.4; milestone M4)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from meda.core.composition import Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import (
    BENCHMARKS_DIR,
    BenchmarkSet,
    InstanceReference,
    check_no_overlap,
    instance_hash,
    problem_hash,
)
from meda.generators.sampler import sample_instances
from meda.solvers.baselines.naive import NaiveSolver, naive_plan
from tests.core.helpers import chip, default_weights, dilution, mixture
from tests.generators.helpers import load_stage

FROZEN = ("tiny_exact_v1", "val_stage1_v1", "val_stage1_v2")
WEIGHTS = default_weights()


def small_set(name: str = "unit", version: int = 1) -> BenchmarkSet:
    return BenchmarkSet(
        name=name,
        version=version,
        instances=(
            replace(dilution([(1, 4)], chip_limits=chip("dmfb_m2")), instance_id="a"),
            replace(dilution([(1, 2)], chip_limits=chip("dmfb_m2")), instance_id="b"),
        ),
        generator={"builder": "test"},
        seeds=(1, 2),
    )


# --------------------------------------------------------------------------- hashing


def test_hash_ignores_id_metadata_and_step_limit() -> None:
    base = dilution([(1, 4)])
    renamed = replace(base, instance_id="other", metadata={"generator": "x"}, step_limit=99)
    assert instance_hash(base) == instance_hash(renamed)


def test_problem_hash_ignores_the_chip_name_but_not_its_limits() -> None:
    """A generator that draws M names the chip after the draw; the problem is the same."""
    base = dilution([(1, 4)], chip_limits=chip("meda_m8"))
    relabelled = replace(base, chip=replace(base.chip, name="meda_m8-M8-S10"))
    assert problem_hash(relabelled) == problem_hash(base)
    smaller_m = replace(base, chip=replace(base.chip, max_mix_volume=6))
    assert problem_hash(smaller_m) != problem_hash(base)


def test_hash_ignores_target_order_but_not_content() -> None:
    a = dilution([(1, 4), (3, 4)])
    flipped = replace(
        a,
        targets=(
            replace(a.targets[1], id=0),
            replace(a.targets[0], id=1),
        ),
    )
    assert instance_hash(a) == instance_hash(flipped)
    for different in (
        dilution([(1, 4), (3, 8)]),
        dilution([(1, 4), (3, 4)], demand=2),
        dilution([(1, 4), (3, 4)], chip_limits=chip("dmfb_m2")),
        dilution([(1, 4), (3, 4)], droplet_volume=2),
        dilution([(1, 4), (3, 4)], tolerance=Rational.of(1, 8)),
        mixture([(1, 1, 1)]),
    ):
        assert instance_hash(different) != instance_hash(a)


# --------------------------------------------------------------------------- set behaviour


def test_round_trip(tmp_path: Path) -> None:
    bench = small_set().with_reference(
        {"a": InstanceReference(Rational.of(15), "optimal", "exact_search", {"nodes": 12})}
    )
    path = bench.write(tmp_path)
    back = BenchmarkSet.read("unit_v1", tmp_path)
    assert back == bench
    assert back.reference["a"].cost == Rational.of(15)
    assert back.reference["a"].detail == {"nodes": 12}
    assert BenchmarkSet.read("unit", tmp_path) == bench  # by name: newest version
    assert path.name == "unit_v1.json"


def test_published_version_is_frozen(tmp_path: Path) -> None:
    bench = small_set()
    bench.write(tmp_path)
    with pytest.raises(FileExistsError, match="bump `version`"):
        bench.write(tmp_path)
    bench.write(tmp_path, overwrite=True)  # explicit opt-in
    assert replace(bench, version=2).write(tmp_path).name == "unit_v2.json"


def test_read_missing_set_lists_what_exists(tmp_path: Path) -> None:
    small_set().write(tmp_path)
    with pytest.raises(FileNotFoundError, match="unit_v1"):
        BenchmarkSet.read("nope", tmp_path)


def test_rejects_unknown_format(tmp_path: Path) -> None:
    data = small_set().to_dict()
    data["format"] = "other/9"
    with pytest.raises(ValueError, match="unsupported benchmark format"):
        BenchmarkSet.from_dict(data)


def test_rejects_duplicate_ids_and_stray_reference() -> None:
    inst = replace(dilution([(1, 4)]), instance_id="a")
    with pytest.raises(ValueError, match="duplicate instance ids"):
        BenchmarkSet("dup", 1, (inst, inst))
    with pytest.raises(ValueError, match="unknown instances"):
        BenchmarkSet(
            "ref",
            1,
            (inst,),
            reference={"ghost": InstanceReference(Rational.of(1), "optimal", "x")},
        )


def test_by_id() -> None:
    bench = small_set()
    assert bench.by_id("a").instance_id == "a"
    with pytest.raises(KeyError, match="no instance 'zzz'"):
        bench.by_id("zzz")


# --------------------------------------------------------------------------- overlap check


def test_overlap_check_flags_a_renamed_instance() -> None:
    bench = small_set()
    clean = [dilution([(3, 8)], chip_limits=chip("dmfb_m2"))]
    check_no_overlap(clean, bench)
    sneaky = [replace(bench.instances[0], instance_id="train-1", step_limit=50)]
    with pytest.raises(ValueError, match="overlap frozen benchmark"):
        check_no_overlap(sneaky, bench)
    with pytest.raises(ValueError, match="overlap frozen benchmark"):
        check_no_overlap(sneaky, [bench])


# --------------------------------------------------------------------------- the frozen files


@pytest.mark.parametrize("name", FROZEN)
def test_frozen_set_loads_and_is_solvable(name: str) -> None:
    bench = BenchmarkSet.read(name, BENCHMARKS_DIR)
    assert len(bench) > 0
    solver = NaiveSolver(WEIGHTS)
    for inst in bench.instances:
        result = solver.solve(inst)
        assert result.status == "feasible", inst.instance_id
        assert result.dag is not None
        validate(result.dag, inst).raise_if_invalid()
        # metadata keeps the naive witness cost; `reference` holds whatever the best known
        # answer is -- an upper bound when the set is built (M4), the proved optimum once the
        # exact search has run over it (M5), which can only be lower.
        assert Rational.parse(inst.metadata["known_upper_bound"]) == result.cost
        stored = bench.reference[inst.instance_id]
        assert stored.status in ("feasible", "optimal")
        assert stored.cost <= result.cost
        if stored.status == "feasible" and stored.lower_bound is None:
            assert stored.cost == result.cost  # a bare upper bound is the naive witness


@pytest.mark.parametrize("name", FROZEN)
def test_frozen_set_json_is_canonical(name: str) -> None:
    """Re-serialising a frozen file reproduces it byte for byte (no silent drift)."""
    path = BENCHMARKS_DIR / f"{name}.json"
    bench = BenchmarkSet.from_dict(json.loads(path.read_text()))
    assert json.dumps(bench.to_dict(), indent=1, sort_keys=False) + "\n" == path.read_text()


def test_frozen_sets_are_disjoint() -> None:
    tiny = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
    val = BenchmarkSet.read("val_stage1_v1", BENCHMARKS_DIR)
    assert tiny.hashes().isdisjoint(val.hashes())


def test_tiny_exact_stores_proved_optima() -> None:
    """After M5 every instance of the exact set carries an optimum, not an upper bound."""
    tiny = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
    assert len(tiny.reference) == len(tiny)
    for inst in tiny.instances:
        stored = tiny.reference[inst.instance_id]
        assert stored.status == "optimal", inst.instance_id
        assert stored.producer == "exact_search"
        assert stored.detail["search_exhausted"] in (True, False)
        assert stored.detail["mix_ops"] >= 0


def test_tiny_exact_holds_the_hand_checked_cases() -> None:
    tiny = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
    ids = {i.instance_id for i in tiny.instances}
    assert {
        "tiny-quarter-m2",
        "tiny-quarter-m4",
        "tiny-three-tenths-m8",
        "tiny-sharing-m2",
    } <= ids
    # Case 3 is infeasible under dmfb_m2 with tolerance 0 (3/10 is not dyadic).
    case3 = tiny.by_id("tiny-three-tenths-m8")
    assert naive_plan(replace(case3, chip=chip("dmfb_m2"))) is None
    # The naive plan for the sharing case uses 4 mixes; the optimum (M5) is 3.
    _, totals, _ = replay(
        tiny.by_id("tiny-sharing-m2"), naive_plan(tiny.by_id("tiny-sharing-m2")) or ()
    )
    assert totals.mix_ops == 4


def test_training_draws_avoid_the_frozen_stage1_set() -> None:
    """Acceptance: no overlap between training samples and the frozen sets."""
    cfg, chip_limits = load_stage("tiny_dilution")
    val = BenchmarkSet.read("val_stage1_v1", BENCHMARKS_DIR)
    tiny = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
    held_out = val.hashes() | tiny.hashes()
    training = sample_instances(
        cfg, chip_limits, np.random.default_rng(99), 5, WEIGHTS, unique=False, exclude=held_out
    )
    check_no_overlap(training, [val, tiny])


def test_instances_in_frozen_sets_round_trip_as_dags() -> None:
    bench = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
    inst: Instance = bench.instances[0]
    plan = naive_plan(inst)
    assert plan is not None
    _, totals, deltas = replay(inst, plan)
    dag = MixingDAG.from_deltas(inst, deltas, totals, "naive")
    assert MixingDAG.from_dict(dag.to_dict()) == dag


def test_val_stage1_v2_is_v1_with_a_better_answer_key() -> None:
    """Same instances (so the same exam), and for each one either a proved optimum or an
    interval [proved lower bound, best known] with both ends no worse than v1's."""
    v1 = BenchmarkSet.read("val_stage1_v1", BENCHMARKS_DIR)
    v2 = BenchmarkSet.read("val_stage1_v2", BENCHMARKS_DIR)
    assert v2.instances == v1.instances
    assert v2.hashes() == v1.hashes()
    assert set(v2.reference) == set(v1.reference)
    for inst in v2.instances:
        old, new = v1.reference[inst.instance_id], v2.reference[inst.instance_id]
        assert new.cost <= old.cost, inst.instance_id
        if new.status == "optimal":
            assert new.lower_bound is None and new.producer == "exact_search"
        else:
            assert new.status == "feasible"
            assert new.lower_bound is not None and 0 < new.lower_bound <= new.cost
            assert new.detail["lower_bound_producer"].startswith("exact_search")
            assert Rational.parse(new.detail["naive_cost"]) == old.cost


TEST_SETS = ("test_s1_v1", "test_s2_v1", "test_s3_v1", "test_s4_v1", "test_s5_v1")


@pytest.mark.parametrize("name", [*TEST_SETS, "test_heldout_v1", "test_emrcm_v1"])
def test_paper_test_sets_share_no_problem_with_any_other_set(name: str) -> None:
    """Benchmark plan B1 (D-68): the test sets are fresh; no problem in any other frozen set,
    and an answer key of [bound, best known] with nothing produced by our method."""
    bench = BenchmarkSet.read(name, BENCHMARKS_DIR)
    assert len(bench) >= 100
    others = [p.stem for p in BENCHMARKS_DIR.glob("*_v*.json") if p.stem != name]
    for other in others:
        assert not bench.hashes() & BenchmarkSet.read(other, BENCHMARKS_DIR).hashes(), other
    for ref in bench.reference.values():
        assert ref.producer == "cheapest_construction", ref  # naive or split tree, not ours
        assert ref.lower_bound is None or ref.lower_bound <= ref.cost


def test_emrcm_protocol_set_follows_the_paper() -> None:
    """Benchmark plan B2 (D-70): e-MRCM's experiments at 1/512, 4X targets, mixes of 8."""
    bench = BenchmarkSet.read("test_emrcm_v1", BENCHMARKS_DIR)
    cells: dict[str, int] = {}
    for inst in bench.instances:
        cells[inst.metadata["cell"]] = cells.get(inst.metadata["cell"], 0) + 1
        assert inst.chip.max_mix_volume == 8 and inst.chip.max_denominator == 512
        assert all(t.droplet_volume == 4 and t.demand == 1 for t in inst.targets)
        n, t = (int(x[1:]) for x in inst.metadata["cell"].split("-"))
        assert inst.num_fluids == n and len(inst.targets) == t
    assert cells.pop("N2-T1") == 511
    assert all(cells.pop(f"N{n}-T1") == 512 for n in (3, 4, 5))
    assert cells == {f"N{n}-T{t}": 50 for n in range(2, 6) for t in range(2, 6)}


def test_dmfb_sweep_is_every_dyadic_target_with_textbook_bit_scanning() -> None:
    """Benchmark plan B3 (D-71, SPEC 8.4): every n/2^d, d = 4..8, on the 1:1 chip at precision
    2^d; bit scanning (the naive plan under M = 2) takes d - tz(n) mixes, one sample unit per
    set bit of n."""
    bench = BenchmarkSet.read("dmfb_sweep_v1", BENCHMARKS_DIR)
    assert len(bench) == 15 + 31 + 63 + 127 + 255
    seen = set()
    for inst in bench.instances:
        d = int(inst.metadata["cell"][1:])
        assert inst.chip.max_mix_volume == inst.chip.min_mix_volume == 2
        assert inst.chip.max_denominator == 2**d
        (target,) = inst.targets
        n = target.composition.num[0] * 2**d // target.composition.den
        seen.add((n, d))
        _, totals, _ = replay(inst, naive_plan(inst))
        tz = (n & -n).bit_length() - 1
        assert totals.mix_ops == d - tz
        assert totals.reagent_by_fluid[0] == bin(n).count("1")
    assert seen == {(n, d) for d in range(4, 9) for n in range(1, 2**d)}


def test_emrcm_val_set_is_fresh_and_weighted_to_four_and_five_fluids() -> None:
    """D-72: the leaf-finisher pilot's development set, e-MRCM's protocol at N = 3-5, with no
    problem shared with any other frozen set (above all `test_emrcm_v1`)."""
    bench = BenchmarkSet.read("val_emrcm_v1", BENCHMARKS_DIR)
    cells: dict[str, int] = {}
    for inst in bench.instances:
        cells[inst.metadata["cell"]] = cells.get(inst.metadata["cell"], 0) + 1
        assert inst.chip.max_mix_volume == 8 and inst.chip.max_denominator == 512
        assert all(t.droplet_volume == 4 and t.demand == 1 for t in inst.targets)
        n, t = (int(x[1:]) for x in inst.metadata["cell"].split("-"))
        assert inst.num_fluids == n and len(inst.targets) == t
    assert cells == {"N3-T1": 20, "N4-T1": 30, "N5-T1": 30, "N3-T3": 10, "N4-T3": 10,
                     "N5-T3": 10, "N4-T5": 5, "N5-T5": 5}  # fmt: skip
    for other in BENCHMARKS_DIR.glob("*_v*.json"):
        if other.stem != "val_emrcm_v1":
            assert not bench.hashes() & BenchmarkSet.read(other.stem, BENCHMARKS_DIR).hashes()
