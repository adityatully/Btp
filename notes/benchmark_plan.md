# Paper benchmark plan (plan phase 6, D-67)

*2026-11-07. How we test our method across all stages and against every published method, so
the numbers are unbiased and true. SPEC 8.4 (tiers) and 12.2–12.3 (evaluation) still apply;
this file says what we build, in what order, and the rules that keep it fair.*

## 1. Why the numbers so far are not paper results

Everything up to E-63 was development. Four biases are built in:

1. **We tuned on the sets we measured on.** The val sets, lit-bench sets and E2 shaped every
   choice from D-37 to D-66. Numbers on them are optimistic.
2. **We picked the hard cases after looking.** Set C (planner > 2% above bound) and set W
   (where a method beat us) were chosen to find gaps. Good for diagnosis, wrong for a headline.
3. **Our own method sets the "best known".** A gap to a best known that includes our plan
   flatters us.
4. **Unequal terms.** Our search gets up to 900 s; the published methods run in milliseconds,
   most optimise steps or waste rather than our cost, and we built them ourselves ("our
   reading").

## 2. Rules (fixed before any test run)

1. **Freeze the method.** One config file and a git tag (`method-v1`): search over the greedy
   MRCM pass, flat prior, 256 simulations per move, time cap, the planner variants. After the
   first test run nothing changes. A bug fix means rerunning every producer on every set, with
   a paper log entry.
2. **Fresh test sets.** New seeds, written down before drawing; `check_no_overlap` against
   every val, lit-bench, E2 and training set. The val sets become "dev" and are never reported
   as results.
3. **Answer keys first.** Lower bound, exact optimum where it finishes, and best known from
   producers *other than ours* are stored before our method runs. Our gap is reported to the
   lower bound (a true upper limit on our gap) and to the optimum where proved.
4. **Readings written in advance.** Each tier gets a paper log D-entry with its bar before it
   runs, as we did for D-37 to D-66.
5. **Every problem counts.** Whole sets, no subsets picked afterwards. Losses are listed.
6. **Three goals, not one.** Our method runs on every set under all three goals the
   literature uses: fluid-first (`costs/geometric`, ours so far), steps-first and waste-first
   (`costs/steps_first`, `costs/waste_first`, as in E-52). Every table shows cost, sample,
   steps and waste as separate columns. Each published method is compared with our method run
   under *its* goal (steps: DFM, Generic; fluid: FacDA, MRCM, e-MRCM, NFSP; waste: STWM-type).
7. **Published methods earn their place.** A reimplementation counts only once it reproduces
   its paper's worked example or printed table (SPEC 10.4). Where it does not, it is labelled
   "our reading" and its paper's printed numbers are shown next to ours.
8. **Coverage is a result.** A method that declines a problem is reported as "not applicable,
   reason", never silently dropped. Comparisons are on the problems both accept, with the
   count.
9. **Time is reported with cost.** Same machine (coe9), one core per problem. Our method's
   cost at 1, 10, 60, 300, 900 s, its time to final plan, and the planner alone (≈1 s) as a
   separate fast method.
10. **Statistics.** Paired per problem: wins / ties / losses, mean and median cost ratio, 95%
    bootstrap interval (numpy, no new dependency). Search seeds: 3 where results can vary.

## 3. Producers in every table

| Group | Producers |
|---|---|
| Ours | search over the greedy MRCM pass (the method); the best planner alone (fast) |
| Published, MEDA k:l | DFM, FacDA, Generic, DFMM, MRCM, e-MRCM (built, reproduced) |
| Published, DMFB 1:1 | NFSP (built); MTMD, DDMT (not built: demand and storage, frontier) |
| Floors | naive, split tree |
| References | exact search (small problems), lower bound |

## 4. The benchmarks, in build order

Each step ends with frozen sets, a run on coe9, a report and a paper log entry.

**B1. Test sets per stage (tier 1, the main table).** *Sets built 2026-11-07 (D-68); runner `scripts/bench.py`, reading D-69.* `test_s1_v1` … `test_s5_v1` and
`test_heldout_v1`: fresh draws from the same stage generators as the val sets, 100 per stage
(about 600 problems). Every producer on every problem. Answers: across all stages, how close
is our method to the bound and to the optimum, and how does it compare with each published
method that applies? Reuses `evaluate.py`, `lit_bench*.py`, `search_check.py`.

**B2. The papers' own experiments (tier 3).** *Built 2026-11-07 (D-70): `test_emrcm_v1` and
`litbench_paper_v1`; NFSP's experiments are 1:1 (M = 2), so they moved to B3.* One set per paper, built from its own protocol,
so we can put our numbers next to its printed table:
- FacDA / Generic: every x/64 at 4X and 8X (Table III) — exists in `litbench_paper_v1`,
  re-frozen as `literature_facda_v1`.
- e-MRCM: N = 2 all 511 targets at P = 1/512; 512 random targets for N = 3–5; 50 random target
  sets per (N, T) for T = 2–5 (Table III).
- NFSP: d = 10, all 1023 targets; its many-target table at d = 9 (Table II).
- MRCM, DFMM: their worked examples and sweeps.
Answers: do we reproduce each paper's numbers, and do we beat it on its own problems and
metric?

**B3. DMFB sanity (tier 2).** `dmfb_sweep_v1`: every n/2^d, d = 4–8, at M = 2. A test of the
engine against published 1:1 numbers, never reported as a win (SPEC 8.4).

**B4. Scaling and generalisation (tiers 4–5).** `frontier_v1`: one base problem varied along
one axis at a time (targets, fluids, demand, storage, accuracy, M), past the trained ranges
(`larger_v1`, `unseen_ratios_v1`). Answers: cost and time against size, and where search
stops paying.

**B5. Real assays (tier 6).** `assays_v1`: ratios from real protocols, one citation each.
Needs sources.

## 5. Decisions for the user (one at a time, when the step is reached)

1. ~~Headline time budget~~ (Q-139): both 60 s and 900 s are run and reported (user,
   2026-11-07); which heads the tables is Tom's call.
2. ~~PPO agent~~: left out of the benchmark (user, 2026-11-07); it is not part of the method.
3. **Before B4:** build MTMD and DDMT (DMFB methods with demand and storage) as adapted
   baselines for the frontier, or report the frontier against floors and bounds only.
4. **Before B5:** which protocols and papers to take the assay ratios from.

## 6. Risks

- **Compute.** B2's sweeps are thousands of problems; at 900 s each they take days on 30
  cores. Most are tiny (seconds); the budget is fixed per set in its D-entry.
- **Time caps make results depend on the machine.** All runs on coe9, the cap and the machine
  named in every table.
- **Reproduction gaps.** e-MRCM's Table III does not reproduce under our reading (Q-117); it
  stays "our reading" with the paper's printed numbers alongside.
