# Literature mini benchmark: results (plan 2.4, D-36)

*2026-10-27. Branch `claude/lit-bench-b3e8e4`. Methods and the choices behind them:
`selection.md`. Script: `scripts/lit_bench.py`. Costs 64/8/1, mix 1, waste 1. Every plan is
replayed by the engine and its DAG validated; no producer went below a proved optimum.*

**Status.** This run (`runs/e22_reused/`, Mac, 436 problems, 137 s + 8 min) has every
published method and our planner on every set. The agent column holds the E-22 agent's stored
greedy results (`notes/headroom/records.jsonl`), so it exists for the val and E2 sets only. The
agent on the new sets needs the workstation run (`label=e22`); this file is updated from it.

## Sets

| set | problems | what |
|---|---|---|
| `val_s1_v2`, `val_s2_v2` | 40 + 40 | the stage validation sets (clean: never trained on) |
| `e2` | 17 s1 + 7 s2 | E2 problems with a proved optimum (Track 2 or the headroom search, E-28) |
| `litbench_s1_v1`, `litbench_s2_v1` | 100 + 100 | fresh stage-generator draws, seed 20261027/8, none equal to any frozen set's problem. **The E-22 agent trained on random draws from the same generators, so some may match its training problems.** |
| `litbench_paper_v1` | 129 s1 + 3 s2 | the papers' own problems: FacDA Ex. 2 and Fig. 2, Generic Fig. 6(b), every x/64 at 4X (M 4) and 8X (M 8) (Generic Table II(b), III), DFMM 7:7:6, MRCM ⟨25,31,44⟩ and ⟨1,1,1,1⟩ (clean) |

## Headline (ratio = cost / best planner; < 1 is cheaper than our planner)

The three planner variants (`share`, `share_look3`, `share_orders`) give the same plan on every
one of these one-target problems, so "best planner" is `share`.

| set | method | accepted | mean ratio | cheaper / equal / dearer | agent, same problems |
|---|---|---|---|---|---|
| val_s1 | FacDA | 30/40 | 1.095 | 9 / 5 / 16 | 0.972 |
| val_s1 | MRCM | 27/40 | 1.006 | 10 / 13 / 4 | 0.984 |
| val_s1 | agent | 40/40 | 0.978 | 7 / 19 / 14 | — |
| fresh s1 | FacDA | 77/100 | 0.982 | 29 / 19 / 29 | not run yet |
| fresh s1 | MRCM | 55/100 | 0.992 | 17 / 28 / 10 | not run yet |
| paper s1 | FacDA | 120/129 | 0.973 | 27 / 54 / 39 | not run yet |
| paper s1 | MRCM | 128/129 | 0.961 | 45 / 78 / 5 | not run yet |
| val_s2 | MRCM | 17/40 | **0.658** | **13** / 4 / 0 | 1.024 |
| val_s2 | DFMM | 15/40 | 0.974 | 8 / 2 / 5 | 1.035 |
| val_s2 | agent | 34/40 (6 fail) | 1.035 | 8 / 12 / 14 | — |
| fresh s2 | MRCM | 41/100 | **0.783** | **29** / 9 / 3 | not run yet |
| fresh s2 | DFMM | 48/100 | 1.093 | 23 / 8 / 17 | not run yet |
| paper s2 | MRCM | 3/3 | 0.815 | 2 / 1 / 0 | not run yet |

DFM, Generic and DFMM on dilutions are dearer than our planner (mean 1.17–1.73 on val_s1,
1.29–1.58 on fresh s1). Against proved optima (E2): on s1 every method that accepts a problem is
within 0.1% of the optimum except Generic (1.239), our planner is at 1.027 and the agent at
1.045. On the 7 proved s2 problems MRCM is at 1.018, our planner at 1.430 and the agent at 1.425.

Full tables per set (sample units, waste, mixes, each paper's own nm / units / nw):
`runs/e22_reused/summary.md`. Per problem: `runs/e22_reused/per_problem.md`.

## Coverage: why a method declines (all sets)

| method | reason | s1 | s2 |
|---|---|---|---|
| DFM, FacDA | last mix smaller than the target droplet (the root mixer is the largest factor of the ratio-sum) | 47 | — |
| DFMM | ratio-sum has a prime factor above M (tolerant targets; no transformation for mixtures) | 51 | 64 |
| DFMM | last mix smaller than the target droplet | 43 | 15 |
| Generic | plan longer than the instance's step limit / no mix found | 18 / 2 | — |
| MRCM | denominator has a prime factor above M (tolerant targets, built exactly) | 51 | 64 |
| MRCM | no tree in our reading (each intermediate needs a smaller denominator than its parent; e.g. 1/3 at volume 5, M 5) | 10 | 4 |
| MRCM | search budget (2,000,000 candidate children) | 0 | 14 |

The dilution methods (DFM, FacDA, Generic) take only two fluids, so they skip every s2 problem.

## Reading

- **Stage 1:** the best published dilution methods (FacDA, MRCM) are level with our planner on
  average (mean ratio 0.96–1.10), cheaper on about a quarter of the problems they take and
  dearer on about as many. Where an optimum is proved, both sit at it. The agent is level too
  (0.978 on val_s1) and beats FacDA and MRCM on the problems they take there.
- **Stage 2:** MRCM, a cost-weighted search over mixing trees, beats our planner clearly:
  cheaper on 13 of the 17 val problems it takes (mean 0.66) and 29 of 41 fresh ones (mean 0.78),
  mostly by drawing about half the sample and half the waste (on the same 17 val problems:
  1.9 against 3.8 sample units, 5.6 against 11.0 waste; fresh: 2.3 against 3.7, 5.9 against 11.0).
  The E-22 agent, which imitates our planner, is no better there (1.024 on MRCM's problems).
  Our teacher is weak on single-target mixtures, and the agent inherits it.
- MRCM covers under half of s2 (exact targets only, and our reading's limits), so "beats on
  what it takes" is the claim, not "beats on s2".
