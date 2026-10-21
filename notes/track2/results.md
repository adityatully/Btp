# Track 2 experiments: results and insights

*2026-10-21. Produced by `scripts/headroom.py`. Raw records: `e2_headroom.jsonl` and
`e3_grading.jsonl`. Literature side: `e1_coverage.md`.*

## Setup
- **Cells:** dilution or mixture (support 2–3) × one target or 2–3 targets × demand 1 or 2–3 ×
  storage loose (10) or tight (2–3). Four more cells use structured multi-target dilution
  (linear gradients). That is 20 cells, with 20 problems each.
- **Small problems:**
  - M is drawn from 3–8, D from the M-smooth values in 4–16, and droplet volume from 1 to M;
  - targets are exact;
  - costs are 64 / 8 / 1, mix 1, waste 1;
  - the step limit follows the depth rule (multiple 4).
- **Solvers:** each problem is solved by naive, the split tree, and exact search (A* with branch
  and bound, fluid-conservation bound) with a 60 s budget.
- **Headroom** is (best construction cost ÷ proved optimum) − 1.

## E2: headroom on problems small enough to prove

### Result 1: exact search proves only the smallest problems
- **Overall:** 65 of 399 proved (16%).
- **Proof rate by number of deliveries (the sum of demand):**

  | Deliveries | 1 | 2 | 3 | 4 | 5 | 6+ |
  |---|---|---|---|---|---|---|
  | Proved | 30/80 | 18/113 | 11/86 | 1/15 | 4/29 | 1/76 |
- **By cell:** single-target cells proved 2–12 of 20 each; multi-target cells proved 0–4 of 20.
- **Speed:** a median of 112 nodes/s (p10 33, p90 568). A state has about 90 candidate actions at
  M = 7, and each child costs about 0.4 ms in Python, mostly `rules.apply` and the heuristic.
  Large M and loose storage are the slowest.
- **When it times out it never improves on the construction:** 0 of 334 timeouts found a plan
  cheaper than the incumbent. A timeout's "best known" plan is just the tree or naive plan.
- **The lower bound is loose:** on timeouts the interval [lower bound, best known] is 13% / 34% /
  62% wide (p10 / median / p90). The fluid-conservation bound counts only the fluid the targets
  contain. It ignores the waste and mixes every real plan needs.

### Result 2: headroom exists, and it sits where the claim is
| Group | Proved | Headroom > 1% | Typical size |
|---|---|---|---|
| Single target, loose storage | 19 | 4 | median 0%, at most 38% (mixture, demand > 1) |
| Single target, tight storage | 28 | 12 | median 1–82% per cell, up to 149% |
| **Multi-target (all cells)** | 18 | **14** | mostly 50–135% |

- **Single-target dilution with loose storage is close to solved by the constructions.** Median
  headroom is 0%, and at most 1.4% on these cells.
- **Multi-target problems almost always have headroom when proved.** The optimum saves 1–4
  units of sample. That holds for random independent targets as well as gradients. Gradients
  with demand show the most (135%).
- **Tight storage creates headroom even for one target.** The constructions build the target at
  its own denominator volume. The optimum builds a smaller intermediate and takes part of it.

### Result 3: what the optimal plans do differently
Mean difference, optimum − best construction, over proved problems with headroom:

| Group | Sample | Waste | Sharing | Mixes |
|---|---|---|---|---|
| Multi-target dilution | −1 to −3 | −2 to −8 | **+1 to +3** | 0 to −5.5 |
| Single target, tight storage | −0.3 to −2.5 | −1.4 to −4 | 0 | about 0 |
| Mixture, demand > 1 | −0.5 to −2.5 | −2 to −4 | 0 to +1 | −1 to −2 |

- **Multi-target:** the saving comes from **sharing intermediates across targets**. That is
  the MTMD mechanism, and exactly what our split tree and naive don't do, since both build
  each target alone.
- **Single target:** the saving comes from smaller intermediates and partial takes.
- **Under 64 / 8 / 1, the cost is mostly sample units.** One unit (64) is up to half the cost
  of a plan this small. So headroom is best reported as **units of sample saved**, next to cost.

### Caveats
- **Selection bias:** only the problems exact search finished are in the headroom numbers, and
  those are the easiest. Headroom on larger problems is unknown. Since sharing grows with T and
  demand, it is probably larger, but this experiment can't show it.
- **Small n:** some cells have 0–4 proved problems.
- **Small D (4–16):** the training range goes up to 256, where depth and headroom both grow.

## E3: grading strength on medium problems
**Setup:**
- Cells: multi-target with demand > 1 (dilution, dilution gradients, mixtures), storage loose
  or tight.
- Size: T 4–6, demand 2–4, D from the M-smooth values in 8–64. That is 9–22 deliveries per
  problem.
- Problems: 8 per cell, 48 in all. 3 were skipped, all mixtures at storage 2 with no witness
  (the known storage-2 limit).
- Search: exact search at 10, 60 and 300 s on each problem.

**Result: on frontier-sized problems we cannot grade at all, and more time does not help.**
- None of the 45 problems is proved.
- **None of the 45** improves on the construction, even at 300 s. The "best known" plan is
  always our own tree or naive plan, and neither shares across targets.
- The interval width, (best known − lower bound) ÷ best known, per cell:

  | Cell | Median | Range |
  |---|---|---|
  | dilution, loose | 52% | 36–63% |
  | dilution, tight | 39% | 22–56% |
  | gradient, loose | 32% | 9–58% |
  | gradient, tight | 28% | 21–53% |
  | mixture, loose | 50% | 26–69% |
  | mixture, tight | 60% | 33–73% |
- **From 10 s to 300 s, the lower bound rises by at most 0.44% of the plan cost.** The search
  covers 7,000–240,000 nodes in 300 s, which is nowhere near enough at this size.

So a frontier benchmark graded today by [lower bound, best known] would accept any plan within
30–60% of a weak construction. It could not tell a good agent from a mediocre one.

## Follow-up: which end of the interval is loose?
On the 65 problems with a proved optimum, we measured the lower bound (the fluid-conservation
bound at the start) against the true optimum:

| | Bound within 5% of the optimum | Construction already optimal |
|---|---|---|
| Single target (47) | 27 | 31 |
| Multi-target (18) | **14** | **4** |

- **The bound is tight exactly when the optimal plan wastes almost nothing.** It is loose
  (26–87% below the optimum) only when waste is forced. Typically that is a 1–2 unit delivery
  of one target: a mix needs at least 2 units, so the rest of the droplet is wasted, and the
  bound does not count waste.
- **In multi-target problems, leftovers go to other deliveries, so the optimum rarely wastes.**
  The bound is within 5% in 14 of 18 cases, and the gap is almost all the construction's excess.
- **Inference for E3 (not proved):** medium multi-target, multi-demand problems have even more
  deliveries to absorb leftovers. So their 28–60% intervals are probably mostly
  **construction excess, not a loose bound.** If that holds, the headroom on the frontier is
  large, and the fix that matters most is a stronger *best-known plan* (option B), not a
  tighter bound (option C).

## Insights
1. **The claim holds where it matters.**
   - Headroom is concentrated in multi-target cells (sharing) and tight-storage cells, which
     are the frontier cells of `e1_coverage.md`.
   - In 14 of the 18 proved multi-target problems, the optimum saved 1–4 sample units over our
     constructions.
   - Single-target dilution with loose storage is close to solved by construction. It is a
     sanity check, not a result.
2. **The answer key, not the agent, is now the bottleneck for the claim.**
   - Exact search proves 16% of small problems (almost none with 4 or more deliveries) and none
     of the medium ones.
   - It never improves on a timeout.
   - Its lower bound is loose, and does not tighten with time.
3. **Our constructions are the wrong reference on the frontier.** They do not share, and sharing
   is exactly where the headroom is. Beating them is a weak result.
4. **Report sample units saved next to cost.** Under 64 / 8 / 1 the cost is mostly sample.

## Decision needed: how to strengthen the answer key
The frontier claim needs two things: a strong "best known" plan, so a gap means something, and
a tighter lower bound, so we can say how close to optimal we are. The options, roughly cheapest
first:

| # | Option | Improves | Cost / risk |
|---|---|---|---|
| A | **Speed up exact search** (no re-validation of moves already known to be legal, cached heuristic, dedupe of equivalent children) | provable region for G2: probably a few × more nodes/s, so somewhat larger small problems | days; no dependency. It won't reach medium problems, which need orders of magnitude. |
| B | **A sharing-aware search baseline**: beam search or large-neighbourhood search over our engine, seeded by the tree | the "best known" end on the frontier. It is also a classical non-learned competitor, which a reviewer will expect anyway. | days; no dependency |
| C | **A tighter admissible bound** (the LP relaxation over fluid flows that SPEC 10.2 reserves, or a waste-aware bound) | the lower end, on every size | uncertain tightness; an LP solver is a new dependency (needs your OK) |
| D | **An exact ILP/SMT formulation** for small and medium problems, as FacDA and MTWM use SMT | proofs beyond A* | needs z3 or an ILP solver (needs your OK); scaling unknown |
| E | **The adapted literature references from E1 §3** (DMFB demand/storage planners, per-target MEDA methods) | a published-method reference on the frontier | part of the baseline work anyway |

Revised by the follow-up above, my recommendation: **B first** (it attacks the loose end), **A** for G2, then **E**; a small forced-waste refinement of the bound is cheaper than **C** and may be enough. Originally: **B and A first**, since neither needs a dependency, then **E** as the
baselines are built. Try **C** time-boxed once you approve a solver. Frontier results would
then be reported as the gap to the best of B and E, with the lower bound shown alongside. A
benchmark is admitted only once its references are strong.
