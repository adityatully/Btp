# Literature mini benchmark: which published methods (plan 2.4, D-36)

*2026-10-27. Scope: stage 1 (single-target dilution) and stage 2 (single-target mixture).
Read from the PDFs in `~/Downloads/sample prepration/`; nothing below is from memory (SPEC 10.4).*

Our problems (val_s1_v2, val_s2_v2, E2 s1/s2): one target, demand 1, droplet volume 1..M,
M 3–10, F 2 (s1) or 3–6 (s2), exact targets with M-smooth D up to 256, or tolerant targets
(ε 1/200 or 1/1000) with any denominator. Every reservoir is pure. Costs 64/8/1, mix 1, waste 1.

## The candidates

| Method (paper) | Solves | Inputs and assumptions | Objective | Solver? | Check we can reproduce |
|---|---|---|---|---|---|
| **DFM** (Kundu et al., TCAD 2022, §III-C) | dilution, 1 target, MEDA k:l | x:y, M, ε. Transform ratio until N = x+y factors into ≤ M (Alg. 1, least-cardinality factor set); mixed radix tree, greedy factor order; each node passes 1 unit up | fewest mixes, then sample | no | Fig. 7: 71:49, M 6 → order {5,4,6}, nm 3, ns 6, nw 7 (order {4,5,6}: ns 8). Example 2: 70:49, ε 0.005 → 71:49. Poddar Table III "FacDA" column = DFM's ns (e.g. 63/64 at 4X: nm 3, ns 9) |
| **FacDA** (same paper, §III-D) | dilution, 1 target, MEDA k:l | DFM's skeleton; any node may share volume with any ancestor | min sample units | SMT (Z3) | Fig. 8(e), Examples 5–7: 71:49, M 6 → ns 4, nw 3, nm 3, every variable given |
| TVODA (same paper, §IV) | dilution, 1 target, given total volume Vt | forest of skeleton trees, Vt may exceed M | min sample, then mixes | SMT with optimisation | Fig. 9 (71:49, Vt 13) |
| **DFMM** (Kundu & Roy, DSD 2022, §IV) | mixture, 1 target, MEDA k:l | r1:…:rt, M; shortest factor list of S = Σr, descending; per level, remainders give reagent units, quotients go down | heuristic tree | no | Fig. 5: 7:7:6, M 5 → nm 3, nr 11, nw 6 |
| STWM (same paper, §V) | mixture, 1 target | DFMM's skeleton, sharing edges, nodes may be pruned | min waste | SMT (Z3) | Fig. 8 (graph only); Fig. 7 averages |
| **MRCM** (Liang et al., ASP-DAC 2019) | mixture and dilution, 1 target, n-input MEDA mixes | concentration vector at precision P, target droplet size, reactant costs W; mixes of ≤ 8 units | min Σ W·reactant use, then mixes, then waste | ILP per node (enumeration suffices) | Fig. 3: ⟨25,31,44⟩, 4X, W = R1 dearest → 3 ops, R 1/2/6, waste 5 |
| **Generic** (Poddar et al., TCAD 2022; DATE 2021 is its short version) | dilution, 1 target, any platform | x:y, ε, platform volume limits (MEDA: mix 2..M, unequal mix/split) | heuristic, few mixes | no | Fig. 6(b): 7/12 → 6/12, 2/12, 7/12, 3 mixes, 2 sample. Table III (MEDA, 4X): 8 targets with nm, ns |
| WSPM (Li et al., TBioCAS 2017) | dilution, 1 target | (m:n) mixes always with buffer; intermediate CFs **rounded** to the 1/2^n grid | min sample, mixes, waste | no | Fig. 10: 80/256 |
| Dong, Chen, Chen (JETTA 2024) | dilution, many targets on an exponential gradient | dyadic targets only (h-bit binary), droplet counts powers of 2, splits that "halve" CF | sample, waste | no | Fig. 10: 848/1024 |
| RASS / cRASS (Dupukuntla et al., APCCAS 2023) | dilution from non-pure stock solutions | stocks of different CF with costs and availability | min stock cost | SMT | — |

## Decision (user, 2026-10-27): implement 5

**DFM, FacDA, DFMM, MRCM, Generic (Poddar).** Three dilution methods (s1), two that also do
mixtures (s2: DFMM, MRCM; both also run on s1). Each has a worked example to test against.

Left out, and why:
- **TVODA**: our targets are at most M units, where one FacDA tree already serves; the rest
  needs an SMT optimiser over a forest.
- **STWM**: its search over the skeleton has one variable per reagent per node; with up to 6
  fluids a plain search without a solver may not finish. Can be added later if wanted.
- **WSPM**: rounds every intermediate concentration to the grid, so its plans do not hit the
  target exactly in an exact engine; the honest result would be "not applicable" almost
  everywhere.
- **Dong**: dyadic targets only, and its split model (a split that halves the concentration)
  is not a physical operation in our engine.
- **RASS**: non-pure stock solutions, which our problems never have.

## What reproduces (tests in `tests/solvers/test_literature.py`)

| Method | Paper check | Result |
|---|---|---|
| DFM | FacDA paper Example 2 (70:49, ε 0.005, M 6 → 71:49 on {6,5,4}); Fig. 7 (order {5,4,6}: nm 3, ns 6, nw 7; {4,5,6}: ns 8) | all match |
| FacDA | Fig. 8(e): 71:49 at M 6, ns 4 | ns 4 matches; our tie-break finds nw 1 where the paper's graph has nw 3 (the paper's graph is also accepted by our constraints) |
| FacDA | Poddar et al. Table III, FacDA column (x/64, M 4): (nm, ns) for 1, 17, 31, 44, 45, 56, 61, 63 | all 8 match |
| DFMM | Fig. 5: 7:7:6, M 5 → nm 3, nr 11, nw 6 | match |
| Generic | Fig. 6(b): 7/12, M 4 → E_cf {6,2,10}, chain 6/12 → 2/12 → 7/12 | exact graph match |
| Generic | Table III, MEDA "Proposed" (x/64, M 4) | 7 of 8 match; 31/64: we find nm 3, ns 3, the paper 4, 4. These match with mixes of at least 2 units; forcing every mix to the paper's 4X target volume matches only 5 |
| MRCM | §3.1 ILP children: S = <2,1,1> → <0,3,22>, <1,0,0>, <0,1,0> (sum 27); S = <1,1,2,1> → sum 11 | match (for the second our tie-break picks another child set with the same sum 11) |
| MRCM | Fig. 3(a), R1 dearest: 3 mixes, R 1/2/6, waste 5 | exact match |
| MRCM | Fig. 3(b), W = <2,1,3>: paper's tree costs 20 | we find a tree at 19 (makes <1,3,0> in two mixes); the paper shows two trees, our search lists all |

## Choices made where the papers leave room

1. **No new dependency.** FacDA's SMT call is an exhaustive search over the same integer
   variables (sample, buffer and sharing volumes per node, each bounded as in III-D). It gives
   the same minimum sample count; ties (left to the solver in the paper) go to the least waste,
   then the first found. Budget 3,000,000 search nodes, then "not applicable". MRCM's per-node
   ILP is an exhaustive search too (budget 2,000,000 candidate children).
2. **n-input mixes become chains of 2-input mixes** (paper log D-29). Our mix count is then
   higher than the paper's nm; both are reported (`own metric` in the results).
3. **Volume.** A method whose last mix is smaller than the target's droplet volume is "not
   applicable" (no silent rescaling). Our targets are reduced ratios: the papers' tables
   sometimes use the unreduced x/64 (Table III's 44/64 is 11/16 for us).
4. **Step limit.** Each instance has a step limit, which binds our planner and the agent too.
   A paper plan longer than it is "not applicable: the plan needs N steps". Leftovers are not
   discarded mid-plan (they go to waste at the end), so this counts only mixes and the output.
5. **FacDA-Transformation** measures each moved ratio against the original target (the paper
   recurses on the moved ratio, which can drift past ε), rounds half up, and stops at the
   instance's `max_denominator`. Ties between least-cardinality factor sets: the one whose DFM
   tree draws least sample, then wastes least. DFM's greedy order breaks ties to the larger
   factor.
6. **DFMM**: which units share a node is not stated (nm, nr, nw do not depend on it); units are
   dealt in fluid order, children last, with one swap where a node would hold a single fluid.
   Ties between shortest factor lists: fewest reagent units, then least waste. No ratio
   transformation is given for mixtures: a ratio-sum with a prime factor above M is "not
   applicable".
7. **Generic**: E_cf is every ⌈√N⌉-th CF from N // 2 down, then up (reproduces {6, 2, 10} at
   N 12); R_cf is the rest in ascending order (Fig. 6(b) omits 1 from R_cf; the algorithm text
   includes it). Loops run sample, buffer, child volume, ascending. Line 1 cites FacDA for
   writing C_t as x/(x+y), so a tolerant target goes through FacDA-Transformation and every
   later mix is exact on that grid (the paper's per-step ε check can add up past ε). The first
   mix must make at least the target droplet volume.
8. **MRCM**: a mixing model has at most two intermediate children (the paper's examples have
   one or two); an intermediate child has a smaller denominator than its parent and no prime
   factor above M; ties between child sets of equal sum go to the cheaper pure draw. The mixer
   limit is the problem's M (the paper's is 8). The target is built exactly; a denominator with
   a prime factor above M is "not applicable" (the paper rounds to a precision P it does not
   set per problem). Trees share no droplets, so the cheapest tree is found by memoised
   recursion over (composition, volume), which gives the same result as listing every tree.
