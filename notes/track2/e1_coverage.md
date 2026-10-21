# E1: coverage map, which published methods apply to which constraint combination

*Track 2, 2026-10-21. A reading exercise: BASELINES.md and PLATFORM_PARAMS.md, plus a fresh
literature search (sources at the end). Items marked **verify** come from abstracts or secondary
sources, not from the paper itself; check them in the PDF before relying on them in writing.*

## 1. The cells

Four constraints, each on or off, all on MEDA (k:l mixing, k + l ≤ M, partial takes):

| Axis | Off | On |
|---|---|---|
| Fluids | dilution (sample + buffer, N = 2) | mixture (N ≥ 3) |
| Targets | single target (ST) | multi-target (MT) |
| Demand | one droplet per target (SD) | several droplets per target (MD) |
| Storage | unbounded | limited |

That gives 16 cells. Tolerance (ε = 0 or ε > 0) and accuracy are **settings within a cell**,
not separate cells: every method below fixes one of them, and the benchmark of a cell uses that
method's own setting.

## 2. Method inventory

### MEDA methods (the real competition; no source code published for any of them)

| Method | Class | Mixing model | Optimises | Accuracy setting | Source |
|---|---|---|---|---|---|
| **WSPM** | dilution, ST | BFS over (concentration, size); size limit Sc | operations, sample | δ grid with rounding (δ = 1/256) | Li et al., ISVLSI 2017; TBioCAS 11(6) 2017 (droplet-size-aware, error-correcting) |
| **FacDA** (DFM is its tree phase) | dilution, ST | factorisation of the ratio sum; **any number of inputs**; w(i,j) units shared to any ancestor; SMT | steps, then sample, waste | ε = 0.001 / 0.005, real ratios | Kundu et al., ASP-DAC 2019; TCAD 41(3) 2021 |
| **TVODA** | dilution, ST, **given output volume** (can exceed M) | same line as FacDA, SMT | volume-aware | as FacDA | same paper as FacDA |
| **MRCM** | mixture, ST | multi-input; multi-unit sharing parent→child | weighted reactant cost | exact | Liang et al., ASP-DAC 2019 |
| **e-MRCM** | mixture, **MT** | MRCM plus lists and queues for sharing across targets; multi-input | weighted reactant cost (geometric 1→64, after CoDOS) | exact, P = 1/512 | Liang et al., TCAD 39(10) 2020 |
| **DFMM / STWM** | mixture, ST | skeleton tree + SMT; 1-unit sharing | waste | exact | Kundu & Roy, DSD 2022 |
| **MTWM** | mixture, **MT** | intra- and inter-skeleton sharing, SMT | waste, reagent | exact | Kundu, ACM TODAES 2023 |
| **Dong splitting trees (SDSA)** | dilution, **MT**, exponential gradients | channel-dispensing splits; multi-input (1:1:1:1) | sample, waste droplets | dyadic only (TargetSize = 2^r, precision h) | Dong, Chen, Chen, JETTA 40(1) 2024 |
| **Robust aliquot ILP** | dilution, **MT** | no splits; differently sized aliquots; no intermediate sharing | waste (zero), robustness to split error | 1/2^p | Poddar, Banerjee, Wille, Bhattacharya, ACM TODAES 26(1) 2020 |
| Saha prime factorisation | dilution, ST and MT | prime-factor mixing tree | operations, reactant | not given in the abstract | Saha, Das, Majumder, Springer 2026 |

### DMFB methods (1:1 mixing; each plan is also a legal MEDA plan)

| Method | Class | Why it matters here | Source |
|---|---|---|---|
| **MTCS / MTR / MTMD** | mixture, MT, **demand** | the closest problem match; only on 1:1 | Kumar et al., ACM TODAES 23(4) 2018 |
| **DDMT** (storage-aware) | dilution, MT, **demand**, **storage** | the only planner with a storage limit | Poddar et al., ACM TODAES 2021 |
| **NFSP** | MT with demand | optimal for its network-flow formulation | Dinh, Yamashita, Ho, ASP-DAC 2014 |
| REMIA / GORMA / WARA / CoDOS / RSMA / MinMix / bit scanning | ST and MT, dilution and mixture | the engine check (tier 2), and what MEDA papers compare against | BASELINES.md §2 |
| EDT (exponential dilution) | dilution, MT, gradients | Dong 2024 compares against it | cited in Dong 2024 |

### Other platforms (cite, do not run)
- Storage-constrained mixing on continuous-flow biochips (SAT based; TCAD 2019, TUM). It says
  the approach could be extended to MEDA, but found no MEDA follow-up.
- Loading-aware, mixing-efficient sample preparation on programmable microfluidic devices
  (2025), a different platform.

## 3. The map

The **Direct** column lists methods built for exactly that cell. The **Adapted** column lists
methods that produce a valid plan for the cell but were designed for a smaller class (labelled
"adapted" in any table).

| # | Cell | Direct (MEDA) | Adapted | Status |
|---|---|---|---|---|
| 1 | dil · ST · SD · ∞ | WSPM, FacDA, Saha; MRCM (dilution is its N = 2 case) | DMFB single-target set (M = 2) | **covered, crowded** |
| 2 | dil · MT · SD · ∞ | Dong (dyadic gradients only), robust aliquot ILP, Saha; e-MRCM and MTWM as their N = 2 case | per-target FacDA/WSPM, merged | **covered** |
| 3 | mix · ST · SD · ∞ | MRCM, DFMM/STWM | e-MRCM at T = 1 | **covered** |
| 4 | mix · MT · SD · ∞ | **e-MRCM**, MTWM | per-target MRCM | **covered** (closest competitor) |
| 5 | dil · ST · MD · ∞ | TVODA (as a total volume, **verify**) | FacDA repeated per droplet; the 1:1 demand methods | **partly covered** |
| 6 | dil · MT · MD · ∞ | none on MEDA (robust aliquot may produce copies, **verify**) | DDMT and NFSP (1:1, valid on MEDA); per-target methods streamed | **frontier** |
| 7 | mix · ST · MD · ∞ | none | MTMD at T = 1 (1:1); MRCM repeated | **frontier** |
| 8 | mix · MT · MD · ∞ | none | MTMD (1:1); e-MRCM repeated per demand | **frontier** |
| 9–16 | any cell · **storage limited** | none on MEDA | DDMT in its own cell (dil · MT · MD · S); every other method only if its plan happens to fit (invalid otherwise, PLATFORM_PARAMS §3) | **frontier** |

So the frontier is precisely:
- **demand > 1 with more than one target, or with a mixture** (cells 6–8);
- **any storage limit** (cells 9–16).

Cell 5 is ambiguous until TVODA is read. It takes an output **volume** rather than a droplet
count, which is our "total-volume target" extension.

### "We won't have any method for the frontier, right?"
**No direct method, but there are two kinds of adapted reference, and both are legitimate:**
1. **DMFB demand and storage methods (MTMD, DDMT, NFSP).** Every 1:1 plan is a legal MEDA plan,
   so their plans are valid references in cells 6–9. They also exist in the exact cells our
   claim is about. Beating them only shows that k:l mixing helps, so they are a floor and not
   the result. They are still the only published planners that *model* demand and storage.
2. **MEDA single-demand methods, repeated or streamed.** Run e-MRCM or FacDA once per needed
   droplet or per experiment, and merge the plans. That is what a practitioner would do today
   without our method. Beating it is the result.

Under storage limits, a method that ignores storage can produce an **invalid** plan. "Fraction
of instances where the baseline's plan fits the storage" is then itself a reported number.

## 4. Several methods in one cell: compare them all
Each heuristic optimises a different objective:
- FacDA: steps first;
- e-MRCM: weighted reactant;
- MTWM: waste;
- robust aliquot: zero waste with no sharing.

One weighted cost would favour whichever matches our weights. So in every covered cell, report
for every method and for the agent:
- the paper's own metric;
- each proxy column (mixes, sample, reagent, buffer, waste; `analysis.dag_metrics` already
  computes these);
- our 64 / 8 / 1 cost.

The claim "competitive with any heuristic" is then read as: the agent is on or near the Pareto
front of the methods in that cell, and best or near-best under each method's own metric when it
is trained or evaluated under that metric's weights. The second half needs costs as an input,
which is deferred. Until then, state it as a limit.

## 5. Mixes with more than two inputs (FacDA, e-MRCM, Dong)

**Our model's actual cost of chaining.** An n-input mix of volumes v1 … vn (sum ≤ M) is the same
droplet as the chain ((v1 + v2) + v3) + … + vn of two-input mixes. In our engine:
- **Fluid and waste are identical.** Every unit taken ends up in the final droplet, and nothing
  is discarded in between. The earlier worry that chaining adds wastage does not hold in our
  model: it adds **n − 2 extra mix operations and nothing else**.
- **Legality holds automatically.** Each prefix volume is between 2 and M, so it satisfies
  min-mix and M, and partial takes from the inputs are unchanged.
- **Two real side effects:**
  - **Intermediate denominators:** a prefix such as (3 S + 2 R)/5 has a new prime factor, the
    prefix volume, which is always ≤ M, so it stays M-smooth. Its denominator can exceed the
    final one, so a per-problem `max_denominator` must leave room. It is per problem now (8.1),
    so this is a setting, not a core change.
  - **Steps:** each chain uses n − 1 steps. The depth step limit already covers it at
    multiple 4, and the witness guard raises the limit otherwise.
- **Cost:** under 64 / 8 / 1 with w_mix = 1, the extra mixes are cheap next to one unit of
  sample (64). The agent is barely penalised for chains.

**Physically**, merging n droplets on MEDA is a sequence of pairwise merges with mixing at the
end. The papers count it as **one** mixing operation; we count n − 1.

**Recommendation (no core change):**
1. **Baseline reproductions** may emit n-input mixes. A translator expands each into the
   pairwise chain (an order that keeps prefix denominators small). The plan then runs through
   our engine and `dag.validate`, so fluid and waste are checked exactly.
2. **A paper-comparable mix count:** `dag_metrics` gains a column that contracts a chain of
   mixes into one when each intermediate is used **whole, once, and immediately** by the next
   mix. That is exactly an n-input mix. Mixes, sample, reagent and waste are then reported on
   the papers' own terms for every producer, agent included.
3. **Rule:** the agent stays at two inputs, and the comparison is fair on fluid and waste, which
   are what the baselines optimise.
4. **Stated limitation:** a mixing-time or makespan comparison would favour n-input methods
   slightly. Under our weights it costs at most w_mix per extra merge.

**Checked in the engine (2026-10-21):** a 3-input mix (3S : 2R : 3B, volume 8 at M = 8) and a
4-input mix (1 : 2 : 2 : 3) built as chains replay with exactly the input fluid, zero waste, and
2 and 3 mixes. Against one n-input mix that is +1 and +2 cost units at 64 / 8 / 1, on plans
costing 213 and 102. Still to do when the translator is built: a unit test, and a check that the
contracted count reproduces FacDA's worked example.

## 6. Claims in our documents this search challenges

| Claim (BASELINES.md) | What the search found | Action |
|---|---|---|
| "No MEDA algorithm takes a demand vector" (§1.2 item 1) | TVODA takes a target **volume** above M, which is demand by volume, for one dilution target. Robust aliquot ILP prepares "multiple copies". Dong's TargetSize is a number of output droplets (a power of 2, for a gradient). | **Narrow the claim:** no MEDA method takes a **per-target demand vector over several targets or over mixtures**. Verify TVODA and robust aliquot in the PDFs. |
| "No RL builds mixing graphs" (§8 item 6) | The 2026 CAE review says Liang et al. "introduced an RL-based framework" for multi-target preparation with MRCM. The reference it cites is the e-MRCM TCAD paper, which BASELINES.md describes as a heuristic (lists and queues). The same review cites Liang's RL droplet-routing papers (ICML 2020), so this is probably a mix-up. | **Verify in the e-MRCM PDF.** If e-MRCM were RL-based, the positioning would change. |
| WSPM is the first MEDA method | Its journal version (TBioCAS 2017) is droplet-size-aware and error-correcting | no change |
| Dong splitting trees: "read, cite, do not implement" | It is the only MEDA multi-target method with sharing for gradients, but it is dyadic only | keep as cited, or use it as a cell-2 reference on dyadic gradients only |

Only Saha 2026 (in the docs already) and the 2026 review are newer than BASELINES.md's sweep. No
2024–2025 MEDA method adds demand or storage.

## 7. What this decides
1. **Benchmarks for "competitive with heuristics":** cells 1–4 (and 5 once TVODA is read).
   Each gets one set per method, in that method's own setting. Priority by coverage and
   closeness: e-MRCM (cell 4) > FacDA (cells 1, 5) > MRCM (cell 3) > WSPM (cell 1) > MTWM
   (cell 4).
2. **Frontier benchmarks:** cells 6–16, with both adapted references from §3.
3. **Build items (no core change):** the n-input translator, the contracted mix count, and per
   method a reproduction of its published table.
4. **Writing:** narrow the demand claim, and resolve the RL question.

## Sources
- [Robust Multi-Target Sample Preparation on MEDA Biochips Obviating Waste Production (ACM TODAES)](https://dl.acm.org/doi/fullHtml/10.1145/3414061)
- [Multi-target Fluid Mixing in MEDA Biochips: Theory and an Attempt toward Waste Minimization (MTWM)](https://dl.acm.org/doi/10.1145/3622785)
- [Multitarget Sample Preparation Using MEDA Biochips (e-MRCM)](https://www.researchgate.net/publication/346845757_Multitarget_Sample_Preparation_Using_MEDA_Biochips)
- [Demand-Driven Single- and Multi-Target Mixture Preparation (MTCS/MTR/MTMD)](https://www.researchgate.net/publication/324415982_Demand-Driven_Single-_and_Multi-Target_Mixture_Preparation_using_Digital_Microfluidic_Biochips)
- [Demand-Driven Multi-Target Sample Preparation on Resource-Constrained DMFBs (DDMT)](https://www.researchgate.net/publication/353572334_Demand-Driven_Multi-Target_Sample_Preparation_on_Resource-Constrained_Digital_Microfluidic_Biochips)
- [Dong, Chen, Chen: Reactant and Waste Minimization on MEDA using Splitting Trees (JETTA 2024)](https://www.eng.auburn.edu/~agrawvd/JETTA/FULL_ISSUE_40-1/P07_40-1_Dong_p87-99.pdf)
- [Mondal et al.: AI-guided Automated Sample Preparation on MEDA, a review (CAE 2026)](https://www.caeaccess.org/archives/volume8/number1/mondal-2026-cae-652921.pdf)
- [FacDA: Factorization Based Dilution (ASP-DAC 2019)](https://dl.acm.org/doi/pdf/10.1145/3287624.3287710)
- [Prime Factorization-Based Mixing Algorithm for MEDA Biochip (Springer 2026)](https://link.springer.com/chapter/10.1007/978-3-032-02790-0_12)
- [MEDA Biochip based Single-Target Fluidic Mixture Preparation with Minimum Wastage (DSD 2022)](https://ieeexplore.ieee.org/document/9996606)
- [Droplet Size-Aware and Error-Correcting Sample Preparation on MEDA (TBioCAS 2017)](https://fds.duke.edu/db/pratt/ECE/faculty/krish/publications/329298)
- [Storage-constrained sample preparation, continuous-flow (TCAD 2019, TUM)](https://www.cda.cit.tum.de/files/eda/2019_tcad_storage_constrainted_sample_preparation.pdf)
- [Loading-Aware Mixing-Efficient Sample Preparation on Programmable Microfluidic Device](https://www.researchgate.net/publication/391951249_Loading-Aware_Mixing-Efficient_Sample_Preparation_on_Programmable_Microfluidic_Device)
- [Algorithms for Producing Linear Dilution Gradient with Digital Microfluidics](https://arxiv.org/pdf/1307.1251)
- [Reinforcement Learning-based Reliable Synthesis of Bioassays on MEDA (ACM TIST 2025), placement, not mixing](https://dl.acm.org/doi/10.1145/3773910)
