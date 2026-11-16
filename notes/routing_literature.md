# Routing literature map (L-20, 2026-11-14)

Read from `~/Downloads/routing` (37 PDFs) and `~/Downloads/Logic synthesis` (classics), plus a
web search for gaps. Per paper: what it solves, its model, its scale, its measure. One line each.
"Ours" = M11's router (SPEC 14.8): many droplets at once, in a running assay, 30-100 cells.

## A. The RL line (Duke: Liang, Elfar, Chakrabarty, Ho) — all about *reliability*, not speed

| Paper | Problem | Method | Scale / measure |
|---|---|---|---|
| Liang et al., ICML 2020, "Adaptive droplet routing in DMFBs using DRL" (**missing PDF**; journal version on disk: TODAES 2024 "Dynamic adaptation using DRL", the "successor" file) | one droplet on a DMFB whose electrodes degrade | DQN/PPO, OpenAI Gym env | success rate under degradation; Raspberry Pi timing |
| Liang et al., ICML 2021, "Parallel droplet control in MEDA using MARL" (on disk) | 2-4 droplets at once on MEDA, degrading electrodes, no camera | MARL (PPO/ACER), PettingZoo env | 30 x 60; baselines single-agent RL and Keszocze 2017 exact; equal when healthy, better when degraded |
| Elfar et al., TCAD 2022, "Formal synthesis of adaptive routing for MEDA" (on disk) | one droplet, health-aware | stochastic game, model checking | 30 x 30 takes ~9 s per operation: does not scale |
| Elfar et al., DATE 2022, "Adaptive routing for MEDA via DRL" (on disk) | same, at commercial size | DRL from health feedback | large arrays (aQdrop 41k electrodes); success probability |
| Liang/Elfar et al., TCAD 2023, "DRL for efficient and reliable routing on MEDA" (on disk) | same + fewer cycles | DRL, validated on fabricated chip | fewer clock cycles than a baseline router |
| Jiang, Yang, Xu, Yao, Ho, Yuan, TCAD 2023, "Cooperative MARL for DMFB routing" (on disk) | many droplets, DMFB | VDN, partial field of view | success rate, completion time vs ICML 2021 |
| ISEDA 2025, "Reliable routing by RL" CaCMARL (on disk) | many droplets + cross-contamination | cooperative MARL + curriculum | 47-58% fewer contaminations, same success/steps |
| TCAD 2026, "Routing-based synthesis via RL" (on disk) | DMFB, no modules: mixing by moving | two DRL agents (transport, mix) + curriculum | PCR/IVD/Protein: 12.9-18.5% faster completion than heuristics |
| Dutta et al. 2024, hexagonal DMFB, dueling DQN (on disk) | one droplet, hexagonal grid | dueling DQN | electrode use |
| Micromachines 2023, "DRL routing for erroneous DMFBs" (PMC10649759) (**not on disk**) | faulty DMFB | DRL | — |

**Reading:** RL for routing is motivated by *unknown, changing chip health*. On a healthy chip
the MARL router only matches the exact offline router. No RL paper reports makespan on a full
assay with modules on MEDA at our scale.

## B. Exact / optimal routing of one or two droplets (MEDA physics)

| Paper | Model | Scale |
|---|---|---|
| Keszocze et al., ASP-DAC 2017, "Exact routing for MEDA" (on disk) | SAT; diagonal moves, shape change; minimal completion time | small |
| Roy, Banerjee, Wille, Bhattacharya, TODAES 2020, "Harnessing the granularity of MEDA for routing" (**missing PDF**, DOI 10.1145/3365993) | SAT; diagonal, reshaping, fractional split-and-merge | small |
| Shiro, Nishikawa, Kong, Tomiyama, Yamashita, Roy, IEEE Access 2022, "Shape-dependent velocity routing" (on disk) | ILP; velocity by shape and direction; 10 x 15 | 1 droplet |
| ICEIC 2023 multi-droplet; NEWCAS 2023 fast; ITC-CSCC 2024 separation; ITC-CSCC 2025 x2 (accelerated IP, size-2 + separation); Appl. Sci. 2025 target-guided; ICEIC 2026 distance-guided size-2; Appl. Sci. 2026 guidance + lower bound (all on disk) | the same group's ILP, made faster by guidance constraints, area limits, lower bounds | ~10 x 10; solve rate 33% → 85% (2026) |
| Nakamura, Yamashita, Tomiyama, Gupta, VLSID 2026, "Polynomial-time routing via Dijkstra" (on disk) | weighted graph so Dijkstra = optimal ILP answer; 2 x 1 droplet, split, morph, shape velocity | up to 100 x 100, 5-30% dead cells; 100 x 100 takes ~3,000 s and GBs of memory; two droplets only |
| Shiro et al., Biosensors 2022, "Minimise chip size in routing" (on disk) | routing that allows splitting, smaller chip | 70.8% smaller chip |
| Eenhakkattu Mana, ..., Ho, Chakrabarty, TCAD 2025, "Calibrated droplet-shape morphing" (on disk) | CFD-validated morphing, routing that uses it | bioassay benchmarks: morphing cuts completion time |

**Reading:** this line (the Ritsumeikan group, where the Dijkstra paper comes from) is about
*one droplet's physics, solved exactly*. It does not handle many droplets competing in time.

## C. Many droplets at once (heuristic, DMFB and MEDA)

| Paper | Method | Notes |
|---|---|---|
| Böhringer, TCAD 2006 (**missing**) | prioritised A* in space-time | the ancestor of ours |
| Su, Hwang, Chakrabarty, ICCAD 2006 (on disk) | first systematic router, fewest cells | DMFB |
| Cho & Pan, TCAD 2008 (**missing**) | one droplet at a time, others frozen; colouring-inspired order | 35x better routability than prioritised A* on hard benchmarks |
| Yuh, Yang, Chang, TCAD 2008, BioRoute (**missing**) | network flow, global then detailed | DMFB |
| Xu & Chakrabarty, DAC 2007 (on disk) | routability estimate inside synthesis | routing-aware placement |
| Maftei, Pop, Madsen, CASES 2010 (on disk) | routing-based synthesis: mix while moving, no modules | smaller chips, faster |
| Chakraborty & Chakraborty, ISVLSI 2019; Howladar et al., JETC 2019 (on disk) | MEDA, homogeneous droplets; snooping index, zone compaction | latest/average arrival, cells |
| Howladar et al., TVLSI 2021 (on disk) | MEDA, cross-contamination-aware | contamination count |
| IETE 2023, capacity-based (on disk) | MEDA, capacity graph + reshaping | -20% latest arrival |
| APCCAS 2025, FCDR (on disk) | MEDA, 3-D flow graph, grow layers until feasible | minimum time steps, compact |
| Lu et al., TODAES 2018 + ASP-DAC 2018 (**missing**) | active matrix: flexible / multi-level routing with split and reshape | -29% latest arrival |
| Manicka et al., Digital Discovery 2023 (on disk) | prioritised 3-D A*, longest first, rerun after each merge | DNA storage, many droplets |
| SwarmRouter, TCAD 2026 (on disk) | hundreds-thousands of droplets on active matrix; spacing + priority preprocessing, path preselection, timed checkpoints | conflict-free, deadlock-free, ACXEL AM16K |
| TDMC, Lab Chip 2026 (**missing**, DOI 10.1039/D6LC00147E) | adaptive-topology A*, conflict resolution, camera feedback | real chip |

## D. Synthesis with routing on MEDA (our whole pipeline)

| Paper | Notes |
|---|---|
| Li et al., DAC 2016 / TBioCAS 2017, "Droplet size-aware HLS for MEDA" (DAC version on disk as "baseline for physics eqns") | schedule + place + route; the velocity model validated on a fabricated chip |
| Kundu et al., RLPM 2023 + TIST/TODAES 2025 (on disk) | RL placement + CAMR priority router; smallest chip |
| Mandal et al., TODAES 2025, online synthesis, virtual topology (on disk) | online placer (our executor's model) |
| Shiro et al. 2024 / 2025, routing with washing droplets (on disk) | residue blocks cells; wash and route together (M12) |

## E. Outside biochips (method, not domain)

Multi-agent path finding (MAPF): prioritised planning (ours), conflict-based search (CBS),
priority-based search (PBS), large-neighbourhood search (MAPF-LNS), LaCAM. The web search found
**no paper that applies CBS/PBS/LNS to droplet routing**.

## Gaps this map shows

1. Exact work routes 1-2 droplets; multi-droplet work is heuristic and measured on
   routing-only benchmarks (latest arrival), not on whole assays.
2. RL routing is about chip health; on a healthy chip it only matches the offline router.
3. No routing paper measures what routing does to a sample-prep plan with shared intermediates
   (storage pressure: E-89's 1.3-3.9x storage x time).
4. Modern MAPF solvers are unused in this field.

## F. Droplet physics, deep read (L-21, 2026-11-15)

Read in full for the movement model: what a move, a shape change and a split cost, and how
droplets keep apart. "Cell" = one unit cell of the paper's grid; "w x h" = width x height.

| Paper | Speed model | Shape change | Split / merge | Apart rule | Calibrated? |
|---|---|---|---|---|---|
| Li et al., DAC 2016 (TBioCAS 2017 the journal version, not on disk) | ODE: EWOD force minus shear, drag, contact-line friction, threshold; speed depends on droplet **size** (square droplets 4x4-16x16 microelectrodes) | - | - | - | **Yes**: fabricated 60 x 30 MEDA chip, 5x5-15x15 droplets |
| Keszocze et al., ASP-DAC 2017 | per-droplet speed from Li: a droplet needs `m_d` time steps per move (the fastest droplet = 1); same in every direction | aspect ratio may change | - | spacing between droplets (SAT) | from Li |
| Roy et al., TODAES 2020 | **uniform speed for all droplets** (stated) | reshaping (aspect ratios) | fractional split-and-merge | fluidic constraints | no |
| Elfar et al., TCAD 2022 (formal) | one cycle per action; success probability from electrode health | 8 morph actions (grow/shrink a corner) | - | single droplet | health model from PCB prototypes |
| **Shiro et al., IEEE Access 2022** (and the group's 2023-2026 papers, washing 2025) | **time per move = volume / actuated cells on the leading edge**: horizontal = w steps, vertical = h steps; diagonal = w + h steps (a horizontal plus a vertical) | A x B -> B x A takes A steps (the washing paper: B x A -> A x B takes B, so the old width); 4 anchor choices | (2023-26) | (2023: no diagonal moves at all) | **no**: "for simplicity, we assume" velocity proportional to actuated cells / volume, citing Li 2017 |
| Nakamura, Yamashita, Tomiyama, Gupta, VLSID 2026 (Dijkstra) | 1x1: 1 unit any of 8 directions; 2 x 1 (tall): horizontal 1, vertical 2, **diagonal 2**; 1 x 2 the mirror | 2 units | split 2x1 -> two 1x1: 1 unit | two 1x1 pieces keep a set distance; no diagonal move next to a dead cell | no ("easily adjusted to other values") |
| Biosensors 2025 (Hamachiyo, Shiro, Nishikawa, Tomiyama, Yamashita; target-guided) | as Nakamura | as Nakamura | split and merge = one 1x1 move; pieces >= 1 cell apart, adjacent pieces act as one size-2 droplet | as left | no |
| Accelerated routing, 2026 (same group) | as Nakamura | morph at any step | pieces count as two droplets only when >= 2 cells apart | diagonal limits next to dead cells | no |
| Eenhakkattu Mana, ..., Ho, Chakrabarty, TCAD 2025 | 1 clock cycle per move, Chebyshev distance | **stretching calibrated**: level-set CFD + COMSOL + experiments; maximum stretch per droplet size | - | - | **Yes** (morphing only) |

**What this means for us**
- Only two things are calibrated on real chips: Li's size-to-speed model (bigger droplets
  are slower; square droplets) and Eenhakkattu Mana's stretching limits. The shape-and-direction
  rule used by the Ritsumeikan line is an assumption derived from Li, and the group's own
  papers differ on the diagonal (w + h in Shiro 2022 and the washing papers; 2 for a 2 x 1 and
  1 for a 1 x 1 in Nakamura 2026 and later, i.e. the longer side).
- The exact papers route droplets of volume <= 2 (1 x 1, 2 x 1, 1 x 2). Our droplets reach
  12 units (3 x 4); the rule "time per move = extent along the move" generalises (a 3 x 3 droplet
  needs 3 ticks per cell), but nobody has used it at that size.
- Our harness today: 1 tick per cell for every shape and direction, no shape change, no split.
  Under the generalised rule our big droplets travel 2-4x slower than now.
- Missing still: Li TBioCAS 2017 (journal), Lu et al. 2018 (active-matrix reshaping);
  `howladar2018.pdf` is a survey (Howladar, Roy, Rahaman, IETE J. Res. 2018), not Lu.
