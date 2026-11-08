# Literature coverage audit: which papers we have not compared against

*2026-11-08 (L-9). Every PDF in `~/Downloads/sample prepration/` read (text extracted, abstract,
problem, experiments and figure/table captions), plus the MEDA sample-preparation papers they
cite. Nothing from memory (SPEC 10.4). User: "we cover and test against all of them".*

The folder holds 25 PDFs; 4 are duplicates (same file or same paper), so **21 distinct papers**.
They describe **about 25 methods**, of which **7 are built and benchmarked**.

## 1. Built and benchmarked (7)

| Method | Paper (file) | Checked against the paper |
|---|---|---|
| DFM, FacDA | Kundu et al., TCAD 2022, "Mixing Models as Integer Factorization" (2021_debraj kundu_Mixing_Models...) | Example 2, Fig. 7, Fig. 8(e); Poddar Table III FacDA column, 8/8 |
| Generic | Poddar et al., TCAD 2022 + DATE 2021 short version (A_Generic..., _2021_sudip roy_Generic...) | Fig. 6(b); Table III 7/8 |
| DFMM | Kundu & Roy, DSD 2022 (2022_debraj kundu_MEDA_Biochip_based_Single-Target...) | Fig. 5 |
| MRCM | Liang et al., ASP-DAC 2019 | Fig. 3(a) exact |
| e-MRCM | Liang et al., TCAD 2020, "Multi-Target Sample Preparation Using MEDA Biochips" | Fig. 11(c) exact; Table III does not reproduce (L-7) |
| NFSP | Dinh et al., ASP-DAC 2014 (A Network-Flow-Based...) | Fig. 2 exact; tables not rerun (D-71) |

## 2. Not built: MEDA methods (our home ground)

| Pri. | Method | Paper (file) | Problem | Optimises | Solver | What we can check against | Our stage |
|---|---|---|---|---|---|---|---|
| **built (L-10)** | **MTWM** | Kundu & Roy, TODAES 28(6) 2023 (2023_Debraj kundu_multitarget...) | mixture, **many targets**, one droplet each, no demand, storage only estimated afterwards | waste, then reagent | SMT | Fig. 15 (2:11:5, 12:5:1, 5:6:14, M 5); Table 2 protein electrophoresis (M 7); Figs. 12-14 random sets (2-5 targets, 3-7 reagents, M 5) against e-MRCM | s4, s5 |
| **built (L-10)** | **STWM** | Kundu & Roy, DSD 2022 and TODAES 2023 (same two files) | mixture, one target | waste | SMT | DSD Fig. 8; TODAES Fig. 5-6 (1:7:3, M 4), Table 1 real protocols against MRCM, Fig. 10 random (3-9 reagents, M 5) | s2 |
| **built (L-11)** | **Robust split-less** | Poddar, Banerjee, Wille, Bhattacharya, TODAES 26(1) 2020 (Robust Multi-Target... = Robust aliquot ILP, same file) | dilution, **many targets with copies (demand)**, no splits, zero waste, robust to split and dispensing errors | zero waste, robustness, then sample | ILP | Fig. 13, 18, 19 worked examples (accuracy 7); Tables 4-6, 8-9 (accuracy 7-9); Table 7 real gradients | s1, s3 |
| **built (L-14)** | **TVODA** | Kundu et al., TCAD 2022 (same file as FacDA) | dilution, one target of total volume above M (demand by volume) | sample, then mixes | SMT | Fig. 9 (71:49, Vt 13) | s1 with demand |
| **built (L-15)** | **WSPM** (+ incremental) | Li et al., TBioCAS 11(6) 2017 (WSPM.pdf) | dilution, one target (incremental: 2-4 in sequence) | sample, then mixes, waste | BFS | Fig. 10 (80/256), Table IV incremental | s1, s3 (tolerant targets only: it rounds intermediates) |
| 2 | **Dong SDSA** | Dong, Chen, Chen, JETTA 40(1) 2024 | dilution, many targets on exponential gradients, dyadic only | sample, waste | none | Fig. 10 (848/1024); its tables against EDT | s3 (dyadic gradients only) |
| 2 | **Fukui reuse** | Fukui & Yamashita, ISOCC 2024 (2024_kensuke...) | mixture, a second target reusing one leftover droplet of DFMM's graph | reagent | none | 2 pages: Fig. 2-3 example (4:3:5 from <3:0:1>), averages only (ratio sums 12, 16) | s4 ("our reading": the rule is only sketched) |
| out | RASS (hRASS, cRASS, vRASS) | APCCAS 2023 (two copies, different files) | dilution from **impure** stock solutions with volume limits | stock cost | SMT | Fig. 1 (36/64) | none: our reservoirs are pure |

## 3. Not built: 1:1 (DMFB) methods, valid on MEDA as 1:1 mixes

| Pri. | Method | Paper (file) | Problem | Optimises | Use for us |
|---|---|---|---|---|---|
| **built (L-12)** | **MTCS / MTR / MTMD** | Kumar et al., TODAES 23(4) 2018 (MTCS : MTMD... = Demand-Driven Single- and Multitarget Mixture..., same file) | mixture, one target / many / **many with demand** | steps, waste | B4 frontier competitor; B3 |
| **built (L-13)** | **DDMT** | Poddar et al., TODAES 27(1) 2021 (two copies) | dilution, many targets, **demand, k storage cells**, one mixer | steps given storage | B4 frontier competitor; B3 |
| **built (L-13, as MTC)** | De Bruijn / ATSP multi-target | Bhattacharjee et al., 2014 (2014_Debasis...) | dilution, many targets within an error bound, no storage | mix-split steps | B3 |
| 3 | SIMOP | 2021 (2021_subhashis...) | dilution, one target, user-ordered criteria; split errors studied | priority-ordered | B3 |

## 4. Cited by these papers, PDF not on disk

The 1:1 baselines every paper compares against: **REMIA** (most cited), **DMRW**, **IDMA**,
**GORMA**, **WARA**, **CoDOS**, **RMA**, **Min-Mix**, **RSM**, **TWM**, **XNFM**, **MDST**. Also
Liu et al., "Reactant minimization ... with various mixing models" (TCAD; k:l mixing on DMFB),
and Saha et al., ASP-DAC 2019 (the conference version of DFM, so covered). These matter for B3
(NFSP's tables compare against REMIA and DMRW) more than for our MEDA claim. We need the PDFs
to build any of them.

## 5. Out of scope

Flow-based chips (retention-time paper 2023, VOSPA, FloSPA), mixer placement (Kundu 2023,
transport-free placement), the chip itself (Chan 2021), and the 2025 survey (no new method).

## 6. Found on the way: real assay ratios for B5

- Robust split-less, Table 7: qPCR {256, 32, 4}/512, Bradford protein assay {51, 102, 154, 205,
  256}/512, sucrose gradient and others, accuracy 9.
- MTWM, Table 1: real single-target mixtures (Splinkerette PCR, colorimetric protein assay ...);
  Table 2: protein electrophoresis (three targets, M 7).

## 7. Suggested build order

1. MEDA many targets and waste: **MTWM, STWM, robust split-less** (each needs a solver: SMT for
   MTWM/STWM, as FacDA we could replace with exhaustive search on small skeletons; ILP for the
   robust method via scipy, already a dependency).
2. 1:1 with demand and storage: **MTMD, DDMT** (settles the B4 decision).
3. **TVODA, WSPM, Dong, Fukui** (each narrow: volume-only demand, rounding, dyadic only, a
   two-page sketch).
4. Debasis 2014, SIMOP, and the classic 1:1 baselines if their PDFs are added.

Each counts only after it reproduces its paper's worked example or table (SPEC 10.4); where it
cannot, it is "our reading" with the paper's numbers shown beside.

## 8. What this means for "first" (2026-11-08)

"First for many targets with demand" is wrong as stated: MTMD (Kumar et al. 2018) handles
many mixture targets with demand, on 1:1 chips. Among the papers on disk, the narrower claim
holds: no method does many targets **with demand** for **mixtures** with **k:l (MEDA)** mixing.
MTWM and e-MRCM make one droplet per target; robust split-less does demand but only dilution;
MTMD and DDMT do demand but only 1:1. To keep the claim: build MTMD and DDMT and compare on
our problems (as 1:1 mixes on MEDA), and check the cited papers not on disk. Q-125 stays open.
