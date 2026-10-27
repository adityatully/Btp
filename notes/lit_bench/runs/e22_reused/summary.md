Records: `notes/lit_bench/runs/e22_reused/records.jsonl`. Ratio = cost / best planner (cheapest of share, share_look3, share_orders), costs 64/8/1, mix 1, waste 1. `vs best: < / = / >` counts problems where the producer is cheaper / equal / dearer than the best planner. `agent on same` is the agent's mean ratio on the problems that producer accepted (agent failures left out). Units, waste and mixes are the engine's (2-input mixes); `own metric` is the paper's (n-input mixes nm, units drawn, waste nw).

### e2_s1 (17 problems, agent ran on 17, failed 0)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 12/17 | 1.001 | 1.000 | 0 / 11 / 1 | 1.002 | 1.001 (12) | 2.2 | 1.2 | 1.2 | nm 1.2, units 4.2, nw 1.2 |
| facda | 12/17 | 1.001 | 1.000 | 0 / 11 / 1 | 1.002 | 1.001 (12) | 2.2 | 1.2 | 1.2 | nm 1.2, units 4.2, nw 1.2 |
| generic | 17/17 | 1.176 | 1.000 | 0 / 13 / 4 | 1.017 | 1.239 (17) | 2.9 | 1.9 | 1.6 | nm 1.6, units 5.9, nw 1.9 |
| dfmm | 12/17 | 1.001 | 1.000 | 0 / 11 / 1 | 1.002 | 1.001 (12) | 2.2 | 1.2 | 1.2 | nm 1.2, units 4.2, nw 1.2 |
| mrcm | 15/17 | 1.000 | 1.000 | 0 / 14 / 1 | 1.004 | 1.000 (15) | 2.2 | 1.0 | 1.5 | nm 1.3, units 4.5, nw 1.0 |
| share | 17/17 | 1.000 | 1.000 | 0 / 17 / 0 | 1.017 | 1.027 (17) | 2.4 | 1.1 | 1.6 | - |
| share_look3 | 17/17 | 1.000 | 1.000 | 0 / 17 / 0 | 1.017 | 1.027 (17) | 2.4 | 1.1 | 1.6 | - |
| share_orders | 17/17 | 1.000 | 1.000 | 0 / 17 / 0 | 1.017 | 1.027 (17) | 2.4 | 1.1 | 1.6 | - |
| agent | 17/17 | 1.017 | 1.000 | 0 / 10 / 7 | - | 1.045 (17) | - | - | - | - |

### e2_s2 (7 problems, agent ran on 7, failed 0)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 0/7 | 3 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| facda | 0/7 | 3 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| generic | 0/7 | 3 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| dfmm | 5/7 | 1.157 | 1.000 | 0 / 3 / 2 | 1.003 | 1.157 (5) | 1.4 | 1.4 | 2.0 | nm 1.6, units 3.8, nw 1.4 |
| mrcm | 7/7 | 0.897 | 1.000 | 1 / 6 / 0 | 1.057 | 1.018 (7) | 1.3 | 1.4 | 2.0 | nm 2.0, units 4.4, nw 1.4 |
| share | 7/7 | 1.000 | 1.000 | 0 / 7 / 0 | 1.057 | 1.430 (7) | 1.9 | 2.1 | 2.1 | - |
| share_look3 | 7/7 | 1.000 | 1.000 | 0 / 7 / 0 | 1.057 | 1.430 (7) | 1.9 | 2.1 | 2.1 | - |
| share_orders | 7/7 | 1.000 | 1.000 | 0 / 7 / 0 | 1.057 | 1.430 (7) | 1.9 | 2.1 | 2.1 | - |
| agent | 7/7 | 1.057 | 1.000 | 1 / 4 / 2 | - | 1.425 (7) | - | - | - | - |

### litbench_s1_v1 (100 problems, agent not run)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 77/100 | 1.399 | 1.160 | 4 / 22 / 51 | - | 1.000 (6) | 4.6 | 6.4 | 3.7 | nm 2.8, units 9.7, nw 6.4 |
| facda | 77/100 | 0.982 | 1.000 | 29 / 19 / 29 | - | 1.000 (6) | 3.3 | 4.1 | 3.6 | nm 2.9, units 7.4, nw 4.1 |
| generic | 87/100 | 1.575 | 1.153 | 9 / 18 / 60 | - | 1.000 (7) | 5.2 | 9.5 | 5.6 | nm 4.7, units 13.3, nw 9.5 |
| dfmm | 45/100 | 1.290 | 1.039 | 1 / 17 / 27 | - | 1.000 (6) | 3.7 | 5.5 | 3.1 | nm 2.4, units 8.7, nw 5.5 |
| mrcm | 55/100 | 0.992 | 1.000 | 17 / 28 / 10 | - | 1.000 (7) | 3.1 | 3.3 | 3.0 | nm 2.8, units 7.1, nw 3.3 |
| share | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | 1.000 (7) | 3.8 | 4.7 | 3.3 | - |
| share_look3 | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | 1.000 (7) | 3.8 | 4.7 | 3.3 | - |
| share_orders | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | 1.000 (7) | 3.8 | 4.7 | 3.3 | - |

### litbench_s2_v1 (100 problems, agent not run)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 0/100 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| facda | 0/100 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| generic | 0/100 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| dfmm | 48/100 | 1.093 | 1.000 | 23 / 8 / 17 | - | - (0) | 3.7 | 10.6 | 7.1 | nm 4.0, units 13.9, nw 10.6 |
| mrcm | 41/100 | 0.783 | 0.733 | 29 / 9 / 3 | - | - (0) | 2.3 | 5.9 | 5.8 | nm 4.0, units 9.3, nw 5.9 |
| share | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | - (0) | 4.0 | 14.8 | 8.0 | - |
| share_look3 | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | - (0) | 4.0 | 14.8 | 8.0 | - |
| share_orders | 100/100 | 1.000 | 1.000 | 0 / 100 / 0 | - | - (0) | 4.0 | 14.8 | 8.0 | - |

### paper_s1 (129 problems, agent not run)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 120/129 | 1.424 | 1.033 | 0 / 33 / 87 | - | 1.000 (6) | 5.2 | 4.8 | 3.1 | nm 2.4, units 10.6, nw 4.8 |
| facda | 120/129 | 0.973 | 1.000 | 27 / 54 / 39 | - | 1.000 (6) | 4.0 | 2.7 | 2.9 | nm 2.4, units 8.5, nw 2.7 |
| generic | 128/129 | 1.172 | 1.017 | 7 / 46 / 75 | - | 1.000 (10) | 4.7 | 4.0 | 3.0 | nm 2.6, units 10.0, nw 4.0 |
| dfmm | 116/129 | 1.437 | 1.117 | 0 / 30 / 86 | - | 1.000 (6) | 5.3 | 4.9 | 3.2 | nm 2.4, units 10.6, nw 4.9 |
| mrcm | 128/129 | 0.961 | 1.000 | 45 / 78 / 5 | - | 1.000 (10) | 3.9 | 2.0 | 3.0 | nm 2.9, units 7.9, nw 2.0 |
| share | 129/129 | 1.000 | 1.000 | 0 / 129 / 0 | - | 1.000 (10) | 4.1 | 2.7 | 2.7 | - |
| share_look3 | 129/129 | 1.000 | 1.000 | 0 / 129 / 0 | - | 1.000 (10) | 4.1 | 2.7 | 2.7 | - |
| share_orders | 129/129 | 1.000 | 1.000 | 0 / 129 / 0 | - | 1.000 (10) | 4.1 | 2.7 | 2.7 | - |

### paper_s2 (3 problems, agent not run)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 0/3 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| facda | 0/3 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| generic | 0/3 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2 | - | - | - | - | - | - | - | - | - | - |
| dfmm | 3/3 | 1.500 | 1.154 | 0 / 1 / 2 | - | - (0) | 2.7 | 6.0 | 5.3 | nm 2.3, units 9.0, nw 6.0 |
| mrcm | 3/3 | 0.815 | 0.864 | 2 / 1 / 0 | - | - (0) | 1.3 | 3.3 | 3.7 | nm 2.0, units 6.3, nw 3.3 |
| share | 3/3 | 1.000 | 1.000 | 0 / 3 / 0 | - | - (0) | 1.7 | 6.3 | 4.7 | - |
| share_look3 | 3/3 | 1.000 | 1.000 | 0 / 3 / 0 | - | - (0) | 1.7 | 6.3 | 4.7 | - |
| share_orders | 3/3 | 1.000 | 1.000 | 0 / 3 / 0 | - | - (0) | 1.7 | 6.3 | 4.7 | - |

### val_s1_v2 (40 problems, agent ran on 40, failed 0)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 30/40 | 1.543 | 1.170 | 2 / 8 / 20 | 0.972 | 2.058 (1) | 4.8 | 6.8 | 3.3 | nm 2.8, units 10.1, nw 6.8 |
| facda | 30/40 | 1.095 | 1.005 | 9 / 5 / 16 | 0.972 | 2.043 (1) | 3.8 | 4.6 | 3.5 | nm 3.0, units 7.9, nw 4.6 |
| generic | 34/40 | 1.622 | 1.013 | 3 / 11 / 20 | 0.986 | 1.000 (3) | 4.2 | 7.0 | 3.9 | nm 3.4, units 10.9, nw 7.0 |
| dfmm | 19/40 | 1.731 | 1.321 | 1 / 5 / 13 | 0.980 | - (0) | 5.3 | 7.1 | 3.5 | nm 2.7, units 10.4, nw 7.1 |
| mrcm | 27/40 | 1.006 | 1.000 | 10 / 13 / 4 | 0.984 | 1.000 (2) | 3.1 | 4.1 | 3.6 | nm 3.4, units 8.5, nw 4.1 |
| share | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 0.978 | 1.000 (3) | 3.4 | 4.3 | 3.0 | - |
| share_look3 | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 0.978 | 1.000 (3) | 3.4 | 4.3 | 3.0 | - |
| share_orders | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 0.978 | 1.000 (3) | 3.4 | 4.3 | 3.0 | - |
| agent | 40/40 | 0.978 | 1.000 | 7 / 19 / 14 | - | 1.178 (3) | - | - | - | - |

### val_s2_v2 (40 problems, agent ran on 40, failed 6)

| producer | accepted | ratio mean | ratio median | vs best: < / = / > | agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |
|---|---|---|---|---|---|---|---|---|---|---|
| dfm | 0/40 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| facda | 0/40 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| generic | 0/40 | 3 fluids: a dilution method takes 2; 4 fluids: a dilution method takes 2; 5 flui | - | - | - | - | - | - | - | - | - | - |
| dfmm | 15/40 | 0.974 | 0.948 | 8 / 2 / 5 | 1.035 | - (0) | 3.5 | 9.3 | 6.7 | nm 3.7, units 13.0, nw 9.3 |
| mrcm | 17/40 | 0.658 | 0.571 | 13 / 4 / 0 | 1.024 | - (0) | 1.9 | 5.6 | 5.8 | nm 4.0, units 10.4, nw 5.6 |
| share | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 1.035 | - (0) | 4.3 | 14.6 | 7.8 | - |
| share_look3 | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 1.035 | - (0) | 4.3 | 14.6 | 7.8 | - |
| share_orders | 40/40 | 1.000 | 1.000 | 0 / 40 / 0 | 1.035 | - (0) | 4.3 | 14.6 | 7.8 | - |
| agent | 34/40 | 1.035 | 1.000 | 8 / 12 / 14 | - | - (0) | - | - | - | - |
