# 3D Bin Packing — MILP Heuristics & Delayed Column Generation

## Overview
The 3D Bin Packing Problem (3D-BPP) is strongly NP-hard with real applications in warehouse logistics and container loading. This project tackles it in two phases:

1. Heuristics + CPLEX - Three heuristic algorithms (Layer-Based, Peak Filling Slice Push, Space Defragmentation) used as warm-starts for IBM CPLEX
2. Delayed Column Generation (DCG) - Full 2D DCG solver implemented as a stepping stone toward scalable 3D solutions

---

## Repo Structure

```
├── heuristics/         # Layer-based, PFSP, Space Defrag
├── solver/             # main_cplex.py — main MILP solver
├── column_generation/  # 2D DCG solver + visualizer
├── logging/            # CPLEX MIP progress logger
├── data/               # Picklist and SKU dimension data (Excel)
├── reports/            # Mid and end-term review reports + PPTs
└── screenshots/        # Pattern visualization outputs
```

---

## Dependencies

```bash
pip install numpy scipy pulp matplotlib
```

IBM CPLEX Optimization Studio required separately (free Community Edition works for the smaller instances).

---

## Quick Start

```python
from column_generation.dcg_2d import TwoDStockCutting

problem = TwoDStockCutting(
    sheet_length=100,
    sheet_width=80,
    piece_demands={(55, 35): 7, (45, 35): 8, (25, 20): 6}
)

obj, patterns = problem.solve()
problem.print_solution(patterns)
problem.visualize_patterns(patterns)
```

---

## Results

| Item Types | Iterations | Time (s) | Sheets Used |
|---|---|---|---|
| 3 | 8 | 2.3 | 4 |
| 5 | 23 | 8.7 | 11 |
| 7 | 35 | 18.4 | 17 |
| 10 | 51 | 42.1 | 26 |

DCG scales significantly better than direct MILP. All generated patterns validated correct.

