# 3D Bin Packing — MILP Heuristics & Delayed Column Generation

## Project Summary

This repository implements research on the 3D Bin Packing Problem (3D-BPP) using a combination of heuristic warm-starts and mathematical programming. The project has two main directions:

- **Phase 1:** 3D packing heuristics and warm-start solutions for IBM CPLEX
- **Phase 2:** a 2D Delayed Column Generation (DCG) solver designed as a stepping stone toward scalable 3D methods

This codebase is intended for experiments with industrial picklists, SKU dimension data, and solver performance on challenging packing instances.

---

## Repository Structure

```
├── heuristics/         # Layer-based, Peak Filling Slice Push, Space Defragmentation heuristics
├── solver/             # main CPLEX MILP solver and model integration
├── column_generation/  # 2D DCG solver, pricing problem, and pattern visualizer
├── logging/            # CPLEX progress logger and experiment logs
├── data/               # picklist and SKU dimension data (Excel files)
├── reports/            # midterm/final reports, presentations, and analysis
└── screenshots/        # visualization outputs and solver snapshots
```

---

## Dependencies

Install the required Python packages:

```bash
pip install numpy scipy pulp matplotlib
```

For optional data handling and visualization:

```bash
pip install pandas openpyxl
```

### External tools

- IBM CPLEX Optimization Studio is required for MILP solver experiments
- The free Community Edition is sufficient for smaller test cases

---

## Usage

### Run the main MILP solver

```bash
python solver/main_cplex.py
```

### Example: Solve a 2D stock-cutting problem

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

## Key Findings

### Phase 1 — Heuristics

| Heuristic | Typical improvement | Robustness | Notes |
|---|---|---|---|
| Layer-Based | low | weaker on high SKU count | fast but limited accuracy |
| Peak Filling Slice Push | moderate | moderate | useful for initial patterns |
| Space Defragmentation | best of the set | better stability | improved utilization on structured data |

The heuristics were useful for warm starts, but direct CPLEX solving remained stronger on many instances. Solver runtime increased sharply beyond 7 SKU types in the studied cases.

### Phase 2 — 2D Delayed Column Generation

| Item types | Total demand | Iterations | Runtime (s) | Sheets used |
|---|---|---|---|---|
| 3 | 23 | 8 | 2.3 | 4 |
| 5 | 85 | 23 | 8.7 | 11 |
| 7 | 140 | 35 | 18.4 | 17 |
| 10 | 220 | 51 | 42.1 | 26 |

The DCG framework demonstrated better scalability than direct MILP on these test cases. The pricing problem remains the primary computational bottleneck, while the master problem is relatively cheap.

---

## Future Work

- Extend the DCG model to full 3D geometric packing
- Develop heuristic or hybrid pricing solvers to reduce solve time
- Implement branch-and-price for integer 3D solutions
- Validate the 3D DCG approach on real picklist data

---

## References

1. Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization.* Athena Scientific.
2. Gilmore, P. C., & Gomory, R. E. (1961). A linear programming approach to the cutting-stock problem. *Operations Research, 9*(6), 849–859.
3. Lodi, A., Martello, S., & Vigo, D. (2002). Recent advances on two-dimensional bin packing problems. *Discrete Applied Mathematics, 123*(1–3), 379–396.
4. Harrath, Y. (2022). A three-stage layer-based heuristic to solve the 3D bin-packing problem under balancing constraint. *Journal of King Saud University - CIS, 34*(10), 6425–6431.
5. Zhang, Z. et al. (2011). Space Defragmentation Heuristic for 2D and 3D Bin Packing Problems. *IJCAI.*
6. IBM ILOG CPLEX Optimization Studio Documentation (2024). IBM Corporation.
