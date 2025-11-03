# VRP Solver

CO2-Optimized Vehicle Routing Problem Solver using Adaptive Large Neighborhood Search (ALNS).

## Quick Start with uv

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

1. Install uv:
```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Running the Solver

1. **Run with default settings:**
```bash
uv run vrp_solver.py
```

2. **Run with custom parameters:**
```bash
# Increase iterations
uv run vrp_solver.py --iterations 5000

# Allow time window violations
uv run vrp_solver.py --max-violations 5 --violation-penalty 500

# Run sanity check only
uv run vrp_solver.py --sanity-check

# Custom random seed
uv run vrp_solver.py --seed 42
```

3. **Run sanity check:**
```bash
uv run python run_sanity_check.py
```

### Dependencies

The project automatically manages these dependencies:
- `numpy` - Numerical computations
- `scipy` - Scientific computing (optimization)
- `alns` - Adaptive Large Neighborhood Search algorithm

### Files

- `vrp_solver.py` - Main solver implementation
- `data.txt` - Problem data (distances, demands, time windows)
- `solution.txt` - Output solutions (appended each run)
- `results.txt` - Additional results and analysis
- `run_sanity_check.py` - Sanity check script

### Example Usage

```bash
# Basic optimization run
uv run vrp_solver.py

# High-iteration run with some flexibility on time windows
uv run vrp_solver.py --iterations 10000 --max-violations 3 --violation-penalty 100

# Quick sanity check to verify implementation
uv run vrp_solver.py --sanity-check
```

### Output

The solver outputs:
- Optimized tour sequence
- Total CO2 emissions (target: ≤45 kg)
- Detailed delivery schedule
- Time window compliance
- Solution saved to `solution.txt`

## Algorithm Details

Uses physics-based CO2 emission model considering:
- Vehicle load
- Speed optimization (20-40 km/h)
- Road gradients
- Time window constraints

The ALNS algorithm uses multiple destroy/repair operators:
- **Destroy:** Random, worst removal, Shaw removal
- **Repair:** Greedy, regret-based insertion