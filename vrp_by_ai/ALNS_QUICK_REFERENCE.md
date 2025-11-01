# ALNS Quick Reference Guide

## 🚀 5-Minute Quick Start

### 1. Install
```bash
pip install alns scipy numpy
```

### 2. Minimal Example
```python
from alns import ALNS, State
from alns.accept import HillClimbing
from alns.select import RandomSelect
from alns.stop import MaxIterations
import numpy.random as rnd

class MyState(State):
    def __init__(self, tour):
        self.tour = tour
    
    def objective(self):
        return sum(dist[self.tour[i]][self.tour[i+1]] 
                   for i in range(len(self.tour)-1))
    
    def copy(self):
        return MyState(self.tour.copy())

def destroy(state, rng):
    destroyed = state.copy()
    # Remove 20% of elements
    return destroyed

def repair(destroyed, rng):
    # Rebuild solution
    return destroyed

initial = MyState([0, 1, 2, 3, 0])
alns = ALNS(rnd.default_rng(seed=42))
alns.add_destroy_operator(destroy)
alns.add_repair_operator(repair)

result = alns.iterate(
    initial,
    RandomSelect(1, 1),
    HillClimbing(),
    MaxIterations(1000)
)

print(f"Best: {result.best_state.objective()}")
```

---

## 📋 Component Cheat Sheet

### State Class
```python
class State:
    def __init__(self, data):
        self.data = data
        self._cache = None  # Optional caching
    
    def objective(self) -> float:
        """Return cost (lower is better)"""
        if self._cache is None:
            self._cache = calculate_cost(self.data)
        return self._cache
    
    def copy(self):
        """Deep copy of state"""
        return State(self.data.copy())
```

### Destroy Operators
```python
def destroy(state, rng):
    """
    MUST copy state first!
    Remove 10-30% of solution
    Return destroyed state
    """
    destroyed = state.copy()  # ⚠️ REQUIRED
    # ... remove elements ...
    return destroyed
```

### Repair Operators
```python
def repair(destroyed, rng):
    """
    Rebuild the solution
    Return complete state
    """
    # ... insert removed elements ...
    destroyed._cache = None  # Invalidate cache
    return destroyed
```

---

## 🎯 Common Patterns

### Pattern 1: VRP with Remove/Insert
```python
def destroy(state, rng):
    destroyed = state.copy()
    removed = []
    
    # Remove 20%
    n = len(state.tour[1:-1])  # Exclude depot
    for _ in range(n // 5):
        idx = rng.integers(1, len(state.tour)-1)
        removed.append(state.tour.pop(idx))
    
    return destroyed, removed  # Return tuple

def repair(data, rng):
    state, removed = data  # Unpack tuple
    
    for node in removed:
        # Find best position
        best_idx = find_best_position(state, node)
        state.tour.insert(best_idx, node)
    
    return state
```

### Pattern 2: Multi-Route VRP
```python
class VRPState:
    def __init__(self, routes, unassigned=[]):
        self.routes = routes  # List[List[int]]
        self.unassigned = unassigned
    
    def objective(self):
        return sum(route_cost(r) for r in self.routes)

def destroy(state, rng):
    destroyed = state.copy()
    # Move customers from routes to unassigned
    for route in destroyed.routes:
        if rng.random() < 0.2:  # 20% chance
            customer = route.pop(rng.integers(len(route)))
            destroyed.unassigned.append(customer)
    return destroyed

def repair(destroyed, rng):
    # Move unassigned back to routes
    while destroyed.unassigned:
        customer = destroyed.unassigned.pop()
        best_route = find_best_route(destroyed, customer)
        best_route.append(customer)
    return destroyed
```

---

## ⚙️ Configuration Quick Picks

### For VRP/Routing Problems
```python
select = RouletteWheel([25, 5, 1, 0], 0.8, num_destroy=3, num_repair=2)
accept = RecordToRecordTravel.autofit(init.objective(), 0.05, 0.001, 3000)
stop = MaxIterations(3000)
```

### For TSP
```python
select = RouletteWheel([20, 5, 1, 0], 0.8, num_destroy=2, num_repair=2)
accept = SimulatedAnnealing.autofit(init.objective(), 0.10, 0.5, 5000)
stop = MaxIterations(5000)
```

### For Scheduling
```python
select = SegmentedRouletteWheel([25, 5, 1, 0], 0.8, 100, num_destroy=3, num_repair=3)
accept = LateAcceptanceHillClimbing(lookback_period=100)
stop = MaxRuntime(300)  # 5 minutes
```

### Quick Testing
```python
select = RandomSelect(num_destroy=1, num_repair=1)
accept = HillClimbing()
stop = MaxIterations(100)
```

---

## 🎨 Acceptance Criteria Quick Reference

| Criterion | Use Case | Key Parameters |
|-----------|----------|----------------|
| `HillClimbing()` | Testing, strict improvement | None |
| `SimulatedAnnealing` | General problems | start_temp, end_temp, method |
| `RecordToRecordTravel` | **VRP (recommended)** | start_gap=0.05, end_gap=0.001 |
| `GreatDeluge` | General problems | alpha=1.05, beta=0.001 |
| `LateAcceptanceHillClimbing` | Robust default | lookback_period=100 |

**Most Common:**
```python
# Auto-fit RRT (recommended)
accept = RecordToRecordTravel.autofit(
    init_obj=initial.objective(),
    start_gap=0.05,  # 5% tolerance
    end_gap=0.001,   # 0.1% at end
    num_iters=3000
)
```

---

## 🎰 Operator Selection Quick Reference

| Scheme | Use Case | Key Parameters |
|--------|----------|----------------|
| `RandomSelect` | Testing | None |
| `RouletteWheel` | **Most problems (recommended)** | scores=[25,5,1,0], decay=0.8 |
| `AlphaUCB` | Many operators | alpha=0.05 |
| `SegmentedRouletteWheel` | Stability | seg_length=100 |

**Most Common:**
```python
select = RouletteWheel(
    scores=[25, 5, 1, 0],  # [best, better, accept, reject]
    decay=0.8,
    num_destroy=3,
    num_repair=2
)
```

---

## 🛑 Stopping Criteria Quick Reference

```python
# Fixed iterations (most common)
stop = MaxIterations(3000)

# Time budget
stop = MaxRuntime(300)  # 300 seconds

# Convergence-based
stop = NoImprovement(1000)  # Stop after 1000 iters without improvement
```

---

## 📊 Monitoring & Debugging

### Add Callbacks
```python
@alns.on_best
def track_best(state, rng):
    print(f"New best: {state.objective():.2f}")

@alns.on_accept
def track_accept(state, rng):
    # Called every iteration
    pass
```

### Plot Results
```python
result = alns.iterate(...)

# Plot objective progression
result.plot_objectives()

# Plot operator performance
result.plot_operator_counts()
```

### Access Statistics
```python
stats = result.statistics
objectives = stats.objectives  # All objective values
total_time = stats.total_runtime
destroy_counts = stats.destroy_operator_counts
# Format: {name: [best_count, better_count, accept_count, reject_count]}
```

---

## 🐛 Common Errors & Fixes

### Error: "State does not have 'objective'"
```python
# ❌ Wrong
class State:
    pass

# ✅ Correct
class State:
    def objective(self) -> float:
        return self.cost
```

### Error: Destroy operator modifies current state
```python
# ❌ Wrong - modifies input
def destroy(state, rng):
    state.tour.pop()
    return state

# ✅ Correct - copies first
def destroy(state, rng):
    destroyed = state.copy()  # ⚠️ Must copy!
    destroyed.tour.pop()
    return destroyed
```

### Error: Non-deterministic results
```python
# ❌ Wrong - uses Python's random
import random
def destroy(state, rng):
    idx = random.randint(0, 10)

# ✅ Correct - uses provided rng
def destroy(state, rng):
    idx = rng.integers(0, 10)
```

### Error: Solutions not improving
```python
# Check:
# 1. Acceptance too strict?
accept = RecordToRecordTravel.autofit(..., start_gap=0.10)  # Increase gap

# 2. Not enough iterations?
stop = MaxIterations(5000)  # Increase

# 3. Operators too conservative?
# Remove 20-30% in destroy (not just 5%)

# 4. Check operator performance
result.plot_operator_counts()  # See which operators help
```

---

## 💡 Pro Tips

### 1. Always Cache Expensive Objectives
```python
class State:
    def __init__(self, tour):
        self.tour = tour
        self._cache = None
    
    def objective(self):
        if self._cache is None:
            self._cache = expensive_calculation(self.tour)
        return self._cache
```

### 2. Invalidate Cache After Modifications
```python
def repair(destroyed, rng):
    # ... modify state ...
    destroyed._cache = None  # ⚠️ Must invalidate!
    return destroyed
```

### 3. Use Multiple Initial Solutions
```python
solutions = [
    nearest_neighbor(),
    random_solution(),
    greedy_construction()
]
initial = min(solutions, key=lambda s: s.objective())
```

### 4. Tune Destruction Percentage
```python
# Too little: slow exploration
n_remove = len(tour) // 20  # 5% - too conservative

# Just right: balanced
n_remove = len(tour) // 5   # 20% - recommended

# Too much: loses structure
n_remove = len(tour) // 2   # 50% - too aggressive
```

### 5. Mix Operator Types
```python
# Good mix for VRP:
alns.add_destroy_operator(random_destroy)   # Diversification
alns.add_destroy_operator(worst_removal)    # Intensification
alns.add_destroy_operator(shaw_removal)     # Problem-specific

alns.add_repair_operator(greedy_repair)     # Fast
alns.add_repair_operator(regret_repair)     # Quality
```

---

## 📏 Parameter Guidelines

### Iterations
- **Small problems (< 50 nodes):** 1000-2000
- **Medium problems (50-200 nodes):** 3000-5000
- **Large problems (> 200 nodes):** 5000-10000

### Destruction Percentage
- **Light:** 10-15% (careful exploration)
- **Medium:** 15-25% (recommended)
- **Heavy:** 25-35% (aggressive exploration)

### Scores (RouletteWheel)
- **Aggressive:** [50, 10, 1, 0] (heavily reward improvements)
- **Balanced:** [25, 5, 1, 0] (recommended)
- **Conservative:** [10, 5, 2, 1] (also reward exploration)

### Decay (RouletteWheel)
- **Adaptive:** 0.5-0.7 (quickly adapt to performance)
- **Balanced:** 0.8 (recommended default)
- **Stable:** 0.9-0.95 (slow adaptation)

---

## 🎓 Learning Path

1. **Start Here:** Read main documentation sections 1-4
2. **Understand:** Study State class (section 5)
3. **Practice:** Implement destroy operators (section 6)
4. **Practice:** Implement repair operators (section 7)
5. **Configure:** Learn acceptance criteria (section 8)
6. **Configure:** Learn operator selection (section 9)
7. **Build:** Complete VRP example (section 13)
8. **Debug:** Troubleshooting guide (section 14)

---

## 📚 Full Documentation

For complete details, examples, and advanced topics:
- **Main Doc:** `ALNS_LIBRARY_DOCUMENTATION.md` (in this directory)
- **Summary:** `DOCUMENTATION_SUMMARY.md` (in this directory)
- **Online:** https://alns.readthedocs.io/

---

## ⚡ TL;DR - Copy-Paste Template

```python
from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
import numpy.random as rnd

# 1. State
class MyState(State):
    def __init__(self, data):
        self.data = data
        self._cache = None
    
    def objective(self):
        if self._cache is None:
            self._cache = calculate_cost(self.data)
        return self._cache
    
    def copy(self):
        return MyState(self.data.copy())

# 2. Operators
def destroy(state, rng):
    destroyed = state.copy()
    # Remove 20%
    return destroyed

def repair(destroyed, rng):
    # Rebuild
    destroyed._cache = None
    return destroyed

# 3. Run
initial = MyState(create_initial())
alns = ALNS(rnd.default_rng(seed=42))
alns.add_destroy_operator(destroy)
alns.add_repair_operator(repair)

result = alns.iterate(
    initial,
    RouletteWheel([25, 5, 1, 0], 0.8, 1, 1),
    RecordToRecordTravel.autofit(initial.objective(), 0.05, 0, 3000),
    MaxIterations(3000)
)

print(f"Best: {result.best_state.objective()}")
```

**Save this template and modify for your problem!**

---

*Last updated: November 2025*  
*Version: 1.0*
