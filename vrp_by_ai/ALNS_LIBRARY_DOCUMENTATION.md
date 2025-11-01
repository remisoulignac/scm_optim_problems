# ALNS Library - Comprehensive Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start Guide](#quick-start-guide)
5. [State Class Implementation](#state-class-implementation)
6. [Destroy Operators](#destroy-operators)
7. [Repair Operators](#repair-operators)
8. [Acceptance Criteria](#acceptance-criteria)
9. [Operator Selection Schemes](#operator-selection-schemes)
10. [Stopping Criteria](#stopping-criteria)
11. [ALNS Algorithm Workflow](#alns-algorithm-workflow)
12. [Best Practices for VRP](#best-practices-for-vrp)
13. [Complete VRP Example](#complete-vrp-example)
14. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

### What is ALNS?

**ALNS (Adaptive Large Neighbourhood Search)** is a powerful metaheuristic algorithm for solving difficult combinatorial optimization problems. It was originally proposed by Ropke and Pisinger (2006) for vehicle routing problems.

### Key Characteristics

- **Large Neighbourhood Search (LNS)**: Explores large subsets of the search space systematically
- **Meta-heuristic**: Uses other heuristics (destroy/repair operators) as building blocks
- **Ruin-and-Recreate**: Destroys parts of a solution, then repairs it
- **Adaptive**: Learns which operators work best and uses them more often

### How ALNS Works (Pseudocode)

```
Input: initial solution s
Output: optimized solution s*

s* ← s
repeat until stopping criteria is met:
    Select destroy and repair operator pair (d, r) using selection scheme
    s_candidate ← r(d(s))
    if s_candidate is accepted:
        s ← s_candidate
    if s_candidate has better objective than s*:
        s* ← s_candidate
    Update operator selection scheme
return s*
```

### When to Use ALNS

✓ **Good for:**
- Vehicle Routing Problems (VRP, CVRP, VRPTW)
- Traveling Salesman Problem (TSP)
- Scheduling problems
- Resource allocation problems
- Any problem where you can destroy and rebuild parts of a solution

✗ **Not ideal for:**
- Simple problems solvable by exact methods
- Problems where you can't define meaningful destroy/repair operators
- Real-time applications (ALNS can take time to converge)

---

## 2. Installation

### Basic Installation
```bash
pip install alns
```

### Installation with Optional Dependencies
```bash
# With MABWiser support (for advanced operator selection)
pip install alns[mabwiser]
```

### Required Dependencies
- `numpy` (automatically installed)
- `matplotlib` (for plotting results)

### Verify Installation
```python
import alns
alns.show_versions()
```

---

## 3. Core Concepts

### 3.1 Solution State

Every problem needs a **State** class that:
- Stores the solution representation
- Implements an `objective()` method that returns the cost (lower is better)
- Can be copied (for destroy operations)

```python
from alns import State

class MyState(State):
    def __init__(self, data):
        self.data = data
    
    def objective(self) -> float:
        # Return the cost of this solution
        return calculate_cost(self.data)
    
    def copy(self):
        # Return a deep copy
        return MyState(self.data.copy())
```

**IMPORTANT:** The ALNS library assumes **minimization** problems. If you want to maximize, negate your objective: `return -f(x)`.

### 3.2 Destroy Operators

Destroy operators **remove parts** of a solution, creating an incomplete state.

**Signature:**
```python
def destroy_operator(current_state, rng):
    """
    Args:
        current_state: The current solution state
        rng: numpy random number generator
    
    Returns:
        destroyed_state: State with some elements removed
    """
    destroyed = current_state.copy()  # MUST copy!
    # ... remove elements from destroyed ...
    return destroyed
```

⚠️ **CRITICAL WARNING:** Always **copy** the input state! The ALNS algorithm passes the actual current state, not a copy.

### 3.3 Repair Operators

Repair operators **rebuild** the destroyed parts of a solution.

**Signature:**
```python
def repair_operator(destroyed_state, rng):
    """
    Args:
        destroyed_state: Incomplete solution from destroy operator
        rng: numpy random number generator
    
    Returns:
        repaired_state: Complete solution
    """
    # ... repair the solution ...
    return destroyed_state  # Can modify in-place
```

### 3.4 The ALNS Iteration Cycle

```
Current Solution (s)
        ↓
    DESTROY (d)
        ↓
Destroyed Solution
        ↓
     REPAIR (r)
        ↓
Candidate Solution (s_c)
        ↓
    EVALUATE
        ↓
  Accept/Reject → Update s if accepted
        ↓
   Update Weights (operator selection learns)
        ↓
    Repeat...
```

---

## 4. Quick Start Guide

### Minimal Working Example

```python
from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RandomSelect
from alns.stop import MaxIterations
import numpy.random as rnd

# 1. Define your State class
class MyState:
    def __init__(self, tour):
        self.tour = tour
    
    def objective(self):
        # Calculate and return cost
        return sum(distance[self.tour[i]][self.tour[i+1]] 
                   for i in range(len(self.tour)-1))
    
    def copy(self):
        return MyState(self.tour.copy())

# 2. Define destroy operator
def random_destroy(state, rng):
    destroyed = state.copy()
    n_remove = len(destroyed.tour) // 5  # Remove 20%
    for _ in range(n_remove):
        idx = rng.integers(0, len(destroyed.tour))
        destroyed.tour.pop(idx)
    return destroyed

# 3. Define repair operator
def greedy_repair(destroyed, rng):
    # ... implement repair logic ...
    return destroyed

# 4. Create initial solution
initial = MyState([0, 1, 2, 3, 4, 0])

# 5. Setup and run ALNS
alns = ALNS(rnd.default_rng(seed=42))
alns.add_destroy_operator(random_destroy)
alns.add_repair_operator(greedy_repair)

select = RandomSelect(num_destroy=1, num_repair=1)
accept = HillClimbing()
stop = MaxIterations(1000)

result = alns.iterate(initial, select, accept, stop)

# 6. Get best solution
best = result.best_state
print(f"Best objective: {best.objective()}")
```

---

## 5. State Class Implementation

### 5.1 Basic State (Minimization)

```python
class TourState:
    def __init__(self, tour):
        self.tour = tour
        self._cached_objective = None
    
    def objective(self) -> float:
        """Returns the cost to minimize"""
        if self._cached_objective is None:
            self._cached_objective = self._calculate_cost()
        return self._cached_objective
    
    def _calculate_cost(self):
        # Expensive calculation here
        return sum(...)
    
    def copy(self):
        return TourState(self.tour.copy())
```

**Performance Tip:** Cache expensive objective calculations!

### 5.2 Contextual State (for Advanced Operator Selection)

If using contextual bandits (MABSelector), implement `get_context()`:

```python
from alns import ContextualState

class MyContextualState(ContextualState):
    def objective(self) -> float:
        return self._calculate_cost()
    
    def get_context(self):
        """Return a numpy array describing the state"""
        return np.array([
            len(self.tour),
            self.current_load,
            self.time_elapsed,
            # ... other features ...
        ])
```

### 5.3 State for VRP with Multiple Routes

```python
class VRPState:
    def __init__(self, routes, unassigned=None):
        """
        routes: List[List[int]] - Each inner list is a route
        unassigned: List[int] - Customers not yet assigned
        """
        self.routes = routes
        self.unassigned = unassigned if unassigned else []
    
    def objective(self) -> float:
        return sum(route_cost(route) for route in self.routes)
    
    def copy(self):
        import copy
        return VRPState(
            copy.deepcopy(self.routes), 
            self.unassigned.copy()
        )
    
    def find_route(self, customer):
        """Helper: find which route contains a customer"""
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Customer {customer} not found")
```

---

## 6. Destroy Operators

### 6.1 Random Removal

**Purpose:** Remove random elements (baseline operator)

```python
def random_removal(state, rng):
    """Remove random customers from tour"""
    destroyed = state.copy()  # MUST COPY!
    
    tour = destroyed.tour[1:-1]  # Exclude depot
    n_remove = max(1, int(len(tour) * 0.2))  # Remove 20%
    
    removed = []
    for _ in range(n_remove):
        if len(tour) > 0:
            idx = rng.integers(0, len(tour))
            removed.append(tour.pop(idx))
    
    destroyed.tour = [0] + tour + [0]  # Re-add depot
    return destroyed
```

### 6.2 Worst Removal

**Purpose:** Remove customers that contribute most to the cost

```python
def worst_removal(state, rng):
    """Remove customers with highest cost contribution"""
    destroyed = state.copy()
    n_remove = max(1, int(len(state.tour) * 0.15))
    
    # Calculate cost contribution of each customer
    contributions = []
    for i in range(1, len(state.tour) - 1):
        prev, curr, next = state.tour[i-1:i+2]
        
        # Cost with customer
        old_cost = dist[prev][curr] + dist[curr][next]
        # Cost without customer
        new_cost = dist[prev][next]
        
        contribution = old_cost - new_cost
        contributions.append((curr, contribution))
    
    # Remove worst (highest contribution)
    contributions.sort(key=lambda x: x[1], reverse=True)
    to_remove = [c[0] for c in contributions[:n_remove]]
    
    destroyed.tour = [n for n in state.tour if n not in to_remove]
    return destroyed
```

### 6.3 Shaw Removal (Relatedness)

**Purpose:** Remove similar/related customers together

```python
def shaw_removal(state, rng):
    """Remove geographically close customers"""
    destroyed = state.copy()
    tour = destroyed.tour[1:-1].copy()
    n_remove = max(1, int(len(tour) * 0.2))
    
    # Pick random seed customer
    seed = rng.choice(tour)
    
    # Calculate distances to seed
    distances = [(c, distance[seed][c]) for c in tour if c != seed]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    
    # Remove seed and closest neighbors
    to_remove = [seed] + [d[0] for d in distances[:n_remove-1]]
    
    tour = [c for c in tour if c not in to_remove]
    destroyed.tour = [0] + tour + [0]
    
    return destroyed
```

### 6.4 String/Segment Removal (for VRP)

**Purpose:** Remove consecutive segments from routes

```python
def string_removal(state, rng):
    """Remove consecutive customers from routes"""
    destroyed = state.copy()
    
    max_string_size = 5
    max_routes = min(2, len(destroyed.routes))
    
    # Pick random center customer
    center = rng.integers(1, num_customers)
    
    removed_routes = []
    for neighbor in nearest_neighbors(center):
        if len(removed_routes) >= max_routes:
            break
        
        route = destroyed.find_route(neighbor)
        if route in removed_routes:
            continue
        
        # Remove substring containing neighbor
        idx = route.index(neighbor)
        size = rng.integers(1, min(len(route), max_string_size) + 1)
        start = idx - rng.integers(size)
        
        # Remove consecutive elements
        for _ in range(size):
            if 0 <= start < len(route):
                destroyed.unassigned.append(route.pop(start))
        
        removed_routes.append(route)
    
    # Remove empty routes
    destroyed.routes = [r for r in destroyed.routes if len(r) > 0]
    
    return destroyed
```

### 6.5 Destroy Operator Best Practices

✓ **DO:**
- Always copy the input state first
- Remove 10-30% of the solution (degree of destruction)
- Use randomization (controlled by `rng`)
- Return a valid (though incomplete) state

✗ **DON'T:**
- Modify the input state directly
- Remove everything (leave some structure)
- Use Python's `random` module (use the provided `rng`)
- Return None or invalid states

---

## 7. Repair Operators

### 7.1 Greedy Repair

**Purpose:** Insert elements at positions with lowest cost increase

```python
def greedy_repair(destroyed, rng):
    """Insert each removed element at best position"""
    state = destroyed.copy() if hasattr(destroyed, 'copy') else destroyed
    
    # If destroy returns (state, removed_nodes) tuple
    if isinstance(destroyed, tuple):
        state, removed = destroyed
    else:
        # Determine removed nodes from context
        removed = get_removed_nodes(state)
    
    for node in removed:
        best_cost = float('inf')
        best_position = -1
        
        # Try all insertion positions
        for idx in range(1, len(state.tour)):
            # Test insertion
            test_tour = state.tour[:idx] + [node] + state.tour[idx:]
            cost = calculate_cost(test_tour)
            
            if cost < best_cost:
                best_cost = cost
                best_position = idx
        
        # Insert at best position
        if best_position != -1:
            state.tour.insert(best_position, node)
    
    return state
```

### 7.2 Regret Repair

**Purpose:** Prioritize elements with high "regret" (large difference between best and 2nd-best positions)

```python
def regret_repair(destroyed, rng):
    """Insert based on regret heuristic"""
    if isinstance(destroyed, tuple):
        state, removed = destroyed
    else:
        state, removed = destroyed, []
    
    removed = removed.copy()
    
    while removed:
        max_regret = -float('inf')
        best_node = None
        best_pos = -1
        
        # Calculate regret for each remaining node
        for node in removed:
            costs = []
            
            # Find costs for all positions
            for idx in range(1, len(state.tour)):
                test_tour = state.tour[:idx] + [node] + state.tour[idx:]
                cost = calculate_cost(test_tour)
                costs.append((idx, cost))
            
            costs.sort(key=lambda x: x[1])
            
            # Regret = difference between best and 2nd best
            if len(costs) >= 2:
                regret = costs[1][1] - costs[0][1]
            else:
                regret = 0
            
            if regret > max_regret or best_node is None:
                max_regret = regret
                best_node = node
                best_pos = costs[0][0]
        
        # Insert node with highest regret
        state.tour.insert(best_pos, best_node)
        removed.remove(best_node)
    
    return state
```

### 7.3 Random Repair

**Purpose:** Insert randomly (for diversification)

```python
def random_repair(destroyed, rng):
    """Insert nodes at random positions"""
    if isinstance(destroyed, tuple):
        state, removed = destroyed
    else:
        state, removed = destroyed, []
    
    # Shuffle removed nodes
    rng.shuffle(removed)
    
    for node in removed:
        # Insert at random valid position
        if len(state.tour) > 1:
            idx = rng.integers(1, len(state.tour))
            state.tour.insert(idx, node)
    
    return state
```

### 7.4 Capacity-Aware Repair (for CVRP)

**Purpose:** Respect vehicle capacity constraints

```python
def greedy_repair_with_capacity(state, rng):
    """Insert customers considering capacity"""
    rng.shuffle(state.unassigned)
    
    while state.unassigned:
        customer = state.unassigned.pop()
        
        # Try to insert in existing routes
        best_cost = float('inf')
        best_route = None
        best_idx = -1
        
        for route in state.routes:
            # Check capacity
            if can_insert(customer, route, vehicle_capacity):
                for idx in range(len(route) + 1):
                    cost = insert_cost(customer, route, idx)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = route
                        best_idx = idx
        
        # Insert or create new route
        if best_route is not None:
            best_route.insert(best_idx, customer)
        else:
            # Create new route
            state.routes.append([customer])
    
    return state

def can_insert(customer, route, capacity):
    """Check if customer fits in route"""
    total_demand = sum(demands[c] for c in route) + demands[customer]
    return total_demand <= capacity
```

### 7.5 Repair Operator Best Practices

✓ **DO:**
- Consider problem constraints (capacity, time windows)
- Use problem-specific knowledge
- Mix greedy and random strategies
- Test insertion feasibility

✗ **DON'T:**
- Create infeasible solutions
- Always use pure greedy (too exploitative)
- Ignore the `rng` parameter
- Take too long (repair is called many times)

---

## 8. Acceptance Criteria

Acceptance criteria decide whether to accept or reject a candidate solution.

### 8.1 Hill Climbing (Strict)

**Only accepts improving solutions**

```python
from alns.accept import HillClimbing

accept = HillClimbing()
```

**Pros:** Simple, guaranteed improvement  
**Cons:** Can get stuck in local optima

### 8.2 Simulated Annealing (SA)

**Probabilistic acceptance with decreasing temperature**

```python
from alns.accept import SimulatedAnnealing

# Manual setup
accept = SimulatedAnnealing(
    start_temperature=10000,
    end_temperature=1,
    step=0.95,  # Decay multiplier
    method='exponential'  # or 'linear'
)

# Auto-fitted (recommended)
accept = SimulatedAnnealing.autofit(
    init_obj=initial_solution.objective(),
    worse=0.05,  # Accept 5% worse solutions initially
    accept_prob=0.5,  # with 50% probability
    num_iters=5000,
    method='exponential'
)
```

**Acceptance probability:**
$$P(\text{accept}) = \exp\left(\frac{f(s) - f(s_c)}{T}\right)$$

where $T$ is temperature, $s$ is current, $s_c$ is candidate.

**Pros:** Escapes local optima, well-studied  
**Cons:** Sensitive to temperature schedule

### 8.3 Record-to-Record Travel (RRT)

**Accepts if within threshold of best solution**

```python
from alns.accept import RecordToRecordTravel

# Manual
accept = RecordToRecordTravel(
    start_threshold=100,
    end_threshold=1,
    step=0.1,
    method='linear'  # or 'exponential'
)

# Auto-fitted (recommended for VRP)
accept = RecordToRecordTravel.autofit(
    init_obj=initial_solution.objective(),
    start_gap=0.05,  # 5% gap initially
    end_gap=0.001,   # 0.1% gap finally
    num_iters=3000,
    method='linear'
)
```

**Acceptance condition:**
$$f(s_c) < f(s^*) + T$$

where $s^*$ is best solution, $T$ is threshold.

**Pros:** Good for VRP, less sensitive than SA  
**Cons:** Still requires tuning

### 8.4 Great Deluge (GD)

**Accepts if below a decreasing water level**

```python
from alns.accept import GreatDeluge

accept = GreatDeluge(
    alpha=1.05,  # Initial threshold = alpha * initial_obj
    beta=0.001   # Threshold decay rate
)
```

**Pros:** Simple, effective  
**Cons:** Parameter tuning needed

### 8.5 Late Acceptance Hill Climbing (LAHC)

**Compares against solution from L iterations ago**

```python
from alns.accept import LateAcceptanceHillClimbing

accept = LateAcceptanceHillClimbing(
    lookback_period=100,  # Compare with solution 100 iters ago
    greedy=True,  # Always accept improvements
    better_history=False
)
```

**Pros:** Single parameter, robust  
**Cons:** Memory overhead

### 8.6 Random Accept

**Random acceptance with decreasing probability**

```python
from alns.accept import RandomAccept

accept = RandomAccept(
    start_prob=0.8,  # 80% acceptance initially
    end_prob=0.1,    # 10% acceptance finally
    step=0.001,
    method='linear'
)
```

### 8.7 Always Accept

**Accepts everything (for testing)**

```python
from alns.accept import AlwaysAccept

accept = AlwaysAccept()
```

### 8.8 Choosing Acceptance Criteria

| Criterion | Best For | Tuning Difficulty |
|-----------|----------|-------------------|
| **HillClimbing** | Quick local search | Easy ⭐ |
| **SimulatedAnnealing** | General problems | Hard ⭐⭐⭐ |
| **RecordToRecordTravel** | VRP, routing | Medium ⭐⭐ |
| **GreatDeluge** | General problems | Medium ⭐⭐ |
| **LAHC** | Robust default | Easy ⭐ |

**Recommendation for VRP:** Use `RecordToRecordTravel.autofit()`

---

## 9. Operator Selection Schemes

Selection schemes decide which destroy/repair operator pair to use in each iteration.

### 9.1 Random Selection

**Uniform random selection**

```python
from alns.select import RandomSelect

select = RandomSelect(
    num_destroy=3,  # Number of destroy operators
    num_repair=2    # Number of repair operators
)
```

**Pros:** Simple, no parameters  
**Cons:** Doesn't learn, ignores performance

### 9.2 Roulette Wheel (Adaptive)

**Probability proportional to operator weights (learns)**

```python
from alns.select import RouletteWheel

select = RouletteWheel(
    scores=[25, 5, 1, 0],  # Rewards: [best, better, accept, reject]
    decay=0.8,             # Weight decay (θ ∈ [0,1])
    num_destroy=3,
    num_repair=2
)
```

**How it works:**
1. Each operator starts with weight $\omega_i = 1$
2. After applying operators with outcome $j$, update:
   $$\omega_i \leftarrow \theta \omega_i + (1-\theta) s_j$$
3. Select operators with probability proportional to weights

**Scores explanation:**
- `scores[0]`: Reward when candidate is new global best (e.g., 25)
- `scores[1]`: Reward when candidate is better than current (e.g., 5)
- `scores[2]`: Reward when candidate is accepted but not improving (e.g., 1)
- `scores[3]`: Reward when candidate is rejected (e.g., 0)

**Decay parameter ($\theta$):**
- $\theta = 0$: Only use last outcome (very adaptive, unstable)
- $\theta = 1$: Never update (no learning)
- $\theta = 0.8$: **Recommended default** (good balance)

**Pros:** Adaptive, proven effective  
**Cons:** Requires score tuning

### 9.3 Segmented Roulette Wheel

**Updates weights every N iterations (segments)**

```python
from alns.select import SegmentedRouletteWheel

select = SegmentedRouletteWheel(
    scores=[25, 5, 1, 0],
    decay=0.8,
    seg_length=100,  # Update weights every 100 iterations
    num_destroy=3,
    num_repair=2
)
```

**Pros:** More stable than regular roulette wheel  
**Cons:** Less responsive to changes

### 9.4 Alpha-UCB (Upper Confidence Bound)

**Balances exploitation and exploration**

```python
from alns.select import AlphaUCB

select = AlphaUCB(
    scores=[25, 5, 1, 0],
    alpha=0.05,  # Exploration parameter (typically ≤ 0.1)
    num_destroy=3,
    num_repair=2
)
```

**Selection formula:**
$$\text{action} = \arg\max_{a} \left\{ \bar{r}_a + \alpha \sqrt{\frac{\ln(1+t)}{T_a}} \right\}$$

where:
- $\bar{r}_a$: Average reward of action $a$
- $T_a$: Number of times action $a$ was played
- $\alpha$: Exploration parameter

**Alpha parameter:**
- $\alpha = 0$: Pure exploitation (greedy)
- $\alpha = 1$: Maximum exploration
- $\alpha = 0.05$: **Recommended** (good balance)

**Pros:** Principled exploration/exploitation tradeoff  
**Cons:** Less tested than roulette wheel for ALNS

### 9.5 MABSelector (Advanced)

**Use any multi-armed bandit algorithm from MABWiser**

⚠️ Requires: `pip install alns[mabwiser]`

```python
from alns.select import MABSelector
from mabwiser.mab import MAB, LearningPolicy

select = MABSelector(
    scores=[25, 5, 1, 0],
    num_destroy=3,
    num_repair=2,
    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.1),
    seed=42
)
```

**Available policies:** EpsilonGreedy, UCB1, Softmax, Thompson Sampling, etc.

### 9.6 Operator Coupling

**Restrict which destroy/repair pairs can be used together**

```python
import numpy as np

# Coupling matrix: coupling[i,j] = True if destroy[i] can pair with repair[j]
coupling = np.array([
    [True,  True,  False],  # Destroy 0 can pair with repair 0,1
    [True,  False, True ],  # Destroy 1 can pair with repair 0,2
    [False, True,  True ],  # Destroy 2 can pair with repair 1,2
])

select = RouletteWheel(
    scores=[25, 5, 1, 0],
    decay=0.8,
    num_destroy=3,
    num_repair=3,
    op_coupling=coupling
)
```

### 9.7 Choosing Selection Schemes

| Scheme | Best For | Complexity |
|--------|----------|------------|
| **RandomSelect** | Testing, baseline | Very Easy ⭐ |
| **RouletteWheel** | Most problems | Easy ⭐⭐ |
| **AlphaUCB** | When many operators | Medium ⭐⭐⭐ |
| **SegmentedRouletteWheel** | Stable performance | Easy ⭐⭐ |

**Recommendation:** Start with `RouletteWheel` with default parameters.

---

## 10. Stopping Criteria

### 10.1 Maximum Iterations

```python
from alns.stop import MaxIterations

stop = MaxIterations(5000)
```

**Use when:** You have a fixed computational budget (number of iterations)

### 10.2 Maximum Runtime

```python
from alns.stop import MaxRuntime

stop = MaxRuntime(300)  # 300 seconds = 5 minutes
```

**Use when:** You have a fixed time budget

### 10.3 No Improvement

```python
from alns.stop import NoImprovement

stop = NoImprovement(1000)  # Stop after 1000 iterations without improvement
```

**Use when:** You want to stop automatically when stuck

### 10.4 Combining Criteria

You cannot directly combine criteria, but you can implement a custom criterion:

```python
class CombinedStop:
    def __init__(self, max_iters, max_runtime):
        self.max_iters = max_iters
        self.start_time = time.time()
        self.max_runtime = max_runtime
        self.iteration = 0
    
    def __call__(self, rng, best, current):
        self.iteration += 1
        elapsed = time.time() - self.start_time
        return (self.iteration >= self.max_iters or 
                elapsed >= self.max_runtime)

stop = CombinedStop(max_iters=10000, max_runtime=600)
```

---

## 11. ALNS Algorithm Workflow

### 11.1 Complete Setup

```python
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# 1. Create initial solution
initial_solution = create_initial_solution()

# 2. Initialize ALNS
alns = ALNS(rnd.default_rng(seed=42))

# 3. Add operators
alns.add_destroy_operator(random_destroy, name="Random")
alns.add_destroy_operator(worst_removal, name="Worst")
alns.add_destroy_operator(shaw_removal, name="Shaw")

alns.add_repair_operator(greedy_repair, name="Greedy")
alns.add_repair_operator(regret_repair, name="Regret")

# 4. Configure ALNS
select = RouletteWheel([25, 5, 1, 0], 0.8, num_destroy=3, num_repair=2)

accept = RecordToRecordTravel.autofit(
    init_obj=initial_solution.objective(),
    start_gap=0.05,
    end_gap=0.001,
    num_iters=3000,
    method='linear'
)

stop = MaxIterations(3000)

# 5. Run ALNS
result = alns.iterate(initial_solution, select, accept, stop)

# 6. Extract results
best_solution = result.best_state
print(f"Best objective: {best_solution.objective()}")
```

### 11.2 Callbacks

Monitor progress during the search:

```python
# Callback when new best solution found
@alns.on_best
def on_new_best(state, rng):
    print(f"New best found: {state.objective():.2f}")

# Callback when better solution found
@alns.on_better
def on_better(state, rng):
    print(f"Better solution: {state.objective():.2f}")

# Callback when solution accepted (but not improving)
@alns.on_accept
def on_accept(state, rng):
    pass  # Could log or track

# Callback when solution rejected
@alns.on_reject
def on_reject(state, rng):
    pass  # Could track rejection rate
```

**Example: Progress tracking**

```python
iteration = [0]
best_cost = [float('inf')]

@alns.on_best
def track_best(state, rng):
    iteration[0] += 1
    new_cost = state.objective()
    improvement = best_cost[0] - new_cost
    print(f"Iter {iteration[0]}: New best = {new_cost:.2f} (↓{improvement:.2f})")
    best_cost[0] = new_cost
```

### 11.3 Results Object

```python
result = alns.iterate(initial, select, accept, stop)

# Access best solution
best = result.best_state
objective = best.objective()

# Access statistics
stats = result.statistics
objectives = stats.objectives  # Array of all objective values
runtimes = stats.runtimes      # Array of iteration runtimes
total_time = stats.total_runtime  # Total time in seconds

# Operator performance
destroy_counts = stats.destroy_operator_counts
repair_counts = stats.repair_operator_counts
# Format: {operator_name: [best_count, better_count, accept_count, reject_count]}
```

### 11.4 Plotting Results

```python
import matplotlib.pyplot as plt

# Plot objective progression
_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax, title="ALNS Objective Progress")
plt.show()

# Plot operator performance
fig = plt.figure(figsize=(12, 6))
result.plot_operator_counts(
    fig=fig,
    title="Operator Performance",
    legend=['Best', 'Better', 'Accepted', 'Rejected']
)
plt.show()
```

---

## 12. Best Practices for VRP

### 12.1 Problem Setup

**For CO2-Optimized VRP with Variable Arc Costs:**

```python
# 1. Decouple speed optimization from tour optimization
def get_optimal_arc_cost(from_node, to_node, current_load):
    """Optimize speed for a single arc"""
    from scipy.optimize import minimize_scalar
    
    distance = distance_matrix[from_node][to_node]
    gradient = slope_matrix[from_node][to_node]
    
    def co2_for_speed(speed):
        return co2_model.calculate_co2(distance, speed, current_load, gradient)
    
    result = minimize_scalar(co2_for_speed, bounds=(20, 40), method='bounded')
    return result.fun  # Minimum CO2

# 2. Tour evaluation considers load changes
def evaluate_tour_cost(tour):
    total_co2 = 0
    current_load = total_demand  # Start full
    
    for i in range(len(tour) - 1):
        node_A, node_B = tour[i], tour[i+1]
        
        # Optimize speed for this arc given current load
        arc_co2 = get_optimal_arc_cost(node_A, node_B, current_load)
        total_co2 += arc_co2
        
        # Update load
        if node_B != 0:  # Not depot
            current_load -= demands[node_B]
    
    return total_co2
```

### 12.2 Destroy Operators for VRP

**Use a mix:**
1. **Random removal** (20%): Diversification
2. **Worst removal** (15%): Remove costly arcs
3. **Shaw removal** (20%): Remove geographically close customers

```python
alns.add_destroy_operator(random_destroy, name="Random")
alns.add_destroy_operator(worst_removal, name="Worst")
alns.add_destroy_operator(shaw_removal, name="Shaw")
```

### 12.3 Repair Operators for VRP

**Greedy + Regret combination:**

```python
alns.add_repair_operator(greedy_repair, name="Greedy")
alns.add_repair_operator(regret_repair, name="Regret")
```

Greedy is fast, regret provides better quality.

### 12.4 Recommended Parameters

```python
# Operator selection
select = RouletteWheel(
    scores=[25, 5, 1, 0],  # Heavy reward for improvements
    decay=0.8,             # Standard decay
    num_destroy=3,
    num_repair=2
)

# Acceptance criterion
accept = RecordToRecordTravel.autofit(
    init_obj=initial_solution.objective(),
    start_gap=0.02,  # 2% gap (stricter than 5%)
    end_gap=0,       # No gap at end (pure hill climbing)
    num_iters=num_iterations,
    method='linear'
)

# Stopping
stop = MaxIterations(3000)  # 3000+ for good results
```

### 12.5 Initial Solution

**Multiple strategies, pick best:**

```python
def create_initial_solutions():
    # Nearest neighbor
    tour_nn = nearest_neighbor()
    cost_nn = evaluate_tour_cost(tour_nn)
    
    # Time window aware
    tour_tw = time_window_aware()
    cost_tw = evaluate_tour_cost(tour_tw)
    
    # Random
    tour_rand = random_tour()
    cost_rand = evaluate_tour_cost(tour_rand)
    
    # Pick best
    solutions = [(tour_nn, cost_nn), (tour_tw, cost_tw), (tour_rand, cost_rand)]
    best_tour, best_cost = min(solutions, key=lambda x: x[1])
    
    return TourState(best_tour)
```

### 12.6 Performance Tips

**1. Cache expensive calculations:**
```python
class TourState:
    def __init__(self, tour):
        self.tour = tour
        self._cached_obj = None
    
    def objective(self):
        if self._cached_obj is None:
            self._cached_obj = evaluate_tour_cost(self.tour)
        return self._cached_obj
```

**2. Invalidate cache after modifications:**
```python
def greedy_repair(state, removed):
    # ... insert nodes ...
    state._cached_obj = None  # Invalidate cache
    return state
```

**3. Use delta evaluation (advanced):**
Instead of recalculating entire tour, calculate only the change:
```python
def insert_cost_delta(node, route, position):
    """Cost change of inserting node at position"""
    if position == 0:
        prev = 0
    else:
        prev = route[position-1]
    
    if position == len(route):
        next = 0
    else:
        next = route[position]
    
    # Cost increase
    old_cost = arc_cost(prev, next)
    new_cost = arc_cost(prev, node) + arc_cost(node, next)
    
    return new_cost - old_cost
```

---

## 13. Complete VRP Example

Here's a complete, runnable example for a CO2-optimized VRP:

```python
"""
Complete ALNS example for CO2-optimized Vehicle Routing Problem
"""

import numpy as np
import numpy.random as rnd
from scipy.optimize import minimize_scalar
from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# ============================================================================
# PROBLEM DATA (simplified - replace with your data)
# ============================================================================

NUM_CUSTOMERS = 30
distance_matrix = np.random.rand(NUM_CUSTOMERS + 1, NUM_CUSTOMERS + 1) * 10
slope_matrix = np.random.rand(NUM_CUSTOMERS + 1, NUM_CUSTOMERS + 1) * 5
demands = np.random.rand(NUM_CUSTOMERS + 1) * 50
demands[0] = 0  # Depot has no demand
total_demand = demands.sum()

# CO2 model (simplified)
def calculate_co2(distance_km, speed_kmh, load_kg, slope_pct):
    base = 0.5 + (speed_kmh - 30)**2 / 100
    load_factor = 1 + (load_kg / 1000)
    slope_factor = 1 + (slope_pct / 10)
    return base * load_factor * slope_factor * distance_km

# ============================================================================
# SPEED OPTIMIZATION FOR EACH ARC
# ============================================================================

def get_optimal_arc_cost(from_node, to_node, current_load):
    """Optimize speed for a single arc to minimize CO2"""
    distance = distance_matrix[from_node, to_node]
    slope = slope_matrix[from_node, to_node]
    
    if distance == 0:
        return 0
    
    def co2_objective(speed):
        return calculate_co2(distance, speed, current_load, slope)
    
    result = minimize_scalar(co2_objective, bounds=(20, 40), method='bounded')
    return result.fun

# ============================================================================
# TOUR EVALUATION
# ============================================================================

def evaluate_tour_cost(tour):
    """Calculate total CO2 for a tour"""
    total_co2 = 0
    current_load = total_demand
    
    for i in range(len(tour) - 1):
        node_a = tour[i]
        node_b = tour[i + 1]
        
        arc_co2 = get_optimal_arc_cost(node_a, node_b, current_load)
        total_co2 += arc_co2
        
        if node_b != 0:
            current_load -= demands[node_b]
    
    return total_co2

# ============================================================================
# STATE DEFINITION
# ============================================================================

class TourState(State):
    def __init__(self, tour):
        self.tour = tour
        self._cached_objective = None
    
    def objective(self):
        if self._cached_objective is None:
            self._cached_objective = evaluate_tour_cost(self.tour)
        return self._cached_objective
    
    def copy(self):
        return TourState(self.tour.copy())

# ============================================================================
# DESTROY OPERATORS
# ============================================================================

def random_destroy(state, rng):
    destroyed = state.copy()
    tour = destroyed.tour[1:-1].copy()
    n_remove = max(1, int(len(tour) * 0.2))
    
    removed = []
    for _ in range(n_remove):
        if len(tour) > 0:
            idx = rng.integers(0, len(tour))
            removed.append(tour.pop(idx))
    
    destroyed.tour = [0] + tour + [0]
    return destroyed, removed

def worst_removal(state, rng):
    destroyed = state.copy()
    tour = destroyed.tour[1:-1].copy()
    n_remove = max(1, int(len(tour) * 0.15))
    
    # Calculate cost contributions (simplified)
    contributions = []
    for node in tour:
        # Simplified: just use distance from depot
        cost = distance_matrix[0, node]
        contributions.append((node, cost))
    
    contributions.sort(key=lambda x: x[1], reverse=True)
    removed = [c[0] for c in contributions[:n_remove]]
    
    for node in removed:
        tour.remove(node)
    
    destroyed.tour = [0] + tour + [0]
    return destroyed, removed

def shaw_removal(state, rng):
    destroyed = state.copy()
    tour = destroyed.tour[1:-1].copy()
    n_remove = max(1, int(len(tour) * 0.2))
    
    if len(tour) == 0:
        return destroyed, []
    
    seed = rng.choice(tour)
    distances = [(n, distance_matrix[seed, n]) for n in tour if n != seed]
    distances.sort(key=lambda x: x[1])
    
    removed = [seed] + [d[0] for d in distances[:n_remove-1]]
    
    for node in removed:
        tour.remove(node)
    
    destroyed.tour = [0] + tour + [0]
    return destroyed, removed

# ============================================================================
# REPAIR OPERATORS
# ============================================================================

def greedy_repair(destroyed, rng):
    state, removed = destroyed
    
    for node in removed:
        best_cost = float('inf')
        best_idx = -1
        
        for idx in range(1, len(state.tour)):
            test_tour = state.tour[:idx] + [node] + state.tour[idx:]
            cost = evaluate_tour_cost(test_tour)
            
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        
        if best_idx != -1:
            state.tour.insert(best_idx, node)
    
    state._cached_objective = None
    return state

def regret_repair(destroyed, rng):
    state, removed = destroyed
    removed = removed.copy()
    
    while removed:
        max_regret = -float('inf')
        best_node = None
        best_pos = -1
        
        for node in removed:
            costs = []
            for idx in range(1, len(state.tour)):
                test_tour = state.tour[:idx] + [node] + state.tour[idx:]
                cost = evaluate_tour_cost(test_tour)
                costs.append((idx, cost))
            
            costs.sort(key=lambda x: x[1])
            
            if len(costs) >= 2:
                regret = costs[1][1] - costs[0][1]
            else:
                regret = 0
            
            if regret > max_regret or best_node is None:
                max_regret = regret
                best_node = node
                best_pos = costs[0][0]
        
        state.tour.insert(best_pos, best_node)
        removed.remove(best_node)
    
    state._cached_objective = None
    return state

# ============================================================================
# INITIAL SOLUTION
# ============================================================================

def create_initial_solution():
    """Nearest neighbor heuristic"""
    unvisited = set(range(1, NUM_CUSTOMERS + 1))
    tour = [0]
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    tour.append(0)
    return TourState(tour)

# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_vrp():
    print("="*70)
    print("CO2-OPTIMIZED VRP SOLVER")
    print("="*70)
    
    # Create initial solution
    initial = create_initial_solution()
    print(f"\nInitial solution CO2: {initial.objective():.3f} kg")
    
    # Setup ALNS
    alns = ALNS(rnd.default_rng(seed=42))
    
    alns.add_destroy_operator(random_destroy, name="Random")
    alns.add_destroy_operator(worst_removal, name="Worst")
    alns.add_destroy_operator(shaw_removal, name="Shaw")
    
    alns.add_repair_operator(greedy_repair, name="Greedy")
    alns.add_repair_operator(regret_repair, name="Regret")
    
    # Configure
    select = RouletteWheel([25, 5, 1, 0], 0.8, num_destroy=3, num_repair=2)
    
    accept = RecordToRecordTravel.autofit(
        init_obj=initial.objective(),
        start_gap=0.05,
        end_gap=0.001,
        num_iters=2000,
        method='linear'
    )
    
    stop = MaxIterations(2000)
    
    # Run
    print("\nRunning ALNS optimization...")
    
    @alns.on_best
    def on_best(state, rng):
        print(f"  New best: {state.objective():.3f} kg CO2")
    
    result = alns.iterate(initial, select, accept, stop)
    
    # Results
    best = result.best_state
    print(f"\nFinal solution CO2: {best.objective():.3f} kg")
    print(f"Improvement: {((initial.objective() - best.objective()) / initial.objective() * 100):.2f}%")
    print(f"\nBest tour: {' -> '.join(map(str, best.tour[:10]))} -> ... -> {best.tour[-1]}")
    
    return result

if __name__ == "__main__":
    result = solve_vrp()
```

---

## 14. Troubleshooting

### Problem: ALNS not improving

**Possible causes:**
1. Destroy operators too aggressive (remove too much)
   - **Solution:** Reduce destruction percentage (10-20% is good)
   
2. Repair operators too slow
   - **Solution:** Simplify repair logic, use caching
   
3. Acceptance criterion too strict
   - **Solution:** Use RRT with larger `start_gap` (0.05-0.10)
   
4. Not enough iterations
   - **Solution:** Increase iterations (3000+ for good results)
   
5. Poor operator selection
   - **Solution:** Check operator performance with `result.plot_operator_counts()`

### Problem: Solutions are infeasible

**Possible causes:**
1. Repair operator creates invalid solutions
   - **Solution:** Add constraint checks in repair
   
2. Initial solution is infeasible
   - **Solution:** Validate initial solution before ALNS
   
3. Destroy operator removes critical elements
   - **Solution:** Add safeguards in destroy (e.g., don't remove depot)

### Problem: Too slow

**Possible causes:**
1. Objective evaluation is expensive
   - **Solution:** Cache objective values, use delta evaluation
   
2. Too many operator pairs
   - **Solution:** Reduce number of operators or use operator coupling
   
3. Repair is too complex
   - **Solution:** Use simpler repair (pure greedy instead of regret)

**Speed optimization checklist:**
- ✓ Cache objective values
- ✓ Use numpy for matrix operations
- ✓ Profile code to find bottlenecks
- ✓ Consider numba or cython for hot loops
- ✓ Reduce number of destroy/repair operators

### Problem: "State does not have attribute 'objective'"

**Cause:** State class doesn't inherit from `State` protocol or doesn't implement `objective()`

**Solution:**
```python
from alns import State

class MyState(State):  # or just implement objective()
    def objective(self) -> float:
        return self.calculate_cost()
```

### Problem: Results are non-deterministic

**Cause:** Random seed not set or used incorrectly

**Solution:**
```python
# Set seed when creating ALNS
alns = ALNS(rnd.default_rng(seed=42))

# Don't use Python's random module, use the rng parameter
def destroy(state, rng):  # Use this rng!
    idx = rng.integers(0, len(state.tour))  # ✓ Correct
    # idx = random.randint(0, len(state.tour))  # ✗ Wrong!
```

### Problem: Destroy operator error "must copy the state"

**Cause:** Modifying the current state directly

**Solution:**
```python
def destroy(state, rng):
    destroyed = state.copy()  # MUST copy first!
    # Now modify 'destroyed', not 'state'
    return destroyed
```

---

## Key Takeaways

### Essential Pattern for VRP

```python
# 1. State with cached objective
class TourState:
    def __init__(self, tour):
        self.tour = tour
        self._cached_obj = None
    
    def objective(self):
        if self._cached_obj is None:
            self._cached_obj = evaluate(self.tour)
        return self._cached_obj
    
    def copy(self):
        return TourState(self.tour.copy())

# 2. Destroy (always copy!)
def destroy(state, rng):
    destroyed = state.copy()
    # ... modify destroyed ...
    return destroyed, removed_items

# 3. Repair
def repair(destroyed, rng):
    state, removed = destroyed
    # ... insert removed_items back ...
    state._cached_obj = None  # Invalidate cache
    return state

# 4. Setup and run
alns = ALNS(rnd.default_rng(seed=42))
alns.add_destroy_operator(destroy)
alns.add_repair_operator(repair)

select = RouletteWheel([25, 5, 1, 0], 0.8, num_destroy=1, num_repair=1)
accept = RecordToRecordTravel.autofit(init.objective(), 0.05, 0, 3000)
stop = MaxIterations(3000)

result = alns.iterate(initial, select, accept, stop)
```

### Recommended Parameters for VRP

- **Destroy:** 3 operators (random, worst, shaw), remove 10-25%
- **Repair:** 2 operators (greedy, regret)
- **Selection:** RouletteWheel with scores=[25, 5, 1, 0], decay=0.8
- **Acceptance:** RecordToRecordTravel.autofit(start_gap=0.05, end_gap=0.001)
- **Stopping:** MaxIterations(3000+)
- **Random seed:** Always set for reproducibility

### Performance Checklist

- [ ] Cache objective calculations
- [ ] Use numpy for numerical operations
- [ ] Minimize tour evaluations
- [ ] Profile code to find bottlenecks
- [ ] Validate operators work correctly
- [ ] Monitor with callbacks
- [ ] Plot results to verify convergence

---

## References

1. **Original ALNS Paper:** Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. Transportation science, 40(4), 455-472.

2. **ALNS for VRP:** Pisinger, D., & Røpke, S. (2019). Large neighborhood search. In Handbook of metaheuristics (pp. 99-127). Springer.

3. **Acceptance Criteria Comparison:** Santini, A., Ropke, S., & Hvattum, L. M. (2018). A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic. Journal of Heuristics, 24(5), 783-815.

4. **ALNS Library Documentation:** https://alns.readthedocs.io/

---

**Version:** 1.0  
**Last Updated:** November 2025  
**For:** CO2-Optimized Vehicle Routing Problem with Variable Arc Costs
