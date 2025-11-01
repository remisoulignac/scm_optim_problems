# URGENT FIX: Operator Signature Issue

## Problem Identified

Your code is failing with:
```
TypeError: greedy_repair() missing 1 required positional argument: 'random_state'
```

## Root Cause

The issue is with how destroy and repair operators interact:

**Your destroy operators return:**
```python
return new_state, removed_nodes  # A tuple
```

**ALNS then passes this to repair as:**
```python
repair_operator(destroyed=(new_state, removed_nodes), rng=random_state)
```

**But your repair expects:**
```python
def greedy_repair(destroyed, random_state):
    state, removed_nodes = destroyed  # Try to unpack
```

The problem: ALNS is passing `rng` as a keyword argument, but your function expects it as positional `random_state`.

## ✅ IMMEDIATE FIX

### Option 1: Change Repair Signature (Easiest)

Change all repair operators to use `rng` instead of `random_state`:

```python
def greedy_repair(destroyed, rng):  # Change from random_state to rng
    """
    Re-inserts removed nodes one by one at the best (lowest CO2) position.
    Note: destroyed is a tuple of (state, removed_nodes) from destroy operator.
    """
    state, removed_nodes = destroyed
    
    for node in removed_nodes:
        best_cost = float('inf')
        best_idx = -1
        
        # Try inserting 'node' into every possible position
        for idx in range(1, len(state.tour)):
            test_tour = state.tour[:idx] + [node] + state.tour[idx:]
            test_cost = evaluate_tour_cost(test_tour)
            
            if test_cost < best_cost:
                best_cost = cost
                best_idx = idx
        
        # Insert at best position
        if best_idx != -1:
            state.tour.insert(best_idx, node)
    
    # Clear cached objective
    state._cached_objective = None
    return state


def regret_repair(destroyed, rng):  # Change from random_state to rng
    """
    Re-inserts nodes based on regret heuristic.
    """
    state, removed_nodes = destroyed
    removed_nodes = removed_nodes.copy()
    
    while removed_nodes:
        max_regret = -float('inf')
        best_node = None
        best_position = -1
        
        # Calculate regret for each removed node
        for node in removed_nodes:
            insertion_costs = []
            
            # Try all positions
            for idx in range(1, len(state.tour)):
                test_tour = state.tour[:idx] + [node] + state.tour[idx:]
                cost = evaluate_tour_cost(test_tour)
                insertion_costs.append((idx, cost))
            
            # Sort by cost
            insertion_costs.sort(key=lambda x: x[1])
            
            if len(insertion_costs) >= 2:
                # Regret = difference between best and second-best
                regret = insertion_costs[1][1] - insertion_costs[0][1]
            else:
                regret = 0
            
            if regret > max_regret or best_node is None:
                max_regret = regret
                best_node = node
                best_position = insertion_costs[0][0]
        
        # Insert node with highest regret
        if best_node is not None:
            state.tour.insert(best_position, best_node)
            removed_nodes.remove(best_node)
    
    state._cached_objective = None
    return state
```

**Also change destroy operators** to use `rng`:

```python
def random_destroy(state, rng):  # Change from random_state to rng
    """Removes random customers (20%) from the tour."""
    tour_no_depot = state.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.2))
    
    removed_nodes = []
    for _ in range(n_to_remove):
        if len(tour_no_depot) > 0:
            idx = rng.integers(0, len(tour_no_depot))
            removed_nodes.append(tour_no_depot.pop(idx))
    
    new_state = TourState([0] + tour_no_depot + [0])
    return new_state, removed_nodes


def worst_removal(state, rng):  # Change from random_state to rng
    # ... rest of the code ...
    return new_state, removed_nodes


def shaw_removal(state, rng):  # Change from random_state to rng
    # ... rest of the code ...
    return new_state, removed_nodes
```

### Option 2: Store Removed Nodes in State (More Elegant)

Change your State class:

```python
class TourState(State):
    """
    State representation for ALNS algorithm.
    Represents a complete tour as a list of nodes.
    """
    
    def __init__(self, tour, removed_nodes=None):
        # tour is a list of nodes, e.g. [0, 1, 2, ..., 30, 0]
        self.tour = tour
        self.removed_nodes = removed_nodes or []
        self._cached_objective = None
    
    def objective(self):
        """Returns the cost (CO2 emissions) of this tour."""
        if self._cached_objective is None:
            self._cached_objective = evaluate_tour_cost(self.tour)
        return self._cached_objective
    
    def copy(self):
        """Create a deep copy of this state"""
        return TourState(self.tour.copy(), self.removed_nodes.copy())
```

Then modify destroy operators to return only state:

```python
def random_destroy(state, rng):
    """Removes random customers (20%) from the tour."""
    destroyed = state.copy()
    tour_no_depot = destroyed.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.2))
    
    removed_nodes = []
    for _ in range(n_to_remove):
        if len(tour_no_depot) > 0:
            idx = rng.integers(0, len(tour_no_depot))
            removed_nodes.append(tour_no_depot.pop(idx))
    
    destroyed.tour = [0] + tour_no_depot + [0]
    destroyed.removed_nodes = removed_nodes  # Store in state
    return destroyed  # Return only state, not tuple
```

And repair operators to read from state:

```python
def greedy_repair(destroyed, rng):
    """Re-inserts removed nodes one by one at the best position."""
    removed_nodes = destroyed.removed_nodes
    
    for node in removed_nodes:
        best_cost = float('inf')
        best_idx = -1
        
        for idx in range(1, len(destroyed.tour)):
            test_tour = destroyed.tour[:idx] + [node] + destroyed.tour[idx:]
            test_cost = evaluate_tour_cost(test_tour)
            
            if test_cost < best_cost:
                best_cost = test_cost
                best_idx = idx
        
        if best_idx != -1:
            destroyed.tour.insert(best_idx, node)
    
    destroyed.removed_nodes = []  # Clear after repair
    destroyed._cached_objective = None
    return destroyed
```

## Quick Apply

**For fastest fix, search and replace in your file:**

1. Find: `def random_destroy(state, random_state):`  
   Replace: `def random_destroy(state, rng):`

2. Find: `def worst_removal(state, random_state):`  
   Replace: `def worst_removal(state, rng):`

3. Find: `def shaw_removal(state, random_state):`  
   Replace: `def shaw_removal(state, rng):`

4. Find: `def greedy_repair(destroyed, random_state):`  
   Replace: `def greedy_repair(destroyed, rng):`

5. Find: `def regret_repair(destroyed, random_state):`  
   Replace: `def regret_repair(destroyed, rng):`

## Test Your Fix

After making changes, run:

```bash
python vrp_solver.py
```

You should see the ALNS iteration starting without errors.

## Expected Output After Fix

```
----------------------------------------------------------------------
RUNNING ALNS OPTIMIZATION
----------------------------------------------------------------------

  Max iterations: 2000
  Destroy operators: 3 (random, worst, shaw)
  Repair operators: 2 (greedy, regret)
  Acceptance: Record-to-Record Travel
  Weight decay: 0.8

  Starting optimization...
  ------------------------------------------------------------------
  ✓ New best: 45.234 kg (↓12.345 kg)
  Iter  100: Progress 5.0% | Best = 43.567 kg | Last improve: 23 iters ago | Time: 12.3s
  ...
```

## Additional Issue Found

Your initial solutions are showing **0.000 kg CO2**, which is wrong! This means `evaluate_tour_cost()` is returning 0.

**Debug this by adding:**

```python
# In solve_vrp(), after creating initial solutions:
print(f"\nDEBUG: Initial tour = {initial_tour_nn[:5]}...")
print(f"DEBUG: Distance matrix sample = {distance_matrix[0][1]:.3f}")
print(f"DEBUG: Calling evaluate_tour_cost...")

test_cost = evaluate_tour_cost(initial_tour_nn)
print(f"DEBUG: Result = {test_cost}")

# Also check arc cost function
arc_cost = get_optimal_arc_cost(0, 1, total_demand)
print(f"DEBUG: Arc cost 0->1 = {arc_cost}")
```

The issue might be:
1. Distance matrix is all zeros
2. Speed optimization is failing
3. CO2 calculation is returning 0

**Check your data.txt loading:**

```python
# After load_data()
print(f"DEBUG: Distance matrix shape: {distance_matrix.shape}")
print(f"DEBUG: Distance matrix [0,1]: {distance_matrix[0,1]:.3f}")
print(f"DEBUG: Max distance: {distance_matrix.max():.3f}")
print(f"DEBUG: Slope matrix [0,1]: {slope_matrix[0,1]:.3f}")
```

## Summary

**Two issues to fix:**

1. **Immediate:** Change `random_state` to `rng` in all operator signatures ✅
2. **Critical:** Debug why evaluate_tour_cost returns 0.000 kg CO2 🔍

Fix #1 first, then tackle #2 if the problem persists.
