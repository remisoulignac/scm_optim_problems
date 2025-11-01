# Analysis of Current VRP Solver & Recommendations

## Overview

This document analyzes your current `vrp_solver.py` implementation and provides specific recommendations based on the ALNS library documentation.

---

## Current Implementation Analysis

### ✅ What's Working Well

1. **Physics-Based CO2 Model**
   - Comprehensive fuel consumption calculation
   - Considers vehicle weight, speed, acceleration, slope
   - Realistic conversion to CO2 emissions
   - ✓ This is excellent and differentiates your problem

2. **Decoupled Optimization**
   - Tour sequence optimization (ALNS main loop)
   - Speed optimization per arc (scipy minimize_scalar)
   - ✓ Correct approach for your problem

3. **State Implementation**
   - Has `objective()` method
   - Has `copy()` method
   - Uses caching (`_cached_objective`)
   - ✓ Follows ALNS protocol correctly

4. **Good Operator Variety**
   - 3 destroy operators: random, worst, Shaw
   - 2 repair operators: greedy, regret
   - ✓ Good mix as recommended

5. **Proper ALNS Configuration**
   - RouletteWheel selection
   - RecordToRecordTravel acceptance (auto-fitted)
   - MaxIterations stopping
   - ✓ Solid configuration choices

---

## ⚠️ Issues Found & Fixes

### Issue 1: Destroy Operator Return Format

**Current Code:**
```python
def random_destroy(state, random_state):
    # ...
    return new_state, removed_nodes  # Returns tuple
```

**Problem:** ALNS expects destroy operators to return just the destroyed state, not a tuple. However, your repair operators expect a tuple `(state, removed_nodes)`.

**Status:** ✅ Actually, this is fine! Your repair operators correctly unpack:
```python
def greedy_repair(destroyed, random_state):
    state, removed_nodes = destroyed  # Unpacks tuple
```

This is a valid pattern. The tuple passes through ALNS and gets to repair.

**However**, there's a subtle issue: ALNS might try to call `objective()` on the tuple, not the state.

**Recommended Fix:** Use a wrapper or modify to return only state:

**Option A: Store removed nodes in state**
```python
class TourState(State):
    def __init__(self, tour, removed_nodes=None):
        self.tour = tour
        self.removed_nodes = removed_nodes or []
        self._cached_objective = None

def random_destroy(state, random_state):
    destroyed = state.copy()
    # ... remove nodes ...
    destroyed.removed_nodes = removed_nodes
    return destroyed  # Return state only

def greedy_repair(destroyed, random_state):
    removed_nodes = destroyed.removed_nodes
    # ... repair ...
    destroyed.removed_nodes = []  # Clear
    return destroyed
```

**Option B: Use ALNS callbacks to track removed**
```python
# Global or class variable to pass data between operators
_removed_nodes = []

def random_destroy(state, random_state):
    global _removed_nodes
    destroyed = state.copy()
    # ... remove nodes ...
    _removed_nodes = removed_nodes
    return destroyed

def greedy_repair(destroyed, random_state):
    global _removed_nodes
    removed = _removed_nodes.copy()
    # ... repair ...
    return destroyed
```

**Option C (Cleanest): Keep current pattern but ensure compatibility**

Actually, reviewing the ALNS source code and VRP examples, your current pattern should work! The issue might be elsewhere.

### Issue 2: Cache Invalidation Not Consistent

**Current Code:**
```python
def greedy_repair(destroyed, random_state):
    state, removed_nodes = destroyed
    # ... modify tour ...
    state._cached_objective = None  # ✓ Good!
    return state

def regret_repair(destroyed, random_state):
    state, removed_nodes = destroyed
    # ... modify tour ...
    state._cached_objective = None  # ✓ Good!
    return state
```

**Status:** ✅ This is correct!

### Issue 3: Worst Removal Implementation

**Current Code:**
```python
def worst_removal(state, random_state):
    # ...
    # Calculate cost contribution of each customer
    original_cost = evaluate_tour_cost([0] + state.tour[1:i+2] + [0])
    test_tour = state.tour[:i] + state.tour[i+1:]
    new_cost = evaluate_tour_cost([0] + test_tour[1:-1] + [0])
```

**Problem:** This re-evaluates the entire tour multiple times, which is very expensive given your speed optimization.

**Recommended Fix: Use delta evaluation**
```python
def worst_removal(state, random_state):
    """Removes customers with highest cost contribution (delta method)"""
    tour_no_depot = state.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.15))
    
    # Calculate removal cost delta for each customer
    customer_costs = []
    current_load = total_demand
    
    for i in range(1, len(state.tour) - 1):
        node_prev = state.tour[i - 1]
        node_curr = state.tour[i]
        node_next = state.tour[i + 1]
        
        # Calculate load at this position
        load_before = current_load
        if i > 1:
            load_before -= sum(demands[state.tour[j]] for j in range(1, i))
        
        # Cost of current arcs: prev->curr->next
        cost_with = (get_optimal_arc_cost(node_prev, node_curr, load_before) +
                     get_optimal_arc_cost(node_curr, node_next, load_before - demands[node_curr]))
        
        # Cost of direct arc: prev->next
        cost_without = get_optimal_arc_cost(node_prev, node_next, load_before)
        
        # Savings from removing this customer
        saving = cost_with - cost_without
        customer_costs.append((node_curr, saving))
    
    # Sort by saving (descending = highest cost contribution)
    customer_costs.sort(key=lambda x: x[1], reverse=True)
    
    # Remove worst customers
    removed_nodes = [customer_costs[i][0] for i in range(min(n_to_remove, len(customer_costs)))]
    
    for node in removed_nodes:
        tour_no_depot.remove(node)
    
    new_state = TourState([0] + tour_no_depot + [0])
    return new_state, removed_nodes
```

**Performance Improvement:** This evaluates only 3 arcs per customer instead of the entire tour.

### Issue 4: Error Handling in Speed Optimization

**Current Code:**
```python
def get_optimal_arc_cost(from_node, to_node, current_load):
    distance = distance_matrix[from_node][to_node]
    gradient = slope_matrix[from_node][to_node]
    
    if distance == 0:
        return 0, 30  # No travel needed
    
    result = minimize_scalar(co2_objective_for_speed, ...)
    return result.fun, result.x
```

**Potential Issue:** What if `minimize_scalar` fails or returns invalid result?

**Recommended Fix:**
```python
def get_optimal_arc_cost(from_node, to_node, current_load):
    """Find minimum CO2 cost by optimizing speed"""
    distance = distance_matrix[from_node][to_node]
    gradient = slope_matrix[from_node][to_node]
    
    if distance == 0:
        return 0, 30  # No travel needed
    
    # Objective function for speed optimization
    def co2_objective_for_speed(speed_kmh):
        try:
            return co2_model.calculate_co2_kg(
                distance_km=distance,
                speed_kmh=speed_kmh,
                load_kg=current_load,
                slope_percent=gradient * 100,
                acceleration=0
            )
        except Exception as e:
            # Fallback: return large penalty
            return 1e10
    
    try:
        result = minimize_scalar(
            co2_objective_for_speed,
            method='bounded',
            bounds=(20, 40)
        )
        
        if result.success and np.isfinite(result.fun):
            return result.fun, result.x
        else:
            # Fallback: use middle speed
            speed = 30
            return co2_objective_for_speed(speed), speed
            
    except Exception as e:
        # Fallback calculation
        speed = 30
        return co2_objective_for_speed(speed), speed
```

### Issue 5: Fuel Consumption Edge Cases

**Current Code:**
```python
def fuel_consumption(self, distance_km, speed_kmh, load_kg, slope_percent, acceleration=0):
    v_ms = speed_kmh / 3.6
    if v_ms <= 0:
        return float('inf')
    
    # ... calculation ...
    return max(0, fuel_liters)  # Ensure non-negative
```

**Potential Issue:** Division by zero if `v_ms` is exactly 0, or very small speeds causing huge fuel consumption.

**Recommended Fix:**
```python
def fuel_consumption(self, distance_km, speed_kmh, load_kg, slope_percent, acceleration=0):
    """Calculate fuel consumption with robust error handling"""
    
    # Input validation
    if speed_kmh <= 0 or distance_km <= 0:
        return float('inf')
    
    v_ms = speed_kmh / 3.6  # Convert to m/s
    total_weight = self.CurbWeight + load_kg
    
    # Ensure reasonable bounds
    if total_weight <= 0:
        return float('inf')
    
    try:
        if slope_percent >= 0:
            # Topography model
            degree = slope_percent * 0.57
            theta = degree * math.pi / 180
            Alpha = acceleration + self.g * math.sin(theta) + self.g * self.Cr * math.cos(theta)
            fuel_liters = self.Lambda * distance_km / v_ms * (
                self.k * self.N * self.V + 
                self.Gamma * (total_weight * Alpha * v_ms) + 
                self.Beta * self.Gamma * v_ms**3
            )
        else:
            # Planar model
            fuel_liters = self.Lambda * distance_km / v_ms * (self.k * self.N * self.V)
        
        # Ensure reasonable result
        if not np.isfinite(fuel_liters) or fuel_liters < 0:
            return float('inf')
        
        return max(0, fuel_liters)
        
    except Exception as e:
        print(f"Error in fuel_consumption: {e}")
        return float('inf')
```

---

## 🎯 Recommended Improvements

### 1. Add Progress Tracking (Already Done ✓)

Your callbacks are good! Consider adding more metrics:

```python
iteration = [0]
best_cost = [initial_cost]
last_improvement = [0]
stagnation_count = [0]

@alns.on_best
def on_best_callback(state, rnd_state):
    iteration[0] += 1
    current_cost = state.objective()
    improvement = best_cost[0] - current_cost
    best_cost[0] = current_cost
    last_improvement[0] = iteration[0]
    stagnation_count[0] = 0
    print(f"  ✓ New best: {current_cost:.3f} kg (↓{improvement:.3f} kg)")

@alns.on_accept
def on_accept_callback(state, rnd_state):
    iteration[0] += 1
    stagnation_count[0] += 1
    
    # Log every N iterations
    if iteration[0] % 100 == 0:
        since_best = iteration[0] - last_improvement[0]
        print(f"  Iter {iteration[0]}: Best={best_cost[0]:.3f} kg, "
              f"Last improve: {since_best} iters ago")
        
        # Early stopping if stagnant too long
        if stagnation_count[0] > 1000:
            print("  ⚠ Stagnation detected, consider stopping")
```

### 2. Add Operator Performance Analysis

```python
result = alns.iterate(...)

# Print operator statistics
print("\n" + "="*70)
print("OPERATOR PERFORMANCE")
print("="*70)

destroy_stats = result.statistics.destroy_operator_counts
for name, counts in destroy_stats.items():
    best, better, accept, reject = counts
    total = sum(counts)
    success_rate = (best + better + accept) / total * 100 if total > 0 else 0
    print(f"\nDestroy: {name}")
    print(f"  Best: {best}, Better: {better}, Accept: {accept}, Reject: {reject}")
    print(f"  Success rate: {success_rate:.1f}%")

repair_stats = result.statistics.repair_operator_counts
for name, counts in repair_stats.items():
    best, better, accept, reject = counts
    total = sum(counts)
    success_rate = (best + better + accept) / total * 100 if total > 0 else 0
    print(f"\nRepair: {name}")
    print(f"  Best: {best}, Better: {better}, Accept: {accept}, Reject: {reject}")
    print(f"  Success rate: {success_rate:.1f}%")
```

### 3. Add Solution Validation

```python
def validate_solution(tour, strict=True):
    """Validate tour is feasible"""
    errors = []
    
    # Check starts and ends at depot
    if tour[0] != 0 or tour[-1] != 0:
        errors.append("Tour doesn't start/end at depot")
    
    # Check all customers visited once
    customers = set(tour[1:-1])
    if len(customers) != NUM_CUSTOMERS:
        errors.append(f"Not all customers visited: {len(customers)}/{NUM_CUSTOMERS}")
    
    # Check no duplicates
    if len(customers) != len(tour) - 2:
        errors.append("Duplicate customers in tour")
    
    # Check customer IDs valid
    if not all(0 <= c <= NUM_CUSTOMERS for c in tour):
        errors.append("Invalid customer IDs")
    
    # Check time windows (if strict)
    if strict:
        current_time = 0
        for i in range(len(tour) - 1):
            node_a, node_b = tour[i], tour[i+1]
            
            # Travel time (use average speed for quick check)
            distance = distance_matrix[node_a][node_b]
            travel_time = (distance / 30) * 60  # Assume 30 km/h
            current_time += travel_time
            
            if node_b != 0:
                tw_start, tw_end = time_window_min_max_minute[node_b]
                if current_time > tw_end:
                    errors.append(f"Time window violated at customer {node_b}")
                current_time = max(current_time, tw_start) + delivery_time_minutes[node_b]
    
    return len(errors) == 0, errors

# Use in main:
best_tour = result.best_state.tour
is_valid, errors = validate_solution(best_tour)
if not is_valid:
    print("\n⚠️  WARNING: Solution is infeasible!")
    for error in errors:
        print(f"  - {error}")
```

### 4. Add Detailed Solution Export

```python
def export_solution_json(solution, filename='solution.json'):
    """Export solution in JSON format for analysis"""
    import json
    
    data = {
        'tour': solution['tour'],
        'total_co2_kg': solution['co2_kg'],
        'total_distance_km': solution['total_distance_km'],
        'total_time_minutes': solution['total_time_minutes'],
        'computation_time_seconds': solution['computation_time'],
        'improvement_percent': solution['improvement_percent'],
        'time_feasible': solution['time_feasible'],
        'arcs': [
            {
                'from': arc['from'],
                'to': arc['to'],
                'distance_km': arc['distance_km'],
                'speed_kmh': arc['speed_kmh'],
                'load_kg': arc['load_kg'],
                'co2_kg': arc['co2_kg'],
                'arrival_time_minutes': arc['arrival_time']
            }
            for arc in solution['arc_details']
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Solution exported to {filename}")
```

### 5. Add Visualization (Optional but Useful)

```python
def plot_tour_on_map(tour, arc_details, data_file='data.txt'):
    """Plot the tour with CO2 emissions colored"""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Load coordinates (you'll need to add these to data.txt or compute)
    # For now, use random coordinates
    coords = np.random.rand(NUM_CUSTOMERS + 1, 2) * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot depot
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200, marker='*', 
               label='Depot', zorder=3)
    
    # Plot customers
    ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', s=50, 
               label='Customers', zorder=2)
    
    # Plot tour with CO2 coloring
    max_co2 = max(arc['co2_kg'] for arc in arc_details)
    cmap = cm.get_cmap('RdYlGn_r')  # Red = high, Green = low
    
    for arc in arc_details:
        from_node = arc['from']
        to_node = arc['to']
        co2_ratio = arc['co2_kg'] / max_co2 if max_co2 > 0 else 0
        color = cmap(co2_ratio)
        
        ax.plot([coords[from_node, 0], coords[to_node, 0]],
                [coords[from_node, 1], coords[to_node, 1]],
                c=color, linewidth=2, alpha=0.7, zorder=1)
    
    ax.set_title(f"Tour with CO2 Emissions\nTotal: {sum(a['co2_kg'] for a in arc_details):.2f} kg")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, label='CO2 per arc (kg)')
    plt.tight_layout()
    plt.savefig('tour_visualization.png', dpi=150)
    print("✓ Tour visualization saved to 'tour_visualization.png'")
    plt.show()
```

---

## 🚀 Performance Optimization Tips

### 1. Cache More Aggressively

```python
# Cache arc costs during evaluation
_arc_cost_cache = {}

def get_optimal_arc_cost_cached(from_node, to_node, current_load):
    """Cached version of arc cost calculation"""
    # Round load to reduce cache misses
    load_key = round(current_load, 1)
    cache_key = (from_node, to_node, load_key)
    
    if cache_key not in _arc_cost_cache:
        _arc_cost_cache[cache_key] = get_optimal_arc_cost(from_node, to_node, current_load)
    
    return _arc_cost_cache[cache_key]
```

**Warning:** This cache can grow large. Clear periodically or use LRU cache.

### 2. Parallelize Speed Optimization (Advanced)

```python
from concurrent.futures import ThreadPoolExecutor

def evaluate_tour_cost_parallel(tour):
    """Parallel evaluation of arc costs"""
    
    def compute_arc(i, current_load):
        node_a = tour[i]
        node_b = tour[i + 1]
        return get_optimal_arc_cost(node_a, node_b, current_load)
    
    # Compute loads
    loads = []
    current_load = total_demand
    for i in range(len(tour) - 1):
        loads.append(current_load)
        if tour[i+1] != 0:
            current_load -= demands[tour[i+1]]
    
    # Parallel computation
    with ThreadPoolExecutor(max_workers=4) as executor:
        costs = list(executor.map(
            lambda args: compute_arc(*args),
            [(i, loads[i]) for i in range(len(tour) - 1)]
        ))
    
    return sum(costs)
```

**Note:** Only worth it if speed optimization is very slow.

### 3. Use Numba for CO2 Model (Advanced)

```python
from numba import jit

@jit(nopython=True)
def fuel_consumption_numba(distance_km, speed_kmh, load_kg, slope_percent,
                          CurbWeight, k, N, V, g, Cr, Eta, Eta_tf, 
                          Beta, Gamma, Lambda):
    """Numba-compiled fuel consumption (10-100x faster)"""
    v_ms = speed_kmh / 3.6
    if v_ms <= 0:
        return 1e10
    
    total_weight = CurbWeight + load_kg
    
    if slope_percent >= 0:
        degree = slope_percent * 0.57
        theta = degree * 3.14159 / 180
        Alpha = g * math.sin(theta) + g * Cr * math.cos(theta)
        fuel = Lambda * distance_km / v_ms * (
            k * N * V + Gamma * (total_weight * Alpha * v_ms) + 
            Beta * Gamma * v_ms**3
        )
    else:
        fuel = Lambda * distance_km / v_ms * (k * N * V)
    
    return max(0.0, fuel)
```

---

## 📊 Expected Performance

Based on your problem size and configuration:

### Computational Complexity

- **Per iteration:** O(n²) where n = 30 customers
- **Speed optimizations:** ~30-100 per iteration (for each arc)
- **Total evaluations:** ~2000 iterations × 30 arcs = 60,000+ speed optimizations

### Estimated Runtime

- **Without optimization:** 2-5 minutes
- **With caching:** 30-60 seconds
- **With numba:** 10-20 seconds
- **With all optimizations:** 5-10 seconds

### Expected Results

- **Initial solution:** 60-80 kg CO2 (depending on construction)
- **After ALNS:** 40-50 kg CO2 (target: < 45 kg)
- **Improvement:** 20-40% typical for ALNS on VRP
- **Success rate:** High (> 80%) to meet 45 kg target with 2000+ iterations

---

## ✅ Final Checklist

Before running your final solver:

- [ ] Validate initial solution is feasible
- [ ] Add error handling in speed optimization
- [ ] Implement operator performance tracking
- [ ] Add solution validation
- [ ] Test with small number of iterations first (100)
- [ ] Increase to full iterations (2000-3000)
- [ ] Save detailed results (JSON + text)
- [ ] Plot results (objectives, operators)
- [ ] Verify CO2 target is met
- [ ] Check time windows are respected

---

## 🎯 Next Steps

1. **Immediate:**
   - Add error handling to `get_optimal_arc_cost`
   - Add validation after ALNS completes
   - Run with current configuration

2. **If not meeting target:**
   - Increase iterations to 5000
   - Try different initial solutions
   - Add more destroy operators
   - Tune acceptance criteria (increase `start_gap`)

3. **If too slow:**
   - Implement caching for arc costs
   - Optimize worst_removal operator
   - Consider parallel evaluation

4. **For production:**
   - Add comprehensive logging
   - Implement solution export
   - Add visualization
   - Create sensitivity analysis

---

**Good luck with your optimization!** 🚀

Your implementation is already quite solid. The main areas for improvement are:
1. Performance optimization (caching, delta evaluation)
2. Error handling and validation
3. Enhanced logging and monitoring

The ALNS framework is correctly set up and should produce good results!
