"""
CO2-Optimized Vehicle Routing Problem Solver
Uses ALNS (Adaptive Large Neighborhood Search) with speed optimization
"""

import numpy as np
import random
import math
from scipy.optimize import minimize_scalar
from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
import time

# ============================================================================
# FUEL CONSUMPTION MODEL (Physics-based CO2 Emission Model)
# ============================================================================

class CO2Model:
    """
    Physics-based CO2 emission model for a medium-duty delivery truck.
    Uses the provided fuel consumption formula considering:
    - Vehicle load
    - Speed
    - Road slope/gradient
    - Aerodynamics and rolling resistance
    """
    
    def __init__(self):
        # Vehicle Parameters
        self.CurbWeight = 5500   # kg, Curb weight (Medium Duty Truck)
        self.L = 28000           # kg, Maximum Load
        self.k = 0.20            # kJ/rev/L, Engine friction factor
        self.N = 33              # rps, Engine speed
        self.V = 5               # L, Engine displacement
        self.Cd = 0.7            # Coefficient of aerodynamics drag
        self.A = 3.912           # m^2, Frontal Surface Area
        
        # Constants
        self.Xi = 1              # Fuel to Air Mass Ratio
        self.g = 9.81            # m/s^2, Gravitational Constant
        self.Ro = 1.2041         # kg/m^3, Air Density
        self.Cr = 0.01           # Coefficient of rolling resistance
        self.Eta = 0.45          # Efficiency parameter for diesel engines
        self.Eta_tf = 0.45       # Vehicle drivetrain efficiency
        self.Kappa = 44          # kJ/g, Heating value of a typical diesel fuel
        self.Psi = 737           # Conversion factor (g/s to L/s)
        self.FuelToCO2 = 2.65    # kg CO2 per liter of fuel
        
        # Derived constants
        self.Beta = self.Cd * self.Ro * self.A / 2
        self.Gamma = 1 / (1000 * self.Eta * self.Eta_tf)
        self.Lambda = self.Xi / (self.Kappa * self.Psi)
    
    def fuel_consumption(self, distance_m, speed_kmh, load_kg, slope_percent, acceleration=0):
        """
        Calculate fuel consumption in liters for a given arc.
        
        Args:
            distance_m: Distance in METERS (not kilometers!)
            speed_kmh: Speed in km/h
            load_kg: Current load in kg
            slope_percent: Road gradient in percent
            acceleration: Acceleration in m/s^2 (default 0 for steady speed)
        
        Returns:
            Fuel consumption in liters
        """
        # Input validation
        if speed_kmh <= 0 or distance_m <= 0:
            return float('inf')
        
        v_ms = speed_kmh / 3.6  # Convert to m/s
        total_weight = self.CurbWeight + load_kg
        
        # Ensure reasonable bounds
        if total_weight <= 0:
            return float('inf')
        
        try:
            # Topography model (slope-dependent)
            if slope_percent >= 0:
                degree = slope_percent * 0.57
                theta = degree * math.pi / 180  # Convert to radians
                Alpha = acceleration + self.g * math.sin(theta) + self.g * self.Cr * math.cos(theta)
                fuel_liters = self.Lambda * distance_m / v_ms * (
                    self.k * self.N * self.V + 
                    self.Gamma * (total_weight * Alpha * v_ms) + 
                    self.Beta * self.Gamma * v_ms**3
                )
            else:
                # Planar model (flat terrain - minimal consumption)
                fuel_liters = self.Lambda * distance_m / v_ms * (self.k * self.N * self.V)
            
            return max(0, fuel_liters)  # Ensure non-negative
            
        except Exception as e:
            print(f"Warning: Error in fuel_consumption calculation: {e}")
            return float('inf')
    
    def calculate_co2_kg(self, distance_m, speed_kmh, load_kg, slope_percent, acceleration=0):
        """
        Calculate CO2 emissions in kg for a given arc.
        
        Args:
            distance_m: Distance in METERS (not kilometers!)
        
        Returns:
            CO2 emissions in kg
        """
        fuel_liters = self.fuel_consumption(distance_m, speed_kmh, load_kg, slope_percent, acceleration)
        return fuel_liters * self.FuelToCO2


# ============================================================================
# DATA LOADING AND PROBLEM SETUP
# ============================================================================

def load_data():
    """Load problem data from data.txt"""
    import numpy as np
    data = {}
    namespace = {'np': np, 'data': data}  # Provide both np and data to the namespace
    
    # Read and clean the data file
    with open('data.txt', 'r') as f:
        content = f.read()
    
    # Remove leading whitespace from each line
    lines = content.split('\n')
    cleaned_lines = [line.lstrip() for line in lines]
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Execute the cleaned content
    exec(cleaned_content, namespace)
    
    # The values in 'distance_matrix_kilometer' are in KILOMETERS (10.1, 15.6, etc.)
    # but the fuel consumption formula expects distance in METERS!
    # So we need to convert from kilometers to meters by multiplying by 1000
    data['distance_matrix'] = data['distance_matrix_kilometer'] * 1000.0
    
    return data


# ============================================================================
# CORE OPTIMIZATION FUNCTIONS
# ============================================================================

# Global variables (will be initialized in main)
co2_model = None
distance_matrix = None
slope_matrix = None
demands = None
total_demand = None
time_window_min_max_minute = None
delivery_time_minutes = None

def get_optimal_arc_cost(from_node, to_node, current_load):
    """
    Finds the minimum CO2 cost for a single arc (i, j) by
    optimizing the speed in range [20, 40] km/h.
    
    Returns:
        Tuple of (optimal_co2_kg, optimal_speed_kmh)
    """
    distance = distance_matrix[from_node][to_node]  # Distance in meters
    gradient = slope_matrix[from_node][to_node]
    
    if distance == 0:
        return 0, 30  # No travel needed
    
    # Objective function for speed optimization
    def co2_objective_for_speed(speed_kmh):
        try:
            return co2_model.calculate_co2_kg(
                distance_m=distance,  # Distance in meters
                speed_kmh=speed_kmh,
                load_kg=current_load,
                slope_percent=gradient * 100,  # Convert to percentage
                acceleration=0  # Assume steady speed
            )
        except Exception as e:
            # Return large penalty if calculation fails
            return float('inf')
    
    # Solve the 1D optimization problem
    try:
        result = minimize_scalar(
            co2_objective_for_speed,
            method='bounded',
            bounds=(20, 40)  # Speed constraint
        )
        
        # Check if optimization was successful
        if result.success and result.fun != float('inf'):
            return result.fun, result.x
        else:
            # Fallback: use middle speed
            speed = 30
            return co2_objective_for_speed(speed), speed
            
    except Exception as e:
        # Fallback calculation with default speed
        speed = 30
        co2 = co2_objective_for_speed(speed)
        return co2, speed


def evaluate_tour_cost(tour_permutation, return_details=False):
    """
    Calculates the total CO2 cost for an entire tour permutation.
    
    Args:
        tour_permutation: List of nodes, e.g. [0, 5, 12, 2, ..., 30, 0]
        return_details: If True, return detailed arc information
    
    Returns:
        If return_details=False: total_co2 (float)
        If return_details=True: (total_co2, arc_details_list, time_feasible)
    """
    total_co2 = 0
    current_load = total_demand  # Start with full truck
    current_time = 0  # Minutes since leaving depot
    arc_details = []
    time_feasible = True
    
    for i in range(len(tour_permutation) - 1):
        node_A = tour_permutation[i]
        node_B = tour_permutation[i + 1]
        
        # Get optimal arc cost and speed
        optimal_arc_co2, optimal_speed = get_optimal_arc_cost(node_A, node_B, current_load)
        
        # Calculate travel time
        distance_m = distance_matrix[node_A][node_B]  # Distance in meters
        distance_km = distance_m / 1000.0  # Convert to km for time calculation
        travel_time_minutes = (distance_km / optimal_speed) * 60
        
        # Update current time
        current_time += travel_time_minutes
        
        # Check time window feasibility for customer nodes
        if node_B != 0:  # Not returning to depot
            tw_start, tw_end = time_window_min_max_minute[node_B]
            
            # Wait if arrived too early
            if current_time < tw_start:
                current_time = tw_start
            
            # Check if arrived too late
            if current_time > tw_end:
                time_feasible = False
            
            # Add delivery time
            current_time += delivery_time_minutes[node_B]
        
        total_co2 += optimal_arc_co2
        
        if return_details:
            arc_details.append({
                'from': node_A,
                'to': node_B,
                'co2_kg': optimal_arc_co2,
                'speed_kmh': optimal_speed,
                'distance_km': distance_km,  # Already converted to km above
                'load_kg': current_load,
                'arrival_time': current_time - (delivery_times[node_B] if node_B != 0 else 0)
            })
        
        # Update load for next arc (unload at customer)
        if node_B != 0:  # Not returning to depot
            current_load -= demands[node_B]
    
    if return_details:
        return total_co2, arc_details, time_feasible
    return total_co2


# ============================================================================
# ALNS STATE DEFINITION
# ============================================================================

class TourState(State):
    """
    State representation for ALNS algorithm.
    Represents a complete tour as a list of nodes.
    """
    
    def __init__(self, tour):
        # tour is a list of nodes, e.g. [0, 1, 2, ..., 30, 0]
        self.tour = tour
        self._cached_objective = None
    
    def objective(self):
        """
        Returns the cost (CO2 emissions) of this tour.
        Cached for performance.
        """
        if self._cached_objective is None:
            self._cached_objective = evaluate_tour_cost(self.tour)
        return self._cached_objective
    
    def copy(self):
        """Create a deep copy of this state"""
        return TourState(self.tour.copy())


# ============================================================================
# ALNS DESTROY OPERATORS
# ============================================================================

def random_destroy(state, rng):
    """
    Removes random customers (20%) from the tour.
    """
    tour_no_depot = state.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.2))
    
    removed_nodes = []
    for _ in range(n_to_remove):
        if len(tour_no_depot) > 0:
            idx = rng.integers(0, len(tour_no_depot))
            removed_nodes.append(tour_no_depot.pop(idx))
    
    new_state = TourState([0] + tour_no_depot + [0])
    return new_state, removed_nodes


def worst_removal(state, rng):
    """
    Removes customers that contribute most to CO2 emissions.
    Uses delta evaluation for better performance.
    """
    tour_no_depot = state.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.15))
    
    # Calculate removal cost delta for each customer (saving from removal)
    customer_costs = []
    current_load = total_demand
    
    for i in range(1, len(state.tour) - 1):
        node_prev = state.tour[i - 1]
        node_curr = state.tour[i]
        node_next = state.tour[i + 1]
        
        # Cost of current arcs: prev -> curr and curr -> next
        cost_prev_curr, _ = get_optimal_arc_cost(node_prev, node_curr, current_load)
        cost_curr_next, _ = get_optimal_arc_cost(node_curr, node_next, current_load - demands[node_curr])
        
        # Cost of direct arc: prev -> next (without curr)
        cost_direct, _ = get_optimal_arc_cost(node_prev, node_next, current_load)
        
        # Saving from removing this customer
        saving = cost_prev_curr + cost_curr_next - cost_direct
        customer_costs.append((node_curr, saving))
        
        # Update load for next customer
        current_load -= demands[node_curr]
    
    # Sort by saving (descending = highest cost contribution)
    customer_costs.sort(key=lambda x: x[1], reverse=True)
    
    # Remove worst customers
    removed_nodes = []
    for i in range(min(n_to_remove, len(customer_costs))):
        node = customer_costs[i][0]
        removed_nodes.append(node)
        tour_no_depot.remove(node)
    
    new_state = TourState([0] + tour_no_depot + [0])
    return new_state, removed_nodes


def shaw_removal(state, rng):
    """
    Removes customers that are similar (geographically close).
    """
    tour_no_depot = state.tour[1:-1].copy()
    n_to_remove = max(1, int(len(tour_no_depot) * 0.2))
    
    if len(tour_no_depot) == 0:
        return TourState([0, 0]), []
    
    # Select a random seed customer
    seed_idx = rng.integers(0, len(tour_no_depot))
    seed_node = tour_no_depot[seed_idx]
    
    # Calculate relatedness (distance) to seed
    relatedness = []
    for node in tour_no_depot:
        if node != seed_node:
            dist = distance_matrix[seed_node][node]
            relatedness.append((node, dist))
    
    # Sort by relatedness (ascending distance)
    relatedness.sort(key=lambda x: x[1])
    
    # Remove seed and closest customers
    removed_nodes = [seed_node]
    for i in range(min(n_to_remove - 1, len(relatedness))):
        removed_nodes.append(relatedness[i][0])
    
    # Remove from tour
    for node in removed_nodes:
        tour_no_depot.remove(node)
    
    new_state = TourState([0] + tour_no_depot + [0])
    return new_state, removed_nodes


# ============================================================================
# ALNS REPAIR OPERATORS
# ============================================================================

def greedy_repair(destroyed, rng):
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
                best_cost = test_cost
                best_idx = idx
        
        # Insert at best position
        if best_idx != -1:
            state.tour.insert(best_idx, node)
    
    # Clear cached objective
    state._cached_objective = None
    return state


def regret_repair(destroyed, rng):
    """
    Re-inserts nodes based on regret heuristic.
    Prioritizes nodes that have high cost difference between best and second-best positions.
    Note: destroyed is a tuple of (state, removed_nodes) from destroy operator.
    """
    state, removed_nodes = destroyed
    removed_nodes = removed_nodes.copy()  # Make a copy to avoid modifying original
    
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


# ============================================================================
# SOLUTION INITIALIZATION
# ============================================================================

def create_nearest_neighbor_solution():
    """
    Creates an initial solution using nearest neighbor heuristic.
    """
    unvisited = set(range(1, 31))
    tour = [0]
    current_node = 0
    
    while unvisited:
        # Find nearest unvisited customer
        nearest = min(unvisited, key=lambda x: distance_matrix[current_node][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current_node = nearest
    
    tour.append(0)  # Return to depot
    return tour


def create_time_window_aware_solution():
    """
    Creates an initial solution considering time windows.
    """
    customers = list(range(1, 31))
    # Sort by earliest time window start
    customers.sort(key=lambda x: time_window_min_max_minute[x][0])
    
    tour = [0] + customers + [0]
    return tour


# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_vrp(max_iterations=2000, random_seed=1234):
    """
    Main solver function using ALNS algorithm.
    
    Args:
        max_iterations: Maximum number of ALNS iterations
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with solution details
    """
    global co2_model, distance_matrix, slope_matrix, demands, total_demand
    global time_window_min_max_minute, delivery_time_minutes
    
    print("="*70)
    print("CO2-OPTIMIZED VEHICLE ROUTING PROBLEM SOLVER")
    print("="*70)
    
    # Initialize CO2 model
    co2_model = CO2Model()
    print("\n✓ CO2 emission model initialized")
    
    # Load data
    data = load_data()
    distance_matrix = data['distance_matrix']
    slope_matrix = data['slope_matrix_gradient']
    demands = data['demands']
    delivery_time_minutes = data['delivery_time_minute']
    time_window_min_max_minute = data['time_window_min_max_minute']
    total_demand = sum(demands)
    
    print(f"✓ Data loaded: {len(demands)-1} customers")
    print(f"  - Total demand: {total_demand:.2f} kg")
    print(f"  - Speed range: 20-40 km/h")
    print(f"  - CO2 target: ≤ 45 kg (10% reduction from 50 kg baseline)")
    print(f"\n  DEBUG - Distance matrix sample:")
    print(f"  - Distance[0,1]: {distance_matrix[0][1]/1000:.3f} km ({distance_matrix[0][1]:.0f} m)")
    print(f"  - Distance[0,2]: {distance_matrix[0][2]/1000:.3f} km ({distance_matrix[0][2]:.0f} m)")
    print(f"  - Max distance: {distance_matrix.max()/1000:.3f} km ({distance_matrix.max():.0f} m)")
    
    # Create initial solutions and pick the best
    print("\n" + "-"*70)
    print("CREATING INITIAL SOLUTION")
    print("-"*70)
    
    initial_tour_nn = create_nearest_neighbor_solution()
    initial_cost_nn = evaluate_tour_cost(initial_tour_nn)
    
    initial_tour_tw = create_time_window_aware_solution()
    initial_cost_tw = evaluate_tour_cost(initial_tour_tw)
    
    # Random solution
    random_customers = list(range(1, 31))
    random.shuffle(random_customers)
    initial_tour_random = [0] + random_customers + [0]
    initial_cost_random = evaluate_tour_cost(initial_tour_random)
    
    print(f"  Nearest Neighbor:     {initial_cost_nn:.3f} kg CO2")
    print(f"  Time Window Aware:    {initial_cost_tw:.3f} kg CO2")
    print(f"  Random:               {initial_cost_random:.3f} kg CO2")
    
    # Pick best initial solution
    if initial_cost_nn <= initial_cost_tw and initial_cost_nn <= initial_cost_random:
        initial_tour = initial_tour_nn
        initial_cost = initial_cost_nn
        method = "Nearest Neighbor"
    elif initial_cost_tw <= initial_cost_random:
        initial_tour = initial_tour_tw
        initial_cost = initial_cost_tw
        method = "Time Window Aware"
    else:
        initial_tour = initial_tour_random
        initial_cost = initial_cost_random
        method = "Random"
    
    print(f"\n✓ Selected initial solution: {method}")
    print(f"  Initial CO2: {initial_cost:.3f} kg")
    
    # Initialize ALNS
    print("\n" + "-"*70)
    print("RUNNING ALNS OPTIMIZATION")
    print("-"*70)
    
    initial_state = TourState(initial_tour)
    alns = ALNS(np.random.default_rng(random_seed))
    
    # Add destroy operators
    alns.add_destroy_operator(random_destroy)
    alns.add_destroy_operator(worst_removal)
    alns.add_destroy_operator(shaw_removal)
    
    # Add repair operators
    alns.add_repair_operator(greedy_repair)
    alns.add_repair_operator(regret_repair)
    
    # Configure ALNS parameters
    # Operator selection: RouletteWheel with scores [best, better, accepted, rejected]
    select = RouletteWheel(
        scores=[25, 5, 1, 1],  # Rewards for: new best, better, accepted, rejected
        decay=0.8,              # Weight decay parameter
        num_destroy=3,          # Number of destroy operators
        num_repair=2            # Number of repair operators
    )
    
    # Acceptance criterion: Record-to-Record Travel
    # Uses autofit to set thresholds based on initial solution
    accept = RecordToRecordTravel.autofit(
        init_obj=initial_cost,
        start_gap=0.05,  # Start threshold: 5% of initial objective
        end_gap=0.001,   # End threshold: 0.1% of initial objective  
        num_iters=max_iterations,
        method='linear'
    )
    
    # Stopping criterion
    stop = MaxIterations(max_iterations)
    
    print(f"\n  Max iterations: {max_iterations}")
    print(f"  Destroy operators: 3 (random, worst, shaw)")
    print(f"  Repair operators: 2 (greedy, regret)")
    print(f"  Acceptance: Record-to-Record Travel")
    print(f"  Weight decay: 0.8")
    print("\n  Starting optimization...")
    print("  " + "-"*66)
    
    # Run ALNS with progress tracking
    start_time = time.time()
    
    # Progress tracking variables
    iteration = [0]  # Use list to make it mutable in nested functions
    best_cost = [initial_cost]
    last_improvement = [0]
    log_interval = max(1, max_iterations // 20)  # Log ~20 times
    
    # Callback for new best solutions
    def on_best_callback(state, rnd_state):
        iteration[0] += 1
        current_cost = state.objective()
        improvement = best_cost[0] - current_cost
        improvement_pct = (improvement / best_cost[0] * 100) if best_cost[0] != float('inf') else 0
        best_cost[0] = current_cost
        last_improvement[0] = iteration[0]
        print(f"  Iter {iteration[0]:4d}: New best = {current_cost:.3f} kg CO2 (↓ {improvement:.3f} kg, {improvement_pct:.2f}%)")
        
        # Display the solution with speeds
        tour = state.tour
        print(f"    Solution: ", end="")
        current_load = total_demand
        route_info = []
        
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            _, optimal_speed = get_optimal_arc_cost(from_node, to_node, current_load)
            route_info.append(f"({to_node}, {optimal_speed:.1f}km/h)")
            
            # Update load
            if to_node != 0:
                current_load -= demands[to_node]
        
        # Print route in readable format
        print(" → ".join(route_info[:10]))
        if len(route_info) > 10:
            print(f"              ... → {' → '.join(route_info[-3:])}")
        print()
    
    # Callback for periodic logging (on accept/reject)
    def on_iteration_callback(state, rnd_state):
        iteration[0] += 1
        if iteration[0] % log_interval == 0:
            elapsed = time.time() - start_time
            progress = (iteration[0] / max_iterations) * 100
            since_improvement = iteration[0] - last_improvement[0]
            print(f"  Iter {iteration[0]:4d}: Progress {progress:.1f}% | Best = {best_cost[0]:.3f} kg | Last improve: {since_improvement} iters ago | Time: {elapsed:.1f}s")
    
    # Register callbacks
    alns.on_best(on_best_callback)
    alns.on_accept(on_iteration_callback)
    
    result = alns.iterate(initial_state, select, accept, stop)
    end_time = time.time()
    
    # Extract results
    best_solution = result.best_state
    best_cost = best_solution.objective()
    best_tour = best_solution.tour
    
    print("  " + "-"*66)
    print(f"  Final iteration: {max_iterations}")
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\n  Computation time: {end_time - start_time:.2f} seconds")
    print(f"  Iterations completed: {max_iterations}")
    print(f"  Last improvement at iteration: {last_improvement[0]}")
    print(f"  Iterations without improvement: {max_iterations - last_improvement[0]}")
    
    # Get detailed solution
    total_co2, arc_details, time_feasible = evaluate_tour_cost(best_tour, return_details=True)
    
    print(f"\n  Initial CO2:  {initial_cost:.3f} kg")
    print(f"  Final CO2:    {total_co2:.3f} kg")
    print(f"  Improvement:  {((initial_cost - total_co2) / initial_cost * 100):.2f}%")
    print(f"  Target (45kg): {'✓ ACHIEVED' if total_co2 <= 45 else '✗ NOT MET'}")
    
    # Check time feasibility
    print(f"\n  Time windows: {'✓ All feasible' if time_feasible else '⚠ Some violations'}")
    
    # Calculate total distance and time
    total_distance = sum([arc['distance_km'] for arc in arc_details])
    total_time = sum([arc['distance_km'] / arc['speed_kmh'] * 60 for arc in arc_details])
    total_time += sum([delivery_time_minutes[best_tour[i]] for i in range(1, len(best_tour)-1)])
    
    print(f"  Total distance: {total_distance:.2f} km")
    print(f"  Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    
    # Print tour
    print(f"\n  Best tour sequence:")
    print(f"    {' -> '.join(map(str, best_tour[:10]))} -> ...")
    print(f"    ... -> {' -> '.join(map(str, best_tour[-5:]))}")
    
    return {
        'tour': best_tour,
        'co2_kg': total_co2,
        'arc_details': arc_details,
        'initial_cost': initial_cost,
        'improvement_percent': (initial_cost - total_co2) / initial_cost * 100,
        'total_distance_km': total_distance,
        'total_time_minutes': total_time,
        'time_feasible': time_feasible,
        'computation_time': end_time - start_time
    }


def print_detailed_solution(solution):
    """
    Print detailed arc-by-arc solution.
    """
    print("\n" + "="*70)
    print("DETAILED SOLUTION")
    print("="*70)
    
    print("\n{:>4} {:>4} {:>10} {:>8} {:>10} {:>10} {:>12}".format(
        "From", "To", "Dist(km)", "Speed", "Load(kg)", "CO2(kg)", "Arrival(min)"
    ))
    print("-"*70)
    
    for arc in solution['arc_details']:
        print("{:4d} {:4d} {:10.3f} {:8.1f} {:10.1f} {:10.4f} {:12.1f}".format(
            arc['from'],
            arc['to'],
            arc['distance_km'],
            arc['speed_kmh'],
            arc['load_kg'],
            arc['co2_kg'],
            arc['arrival_time']
        ))
    
    print("-"*70)
    print("{:>32} {:10.1f} {:10.4f}".format(
        "TOTALS:",
        solution['total_distance_km'],
        solution['co2_kg']
    ))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Solve the VRP
    solution = solve_vrp(max_iterations=2000, random_seed=1234)
    
    # Print detailed solution
    print_detailed_solution(solution)
    
    # Save solution to file
    print("\n" + "="*70)
    print("SAVING SOLUTION")
    print("="*70)
    
    with open('solution.txt', 'w') as f:
        f.write("CO2-OPTIMIZED VEHICLE ROUTING PROBLEM SOLUTION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total CO2 Emissions: {solution['co2_kg']:.3f} kg\n")
        f.write(f"Total Distance: {solution['total_distance_km']:.2f} km\n")
        f.write(f"Total Time: {solution['total_time_minutes']:.1f} minutes\n")
        f.write(f"Computation Time: {solution['computation_time']:.2f} seconds\n")
        f.write(f"Improvement over initial: {solution['improvement_percent']:.2f}%\n")
        f.write(f"Target (45 kg): {'ACHIEVED ✓' if solution['co2_kg'] <= 45 else 'NOT MET ✗'}\n\n")
        f.write(f"Tour sequence:\n")
        f.write(f"{' -> '.join(map(str, solution['tour']))}\n\n")
        f.write("\nDetailed Arc Information:\n")
        f.write("-"*70 + "\n")
        f.write("{:>4} {:>4} {:>10} {:>8} {:>10} {:>10} {:>12}\n".format(
            "From", "To", "Dist(km)", "Speed", "Load(kg)", "CO2(kg)", "Arrival(min)"
        ))
        f.write("-"*70 + "\n")
        for arc in solution['arc_details']:
            f.write("{:4d} {:4d} {:10.3f} {:8.1f} {:10.1f} {:10.4f} {:12.1f}\n".format(
                arc['from'], arc['to'], arc['distance_km'], arc['speed_kmh'],
                arc['load_kg'], arc['co2_kg'], arc['arrival_time']
            ))
    
    print("✓ Solution saved to 'solution.txt'")
    print("\nDone!")
