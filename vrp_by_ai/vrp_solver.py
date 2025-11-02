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
    Based on Moradi, Arts and Velazquez-Martinez - 
    "Load Asymptotics and Dynamic Speed Optimization for the Greenest Path Problem: 
    A Comprehensive Analysis"
    
    Uses the provided fuel consumption formula considering:
    - Vehicle load
    - Speed
    - Road slope/gradient (in radians)
    - Aerodynamics and rolling resistance
    """
    
    def __init__(self):
        # Vehicle Parameters
        self.w = 5000            # kg, Truck weight (empty)
        self.k = 0.20            # Engine friction factor
        self.N = 33              # rps, Engine speed
        self.D = 5               # L, Engine displacement
        self.Cd = 0.7            # Coefficient of aerodynamic drag
        self.S = 8               # m^2, Frontal area
        
        # Constants
        self.epsilon = 1         # Fuel-to-air ratio
        self.kapa = 44           # kJ/g, Heating value of diesel
        self.psi = 737           # Conversion factor
        self.g = 9.81            # m/s^2, Gravitational constant
        self.Cr = 0.01           # Coefficient of rolling resistance
        self.ro = 1.2041         # kg/m^3, Air density
        self.eta = 0.45          # Engine efficiency
        self.eta_t = 0.45        # Drivetrain efficiency
        self.FuelToCO2 = 2.65    # kg CO2 per liter of fuel
        
        # Derived constants (pre-compute for efficiency)
        # P = epsilon * k * N * D / (kapa * psi)
        self.P = self.epsilon * self.k * self.N * self.D / (self.kapa * self.psi)
        
        # Q = epsilon / (1000 * eta * eta_t * kapa * psi)
        self.Q = self.epsilon / (1000 * self.eta * self.eta_t * self.kapa * self.psi)
        
        # R = epsilon * Cd * ro * S / (2000 * eta * eta_t * kapa * psi)
        self.R = self.epsilon * self.Cd * self.ro * self.S / (2000 * self.eta * self.eta_t * self.kapa * self.psi)
    
    def fuel_consumption(self, distance_km, speed_kmh, load_kg, slope_gradient):
        """
        Calculate fuel consumption in liters for a given arc.
        
        Args:
            distance_km: Distance in KILOMETERS
            speed_kmh: Speed in km/h
            load_kg: Current load in kg (sum of remaining demands)
            slope_gradient: Road gradient (slope as decimal, e.g., 0.05 for 5% grade)
        
        Returns:
            Fuel consumption in liters
        """
        # Input validation
        if speed_kmh <= 0 or distance_km <= 0:
            return float('inf')
        
        try:
            # Convert speed to m/s
            speed_ms = speed_kmh / 3.6
            
            # Convert distance to meters
            distance_m = distance_km * 1000
            
            # Angle in radians (approximation with 1% precision if slope < 0.15)
            angle_radians = slope_gradient
            
            # Calculate acceleration component due to slope and rolling resistance
            acceleration_theta = (self.g * math.sin(angle_radians) + 
                                 self.g * self.Cr * math.cos(angle_radians))
            
            # Calculate fuel consumption using the new formula:
            # fuel = P * distance_m / speed_ms + 
            #        max(0, Q * distance_m * acceleration_theta * (w + load) + 
            #               R * distance_m * speed_ms^2)
            
            planar_component = self.P * distance_m / speed_ms
            
            load_slope_component = self.Q * distance_m * acceleration_theta * (self.w + load_kg)
            
            aerodynamic_component = self.R * distance_m * speed_ms**2
            
            fuel_liters = planar_component + max(0, load_slope_component + aerodynamic_component)
            
            return max(0, fuel_liters)  # Ensure non-negative
            
        except Exception as e:
            print(f"Warning: Error in fuel_consumption calculation: {e}")
            return float('inf')
    
    def calculate_co2_kg(self, distance_km, speed_kmh, load_kg, slope_gradient):
        """
        Calculate CO2 emissions in kg for a given arc.
        
        Args:
            distance_km: Distance in KILOMETERS
            speed_kmh: Speed in km/h
            load_kg: Current load in kg (sum of remaining demands)
            slope_gradient: Road gradient (slope as decimal)
        
        Returns:
            CO2 emissions in kg
        """
        fuel_liters = self.fuel_consumption(distance_km, speed_kmh, load_kg, slope_gradient)
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
    
    # Map new variable names to expected names
    # The new formula expects distance in KILOMETERS
    data['distance_matrix'] = np.array(namespace['distance_matrix_km'])
    data['slope_matrix_gradient'] = np.array(namespace['average_slope_matrix_gradient'])
    data['demands'] = np.array(namespace['demand_weight_kg'])
    
    # Convert time windows from part of 24H to minutes
    not_before = namespace['not_before_part_of_24H']
    not_after = namespace['not_after_part_of_24H']
    
    time_window_min_max_minute = []
    for i in range(len(not_before)):
        tw_start = not_before[i] * 24 * 60  # Convert to minutes
        tw_end = not_after[i] * 24 * 60     # Convert to minutes
        time_window_min_max_minute.append((tw_start, tw_end))
    
    data['time_window_min_max_minute'] = time_window_min_max_minute
    data['delivery_time_minute'] = namespace['delivery_time_minute']
    
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
max_time_window_violations = 0  # Maximum allowed violations
time_window_violation_penalty = 1000.0  # Penalty per excess violation in kg CO2

def get_optimal_arc_cost(from_node, to_node, current_load):
    """
    Finds the minimum CO2 cost for a single arc (i, j) by
    optimizing the speed in range [20, 40] km/h.
    
    Args:
        from_node: Starting node
        to_node: Destination node
        current_load: Current load in kg (sum of remaining demands to deliver)
    
    Returns:
        Tuple of (optimal_co2_kg, optimal_speed_kmh)
    """
    distance_km = distance_matrix[from_node][to_node]  # Distance in kilometers
    gradient = slope_matrix[from_node][to_node]  # Slope as gradient (decimal)
    
    if distance_km == 0:
        return 0, 30  # No travel needed
    
    # Objective function for speed optimization
    def co2_objective_for_speed(speed_kmh):
        try:
            return co2_model.calculate_co2_kg(
                distance_km=distance_km,  # Distance in kilometers
                speed_kmh=speed_kmh,
                load_kg=current_load,
                slope_gradient=gradient  # Gradient as decimal (not percentage)
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


def evaluate_tour_cost(tour_permutation, return_details=False, max_violations=0, violation_penalty=1000.0, for_objective=True):
    """
    Calculates the total CO2 cost for an entire tour permutation.
    
    Args:
        tour_permutation: List of nodes, e.g. [0, 5, 12, 2, ..., 30, 0]
        return_details: If True, return detailed arc information
        max_violations: Maximum allowed time window violations (0 = strict)
        violation_penalty: Penalty per violation beyond max_violations (in kg CO2)
        for_objective: If True, apply penalty for objective function; if False, return pure CO2
    
    Returns:
        If return_details=False: penalized_cost (float) or pure_co2 (float) depending on for_objective
        If return_details=True: (pure_co2, penalized_cost, arc_details_list, violation_count, violated_nodes)
    """
    pure_co2 = 0
    current_load = total_demand  # Start with full truck
    current_time = 420  # Start at 07:00 (7 * 60 = 420 minutes)
    arc_details = []
    violation_count = 0
    violated_nodes = []
    
    for i in range(len(tour_permutation) - 1):
        node_A = tour_permutation[i]
        node_B = tour_permutation[i + 1]
        
        # Get optimal arc cost and speed
        optimal_arc_co2, optimal_speed = get_optimal_arc_cost(node_A, node_B, current_load)
        
        # Calculate travel time
        distance_km = distance_matrix[node_A][node_B]  # Distance in kilometers
        travel_time_minutes = (distance_km / optimal_speed) * 60
        
        # Update current time
        current_time += travel_time_minutes
        
        # Check time window feasibility for customer nodes
        if node_B != 0:  # Not returning to depot
            tw_start, tw_end = time_window_min_max_minute[node_B]
            
            # Check if arrived too early or too late
            arrival_time = current_time
            if current_time < tw_start:
                violation_count += 1
                violated_nodes.append(node_B)
                current_time = tw_start  # Wait until window opens
            elif current_time > tw_end:
                violation_count += 1
                violated_nodes.append(node_B)
            
            # Add delivery time
            current_time += delivery_time_minutes[node_B]
        
        pure_co2 += optimal_arc_co2
        
        if return_details:
            arc_details.append({
                'from': node_A,
                'to': node_B,
                'co2_kg': optimal_arc_co2,
                'speed_kmh': optimal_speed,
                'distance_km': distance_km,
                'load_kg': current_load,
                'arrival_time': current_time - (delivery_time_minutes[node_B] if node_B != 0 else 0)
            })
        
        # Update load for next arc (unload at customer)
        if node_B != 0:  # Not returning to depot
            current_load -= demands[node_B]
    
    # Calculate penalized cost (for objective function)
    penalized_cost = pure_co2
    if violation_count > max_violations:
        excess_violations = violation_count - max_violations
        penalized_cost += excess_violations * violation_penalty
    
    if return_details:
        return pure_co2, penalized_cost, arc_details, violation_count, violated_nodes
    
    # Return appropriate value based on context
    return penalized_cost if for_objective else pure_co2


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
            self._cached_objective = evaluate_tour_cost(
                self.tour, 
                max_violations=max_time_window_violations,
                violation_penalty=time_window_violation_penalty
            )
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


def format_comprehensive_schedule(tour, arc_details, violated_nodes):
    """
    Create a comprehensive schedule combining arc details and delivery information.
    Returns a list of strings for printing/saving.
    """
    lines = []
    lines.append("    Complete Schedule:")
    lines.append("    " + "-"*130)
    lines.append(f"    {'From':>4} {'To':>4} {'Dist':>7} {'Speed':>7} {'Load':>8} {'CO2':>8} {'Arrival':>9} {'Delivery':>9} {'Duration':>9} {'Time Window':>18} {'Status':>10}")
    lines.append(f"    {'':>4} {'':>4} {'(km)':>7} {'(km/h)':>7} {'(kg)':>8} {'(kg)':>8} {'(time)':>9} {'(time)':>9} {'(min)':>9} {'':>18} {'':>10}")
    lines.append("    " + "-"*130)
    
    current_time = 420  # Start at 07:00 (7 * 60 = 420 minutes)
    violated_set = set(violated_nodes)
    
    # Depot start
    lines.append(f"    {'-':>4} {0:>4} {'-':>7} {'-':>7} {'-':>8} {'-':>8} {'07:00':>9} {'-':>9} {'-':>9} {'-':>18} {'DEPOT':>10}")
    
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        arc = arc_details[i]
        
        # Travel to node
        distance_km = arc['distance_km']
        speed_kmh = arc['speed_kmh']
        load_kg = arc['load_kg']
        co2_kg = arc['co2_kg']
        travel_time = (distance_km / speed_kmh) * 60
        current_time += travel_time
        
        if to_node == 0:  # Return to depot
            lines.append(f"    {from_node:>4} {0:>4} {distance_km:>7.2f} {speed_kmh:>7.1f} {load_kg:>8.1f} {co2_kg:>8.4f} {f'{int(current_time//60):02d}:{int(current_time%60):02d}':>9} {'-':>9} {'-':>9} {'-':>18} {'DEPOT':>10}")
        else:
            # Get time window
            tw_start, tw_end = time_window_min_max_minute[to_node]
            
            # Wait if early
            arrival_time = current_time
            if current_time < tw_start:
                current_time = tw_start
            
            # Check status
            if arrival_time > tw_end:
                status = "✗ LATE"
            elif arrival_time < tw_start:
                status = "✗ EARLY"
            else:
                status = "✓ ON-TIME"
            
            # Format
            arrival_str = f"{int(arrival_time//60):02d}:{int(arrival_time%60):02d}"
            delivery_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            delivery_duration = delivery_time_minutes[to_node]
            tw_str = f"{int(tw_start//60):02d}:{int(tw_start%60):02d}-{int(tw_end//60):02d}:{int(tw_end%60):02d}"
            
            lines.append(f"    {from_node:>4} {to_node:>4} {distance_km:>7.2f} {speed_kmh:>7.1f} {load_kg:>8.1f} {co2_kg:>8.4f} {arrival_str:>9} {delivery_time_str:>9} {delivery_duration:>9} {tw_str:>18} {status:>10}")
            
            # Add delivery time
            current_time += delivery_duration
    
    lines.append("    " + "-"*130)
    return lines


# ============================================================================
# SANITY CHECK
# ============================================================================

def sanity_check():
    """
    Sanity check to verify the implementation with a known test case.
    Tests the tour with all nodes in order (0, 1, 2, ..., 30, 0) at constant 25 km/h.
    
    Expected results:
    - On-time delivery: 53%
    - Total Distance: 255.5 km
    - Total CO2 Emissions: 173.1 kgCO2
    """
    global co2_model, distance_matrix, slope_matrix, demands, total_demand
    global time_window_min_max_minute, delivery_time_minutes
    
    print("="*70)
    print("SANITY CHECK - SEQUENTIAL TOUR AT 25 KM/H")
    print("="*70)
    
    # Initialize CO2 model
    co2_model = CO2Model()
    
    # Load data
    data = load_data()
    distance_matrix = data['distance_matrix']
    slope_matrix = data['slope_matrix_gradient']
    demands = data['demands']
    delivery_time_minutes = data['delivery_time_minute']
    time_window_min_max_minute = data['time_window_min_max_minute']
    total_demand = sum(demands)
    
    # Create sequential tour: 0 -> 1 -> 2 -> ... -> 30 -> 0
    tour = list(range(31)) + [0]
    
    # Calculate with fixed speed of 25 km/h
    total_co2 = 0
    total_distance = 0
    current_load = total_demand
    current_time = 7 * 60  # Start at 07:00 (420 minutes from midnight)
    
    on_time_count = 0
    late_count = 0
    
    print("\nID Order | Delivery Time | Delivery Windows | Status")
    print("-" * 70)
    print(f"{'DC':>8} | {current_time//60:02d}:{current_time%60:02d}          |              |")
    
    for i in range(len(tour) - 1):
        node_A = tour[i]
        node_B = tour[i + 1]
        
        # Calculate arc cost at 25 km/h
        distance_km = distance_matrix[node_A][node_B]
        gradient = slope_matrix[node_A][node_B]
        
        arc_co2 = co2_model.calculate_co2_kg(
            distance_km=distance_km,
            speed_kmh=25,
            load_kg=current_load,
            slope_gradient=gradient
        )
        
        total_co2 += arc_co2
        total_distance += distance_km
        
        # Calculate travel time
        travel_time_minutes = (distance_km / 25) * 60
        current_time += travel_time_minutes
        
        # Check time windows and delivery (only for customers, not depot return)
        if node_B != 0:
            tw_start, tw_end = time_window_min_max_minute[node_B]
            
            # Format time windows
            tw_start_str = f"{int(tw_start//60):02d}:{int(tw_start%60):02d}"
            tw_end_str = f"{int(tw_end//60):02d}:{int(tw_end%60):02d}"
            tw_window = f"{tw_start_str}-{tw_end_str}"
            
            # Wait if arrived too early
            arrival_time = current_time
            if current_time < tw_start:
                current_time = tw_start
            
            # Check if on-time
            if arrival_time <= tw_end:
                status = "✓ On-time"
                on_time_count += 1
            else:
                status = "✗ Late"
                late_count += 1
            
            # Format delivery time
            delivery_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            
            print(f"{node_B:>8} | {delivery_time_str}          | {tw_window:12} | {status}")
            
            # Add delivery time
            current_time += delivery_time_minutes[node_B]
            
            # Update load
            current_load -= demands[node_B]
        else:
            # Return to depot
            return_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            print(f"{'DC':>8} | {return_time_str}          |              |")
    
    print("-" * 70)
    
    # Calculate on-time percentage
    on_time_percentage = (on_time_count / 30) * 100
    
    print(f"\n{'RESULTS':^70}")
    print("=" * 70)
    print(f"On-time delivery:        {on_time_percentage:.0f}% ({on_time_count}/30)")
    print(f"Total Distance Travelled: {total_distance:.1f} km")
    print(f"Total CO2 Emissions:      {total_co2:.1f} kgCO2")
    
    print(f"\n{'EXPECTED VALUES':^70}")
    print("=" * 70)
    print(f"On-time delivery:        53% (16/30)")
    print(f"Total Distance Travelled: 255.5 km")
    print(f"Total CO2 Emissions:      173.1 kgCO2")
    
    print(f"\n{'COMPARISON':^70}")
    print("=" * 70)
    
    # Distance comparison
    distance_diff = abs(total_distance - 255.5)
    distance_match = "✓ MATCH" if distance_diff < 0.5 else f"✗ DIFF: {distance_diff:.1f} km"
    print(f"Distance:    {distance_match}")
    
    # CO2 comparison
    co2_diff = abs(total_co2 - 173.1)
    co2_match = "✓ MATCH" if co2_diff < 1.0 else f"✗ DIFF: {co2_diff:.1f} kgCO2"
    print(f"CO2:         {co2_match}")
    
    # On-time comparison
    ontime_diff = abs(on_time_percentage - 53)
    ontime_match = "✓ MATCH" if ontime_diff < 1 else f"✗ DIFF: {ontime_diff:.0f}%"
    print(f"On-time:     {ontime_match}")
    
    print("\n" + "="*70)
    
    if distance_diff < 0.5 and co2_diff < 1.0 and ontime_diff < 1:
        print("✓ SANITY CHECK PASSED - Implementation is correct!")
    else:
        print("⚠ SANITY CHECK FAILED - Please review the implementation")
    
    print("="*70 + "\n")
    
    return {
        'on_time_percentage': on_time_percentage,
        'total_distance': total_distance,
        'total_co2': total_co2
    }


# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_vrp(max_iterations=2000, random_seed=1234, sanity_check_only=False, 
              max_violations=0, violation_penalty=1000.0):
    """
    Main solver function using ALNS algorithm.
    
    Args:
        max_iterations: Maximum number of ALNS iterations
        random_seed: Random seed for reproducibility
        sanity_check_only: If True, only run the sanity check and exit
        max_violations: Maximum allowed time window violations (0 = strict)
        violation_penalty: Penalty per violation beyond max_violations (in kg CO2)
    
    Returns:
        Dictionary with solution details
    """
    global co2_model, distance_matrix, slope_matrix, demands, total_demand
    global time_window_min_max_minute, delivery_time_minutes
    global max_time_window_violations, time_window_violation_penalty
    
    # If sanity check only, run it and return
    if sanity_check_only:
        return sanity_check()
    
    # Set global violation parameters
    max_time_window_violations = max_violations
    time_window_violation_penalty = violation_penalty
    
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
    print(f"  - Time window violations allowed: {max_violations}")
    print(f"  - Violation penalty: {violation_penalty:.1f} kg CO2 per excess violation")
    print(f"\n  DEBUG - Distance matrix sample:")
    print(f"  - Distance[0,1]: {distance_matrix[0][1]:.3f} km")
    print(f"  - Distance[0,2]: {distance_matrix[0][2]:.3f} km")
    print(f"  - Max distance: {distance_matrix.max():.3f} km")
    
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
        
        # Get pure CO2 and violation info
        pure_co2, penalized_cost, arc_details, violation_count, violated_nodes = evaluate_tour_cost(
            state.tour, 
            return_details=True,
            max_violations=max_violations,
            violation_penalty=violation_penalty
        )
        
        # Calculate on-time delivery percentage
        on_time_count = 30 - violation_count
        on_time_pct = (on_time_count / 30) * 100
        
        improvement = best_cost[0] - current_cost
        improvement_pct = (improvement / best_cost[0] * 100) if best_cost[0] != float('inf') else 0
        best_cost[0] = current_cost
        last_improvement[0] = iteration[0]
        
        print(f"  Iter {iteration[0]:4d}: New best = {pure_co2:.3f} kg CO2 (↓ {improvement:.3f} kg, {improvement_pct:.2f}%) | On-time: {on_time_pct:.0f}% ({on_time_count}/30)")
        
        # Display the complete solution with speeds (non-truncated)
        tour = state.tour
        print(f"    Complete Route:")
        current_load = total_demand
        
        # Build complete route with node IDs and speeds
        route_parts = []
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            _, optimal_speed = get_optimal_arc_cost(from_node, to_node, current_load)
            route_parts.append(f"{to_node}@{optimal_speed:.1f}km/h")
            
            # Update load
            if to_node != 0:
                current_load -= demands[to_node]
        
        # Print complete route (non-truncated) with line breaks for readability
        route_str = " → ".join(route_parts)
        # Break into lines of ~100 characters for better readability
        line_length = 100
        words = route_str.split(" → ")
        current_line = "    "
        for i, word in enumerate(words):
            if len(current_line) + len(word) + 3 > line_length and current_line != "    ":
                print(current_line)
                current_line = "    → " + word
            else:
                if current_line == "    ":
                    current_line += word
                else:
                    current_line += " → " + word
        if current_line != "    ":
            print(current_line)
        
        # Display comprehensive schedule
        schedule_lines = format_comprehensive_schedule(tour, arc_details, violated_nodes)
        for line in schedule_lines:
            print(line)
        
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
    
    # Get detailed solution (pure CO2 without penalty)
    pure_co2, penalized_cost, arc_details, violation_count, violated_nodes = evaluate_tour_cost(
        best_tour, 
        return_details=True, 
        max_violations=max_violations, 
        violation_penalty=violation_penalty
    )
    
    print(f"\n  Initial CO2:  {initial_cost:.3f} kg")
    print(f"  Final CO2:    {pure_co2:.3f} kg (pure, without penalty)")
    print(f"  Improvement:  {((initial_cost - pure_co2) / initial_cost * 100):.2f}%")
    print(f"  Target (45kg): {'✓ ACHIEVED' if pure_co2 <= 45 else '✗ NOT MET'}")
    
    # Check time feasibility
    if violation_count == 0:
        print(f"\n  Time windows: ✓ All customers served on time")
    elif violation_count <= max_violations:
        print(f"\n  Time windows: ⚠ {violation_count} violations (within allowed limit of {max_violations})")
        print(f"    Violated nodes: {violated_nodes}")
    else:
        excess = violation_count - max_violations
        print(f"\n  Time windows: ✗ {violation_count} violations (exceeded limit by {excess})")
        print(f"    Violated nodes: {violated_nodes}")
        print(f"    Penalty applied: {excess * violation_penalty:.1f} kg CO2")
    
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
        'co2_kg': pure_co2,
        'arc_details': arc_details,
        'initial_cost': initial_cost,
        'improvement_percent': (initial_cost - pure_co2) / initial_cost * 100,
        'total_distance_km': total_distance,
        'total_time_minutes': total_time,
        'violation_count': violation_count,
        'violated_nodes': violated_nodes,
        'computation_time': end_time - start_time,
        'max_violations': max_violations,
        'violation_penalty': violation_penalty,
        'random_seed': random_seed
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CO2-Optimized Vehicle Routing Problem Solver')
    parser.add_argument('--iterations', type=int, default=2000, 
                        help='Maximum number of ALNS iterations (default: 2000)')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='Random seed for reproducibility (default: 1234)')
    parser.add_argument('--max-violations', type=int, default=0, 
                        help='Maximum allowed time window violations (default: 0 = strict)')
    parser.add_argument('--violation-penalty', type=float, default=1000.0, 
                        help='Penalty per excess violation in kg CO2 (default: 1000.0)')
    parser.add_argument('--sanity-check', action='store_true', 
                        help='Run only the sanity check and exit')
    
    args = parser.parse_args()
    
    # Solve the VRP
    solution = solve_vrp(
        max_iterations=args.iterations, 
        random_seed=args.seed,
        sanity_check_only=args.sanity_check,
        max_violations=args.max_violations,
        violation_penalty=args.violation_penalty
    )
    
    # Print detailed solution (skip if sanity check only)
    if not args.sanity_check:
        print_detailed_solution(solution)
    
    # Save solution to file
    print("\n" + "="*70)
    print("SAVING SOLUTION")
    print("="*70)
    
    # Calculate node list with speeds
    tour = solution['tour']
    current_load = total_demand
    node_speed_list = []
    
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        _, optimal_speed = get_optimal_arc_cost(from_node, to_node, current_load)
        node_speed_list.append(f"{to_node}@{optimal_speed:.1f}km/h")
        
        # Update load
        if to_node != 0:
            current_load -= demands[to_node]
    
    with open('solution.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("CO2-OPTIMIZED VEHICLE ROUTING PROBLEM SOLUTION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random Seed: {solution['random_seed']}\n")
        f.write(f"Max Violations Allowed: {solution['max_violations']}\n")
        f.write(f"Violation Penalty: {solution['violation_penalty']:.1f} kg CO2\n\n")
        f.write(f"Total CO2 Emissions: {solution['co2_kg']:.3f} kg\n")
        f.write(f"Total Distance: {solution['total_distance_km']:.2f} km\n")
        f.write(f"Total Time: {solution['total_time_minutes']:.1f} minutes\n")
        f.write(f"Computation Time: {solution['computation_time']:.2f} seconds\n")
        f.write(f"Improvement over initial: {solution['improvement_percent']:.2f}%\n")
        f.write(f"Target (45 kg): {'ACHIEVED ✓' if solution['co2_kg'] <= 45 else 'NOT MET ✗'}\n")
        f.write(f"Violations: {solution['violation_count']}/30 customers\n")
        f.write(f"On-time delivery: {100 - (solution['violation_count']/30*100):.0f}%\n\n")
        
        f.write(f"Tour sequence (simple):\n")
        f.write(f"{' -> '.join(map(str, solution['tour']))}\n\n")
        
        f.write(f"Complete Route with Speeds:\n")
        f.write("-"*70 + "\n")
        # Print route in chunks of 80 characters for readability
        route_str = " → ".join(node_speed_list)
        line_length = 70
        words = route_str.split(" → ")
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 3 > line_length and current_line:
                f.write(current_line + "\n")
                current_line = "→ " + word
            else:
                if current_line:
                    current_line += " → " + word
                else:
                    current_line = word
        if current_line:
            f.write(current_line + "\n")
        f.write("-"*70 + "\n\n")
        
        # Add comprehensive schedule with all details
        f.write("Complete Schedule (Arc Details + Delivery Information):\n")
        f.write("="*70 + "\n")
        schedule_lines = format_comprehensive_schedule(tour, solution['arc_details'], solution['violated_nodes'])
        for line in schedule_lines:
            f.write(line.replace("    ", "") + "\n")  # Remove console indentation for file
        
        f.write(f"\nOn-time deliveries: {30 - solution['violation_count']}/30 ({100 - (solution['violation_count']/30*100):.0f}%)\n")
        if solution['violation_count'] > 0:
            f.write(f"Late deliveries: {solution['violated_nodes']}\n")
        f.write("\n")
    
    print("✓ Solution appended to 'solution.txt'")
    print("\nDone!")
