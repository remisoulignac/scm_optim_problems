"""
Run only the sanity check for the VRP solver
"""

from vrp_solver import solve_vrp

if __name__ == "__main__":
    # Run sanity check only
    result = solve_vrp(sanity_check_only=True)
    
    print("\nSanity check completed!")
    print(f"Result: {result}")
