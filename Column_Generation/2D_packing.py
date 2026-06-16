import numpy as np
from scipy.optimize import linprog

class TwoDStockCutting:
    """
    2D Stock Cutting Problem Solver using Column Generation
    
    Master Problem: Minimize number of large sheets used
    Pricing Problem: 2D Knapsack with length and width constraints
    """
    
    def __init__(self, sheet_length, sheet_width, piece_demands):
        """
        Args:
            sheet_length: Length of large sheet (L)
            sheet_width: Width of large sheet (W)
            piece_demands: Dict {(length, width): demand_quantity}
                          e.g., {(20, 15): 4, (45, 30): 5, (50, 25): 3}
        """
        self.L = sheet_length
        self.W = sheet_width
        self.piece_demands = piece_demands
        self.pieces = list(piece_demands.keys())
        self.demands = [piece_demands[p] for p in self.pieces]
        self.n_pieces = len(self.pieces)
        
        self.patterns = []
        
        self._initialize_patterns()
        #For basic patterns at the start
        
    def _initialize_patterns(self):
        """Create initial patterns - one piece type per pattern"""
        for i in range(self.n_pieces):
            pattern = [0] * self.n_pieces
            pattern[i] = 1
            self.patterns.append(pattern)
    
    def solve_master_problem(self):
        """
        Solve the Restricted Master Problem (RMP) as LP
        
        minimize: sum(x_p) for all patterns p
        subject to: sum(a_ip * x_p) >= d_i for all pieces i
                    x_p >= 0
        
        Returns: objective value, solution, dual values (prices p_i)
        """
        n_patterns = len(self.patterns)
        
        # Objective: minimize sum of all pattern usage (all coefficients = 1)
        c = np.ones(n_patterns)
        
        # Constraint matrix: A[i][p] = number of piece i in pattern p
        # Use negative for >= constraints (converted to <= for linprog)
        A_ub = -np.array(self.patterns).T
        b_ub = -np.array(self.demands)
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        
        if not result.success:
            raise Exception("Master problem failed to solve")
        
        # Extract dual values (shadow prices) - these are the p_i values
        dual_values = -result.ineqlin.marginals  # Dual variables for constraints
        
        return result.fun, result.x, dual_values
    
    def solve_pricing_problem(self, p):
        """
        Solve 2D Knapsack Pricing Problem using Dynamic Programming
        
        maximize: sum(p_i * z_i)
        subject to: sum(length_i * z_i) <= L
                    sum(width_i * z_i) <= W
                    z_i >= 0, integer
        
        Args:
            p: dual values (prices) from master problem [p_0, p_1, ..., p_n]
        
        Returns: reduced_cost, new_pattern (or None if no improvement)
        """
        # DP table: F[(l, w)] = (max_value, pattern)
        # where max_value = sum(p_i * z_i)
        F = {}
        F[(0, 0)] = (0.0, [0] * self.n_pieces)
        
        # For each piece type
        for i in range(self.n_pieces):
            piece_len, piece_width = self.pieces[i]
            piece_value = p[i]  # Price (dual value) for this piece
            
            # Create new DP entries by adding this piece
            new_entries = {}
            
            for (curr_len, curr_width), (curr_val, curr_pattern) in F.items():
                # Try adding multiples of piece i
                max_count_len = (self.L - curr_len) // piece_len
                max_count_width = (self.W - curr_width) // piece_width
                max_count = min(max_count_len, max_count_width)
                
                for count in range(1, max_count + 1):
                    new_len = curr_len + count * piece_len
                    new_width = curr_width + count * piece_width
                    
                    if new_len <= self.L and new_width <= self.W:
                        new_val = curr_val + count * piece_value
                        new_pattern = curr_pattern.copy()
                        new_pattern[i] += count
                        
                        # Update if better
                        if (new_len, new_width) not in F or F[(new_len, new_width)][0] < new_val:
                            new_entries[(new_len, new_width)] = (new_val, new_pattern)
            
            F.update(new_entries)
        
        # Find best pattern
        best_val = 0.0
        best_pattern = None
        
        for (curr_val, curr_pattern) in F.values():
            if curr_val > best_val:
                best_val = curr_val
                best_pattern = curr_pattern
        
        # Reduced cost = 1 - sum(p_i * z_i)
        reduced_cost = 1.0 - best_val
        
        if reduced_cost < -1e-6:  # Negative reduced cost found
            return reduced_cost, best_pattern
        else:
            return reduced_cost, None
    
    def solve_column_generation(self, max_iterations=100, verbose=True):
        """
        Column Generation Loop
        
        Returns: objective value, patterns used with quantities
        """
        if verbose:
            print(f"Sheet dimensions: {self.L} x {self.W}")
            print(f"Piece demands: {self.piece_demands}")
            print()
        
        for iteration in range(max_iterations):
            # Solve master problem
            obj_val, solution, p = self.solve_master_problem()
            
            if verbose:
                print(f"Iteration {iteration + 1}:")
                print(f"  Objective (LP relaxation): {obj_val:.4f}")
                print(f"  Dual values (prices p_i): {p}")
            
            # Solve pricing problem
            reduced_cost, new_pattern = self.solve_pricing_problem(p)
            
            if verbose:
                print(f"  Reduced cost: {reduced_cost:.6f}")
            
            if new_pattern is None:
                if verbose:
                    print(f"\nColumn generation converged!")
                    print(f"Final LP objective: {obj_val:.4f}")
                    print(f"Rounded up (integer solution): {int(np.ceil(obj_val))}")
                
                # Return solution
                pattern_usage = {}
                for idx, usage in enumerate(solution):
                    if usage > 1e-6:
                        pattern_usage[idx] = usage
                
                return obj_val, pattern_usage
            
            # Add new pattern
            self.patterns.append(new_pattern)
            if verbose:
                print(f"  New pattern added: {new_pattern}")
                print()
        
        print("Warning: Maximum iterations reached")
        return None, None
    
    def print_solution(self, pattern_usage):
        """Print the solution in readable format"""
        print("\n" + "=" * 60)
        print("SOLUTION SUMMARY")
        print("=" * 60)
        
        total_sheets = 0.0
        for pattern_idx, usage in pattern_usage.items():
            pattern = self.patterns[pattern_idx]
            print(f"\nPattern {pattern_idx} (use {usage:.2f} times):")
            for i, count in enumerate(pattern):
                if count > 0:
                    piece = self.pieces[i]
                    print(f"  - {count}x piece of size {piece[0]} x {piece[1]}")
            total_sheets += usage
        
        print(f"\nTotal sheets needed (LP relaxation): {total_sheets:.2f}")
        print(f"Rounded up (integer solution): {int(np.ceil(total_sheets))}")



if __name__ == "__main__":
    
    # Example 1: Simple 2D cutting stock problem
    print("EXAMPLE 1: Simple 2D Cutting Stock")
    print("=" * 60)
    
    problem1 = TwoDStockCutting(
        sheet_length=100,
        sheet_width=100,
        piece_demands={
            (20, 15): 4,   # 4 pieces of 20x15
            (45, 30): 5,   # 5 pieces of 45x30
            (50, 25): 3    # 3 pieces of 50x25
        }
    )
    
    obj_val, pattern_usage = problem1.solve_column_generation()
    
    if pattern_usage:
        problem1.print_solution(pattern_usage)
    
    
    # Example 2: More complex problem
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: More Complex 2D Cutting Stock")
    print("=" * 60)
    
    problem2 = TwoDStockCutting(
        sheet_length=200,
        sheet_width=150,
        piece_demands={
            (30, 20): 10,
            (50, 40): 8,
            (60, 30): 6,
            (40, 35): 4
        }
    )
    
    obj_val2, pattern_usage2 = problem2.solve_column_generation(max_iterations=20)
    
    if pattern_usage2:
        problem2.print_solution(pattern_usage2)
