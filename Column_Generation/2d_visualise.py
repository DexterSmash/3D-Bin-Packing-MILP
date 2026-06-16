import numpy as np
from scipy.optimize import linprog
import pulp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class TwoDStockCutting:
    """
    Solves the 2D cutting stock problem using column generation.
    Uses PuLP for the pricing subproblem with proper non-overlap constraints.
    """
    
    def __init__(self, sheet_length, sheet_width, piece_demands):
        self.L = sheet_length
        self.W = sheet_width
        self.piece_demands = piece_demands
        self.pieces = list(piece_demands.keys())
        self.demands = [piece_demands[p] for p in self.pieces]
        self.n_pieces = len(self.pieces)
        self.patterns = []
        self.pattern_layouts = []  # Store actual x,y positions
        
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                       '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        
        # Start with basic patterns
        for i in range(self.n_pieces):
            pattern = [0] * self.n_pieces
            pattern[i] = 1
            self.patterns.append(pattern)
            # Store simple layout for initial patterns
            self.pattern_layouts.append([(0, 0, self.pieces[i][0], self.pieces[i][1], i)])
    
    def solve_master_problem(self):
        """Solve restricted master problem"""
        n_patterns = len(self.patterns)
        c = np.ones(n_patterns)
        A_ub = -np.array(self.patterns).T
        b_ub = -(np.array(self.demands)+ 1e-6)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        
        if not result.success:
            raise Exception("Master problem failed")
        
        dual_prices = -result.ineqlin.marginals
        return result.fun, result.x, dual_prices
    
    def solve_pricing_problem_milp(self, prices, time_limit=60):
        """
        Solve 2D knapsack pricing using PuLP with ALL constraints from your code
        """
        # Maximum number of each piece type
        max_counts = []
        for i, (piece_len, piece_width) in enumerate(self.pieces):
            max_by_len = self.L // piece_len
            max_by_width = self.W // piece_width
            max_count = min(max_by_len * max_by_width, 12)
            max_counts.append(max_count)
        
        # Create items (individual pieces)
        items = []
        item_to_type = []
        item_lengths = {}
        item_widths = {}
        
        for piece_type in range(self.n_pieces):
            for copy_idx in range(max_counts[piece_type]):
                item_id = len(items)
                items.append(item_id)
                item_to_type.append(piece_type)
                item_lengths[item_id] = self.pieces[piece_type][0]
                item_widths[item_id] = self.pieces[piece_type][1]
        
        n_items = len(items)
        if n_items == 0:
            return 1.0, None, None
        
        # Create PuLP problem
        prob = pulp.LpProblem("2D_Knapsack_Pricing", pulp.LpMaximize)
        
        # Big M
        M = max(self.L, self.W) * 3
        
        # VARIABLES
        # Position variables
        x = pulp.LpVariable.dicts("x", items, lowBound=0, upBound=self.L, cat='Continuous')
        y = pulp.LpVariable.dicts("y", items, lowBound=0, upBound=self.W, cat='Continuous')
        
        # Binary: is item used?
        used = pulp.LpVariable.dicts("used", items, cat='Binary')
        
        # Relative position variables (for non-overlap)
        lr = {}  # i is left of j (i.r <= j.l)
        rl = {}  # i is right of j (j.r <= i.l)
        ab = {}  # i is above j (i.t <= j.b)
        be = {}  # i is below j (j.t <= i.b)
        
        for i in items:
            for j in items:
                if i < j:
                    lr[i, j] = pulp.LpVariable(f"lr_{i}_{j}", cat='Binary')
                    rl[i, j] = pulp.LpVariable(f"rl_{i}_{j}", cat='Binary')
                    ab[i, j] = pulp.LpVariable(f"ab_{i}_{j}", cat='Binary')
                    be[i, j] = pulp.LpVariable(f"be_{i}_{j}", cat='Binary')
        
        # OBJECTIVE: maximize value
        prob += pulp.lpSum([used[i] * prices[item_to_type[i]] for i in items])
        
        # CONSTRAINT c10-c12: Placement within sheet bounds
        # x[i] + length[i] <= L (when used[i] = 1)
        # y[i] + width[i] <= W (when used[i] = 1)
        for i in items:
            l_i = item_lengths[i]
            w_i = item_widths[i]
            
            # When item is used, it must fit in the sheet
            prob += x[i] + l_i * used[i] <= self.L, f"fit_length_{i}"
            prob += y[i] + w_i * used[i] <= self.W, f"fit_width_{i}"
            
            # When not used, position is zero (optional but helps)
            prob += x[i] <= M * used[i], f"pos_x_{i}"
            prob += y[i] <= M * used[i], f"pos_y_{i}"
        
        # CONSTRAINT c14-c19: Relative positioning
        # These ensure proper spatial relationships
        for i in items:
            for j in items:
                if i < j:
                    l_i = item_lengths[i]
                    w_i = item_widths[i]
                    l_j = item_lengths[j]
                    w_j = item_widths[j]
                    
                    # c14: i is left of j means x[i] + l[i] <= x[j]
                    prob += x[i] + l_i <= x[j] + M * (1 - lr[i, j]), f"left_right_{i}_{j}"
                    
                    # c15: i is right of j means x[j] + l[j] <= x[i]
                    prob += x[j] + l_j <= x[i] + M * (1 - rl[i, j]), f"right_left_{i}_{j}"
                    
                    # c16: i is below j means y[i] + w[i] <= y[j]
                    prob += y[i] + w_i <= y[j] + M * (1 - be[i, j]), f"below_above_{i}_{j}"
                    
                    # c17: i is above j means y[j] + w[j] <= y[i]
                    prob += y[j] + w_j <= y[i] + M * (1 - ab[i, j]), f"above_below_{i}_{j}"
        
        # CONSTRAINT c20: Non-overlap - if both items are used, 
        # at least one relative position must be true
        for i in items:
            for j in items:
                if i < j:
                    prob += (lr[i, j] + rl[i, j] + ab[i, j] + be[i, j] >= 
                            used[i] + used[j] - 1), f"no_overlap_{i}_{j}"
        
        # CONSTRAINT c21-c23: Avoid conflicting relative positions
        for i in items:
            for j in items:
                if i < j:
                    # Can't be both left and right
                    prob += lr[i, j] + rl[i, j] <= 1, f"conflict_x_{i}_{j}"
                    # Can't be both above and below
                    prob += ab[i, j] + be[i, j] <= 1, f"conflict_y_{i}_{j}"
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
        prob.solve(solver)
        
        if prob.status == pulp.LpStatusOptimal:
            max_value = pulp.value(prob.objective)
            reduced_cost = 1.0 - max_value
            
            if reduced_cost < -1e-6:
                # Extract pattern and ACTUAL positions
                pattern = [0] * self.n_pieces
                layout = []  # (x, y, length, width, piece_type)
                
                for i in items:
                    if pulp.value(used[i]) > 0.5:
                        piece_type = item_to_type[i]
                        pattern[piece_type] += 1
                        
                        x_pos = pulp.value(x[i])
                        y_pos = pulp.value(y[i])
                        length = item_lengths[i]
                        width = item_widths[i]
                        
                        layout.append((x_pos, y_pos, length, width, piece_type))
                
                return reduced_cost, pattern, layout
            else:
                return reduced_cost, None, None
        else:
            return 1.0, None, None
    
    def solve(self, max_iter=50, pricing_time_limit=120):
        """Main column generation loop"""
        print(f"Sheet: {self.L} x {self.W}")
        print(f"Demands: {self.piece_demands}\n")
        
        for iteration in range(max_iter):
            obj, solution, prices = self.solve_master_problem()
            
            print(f"Iteration {iteration + 1}: obj = {obj:.4f}")
            
            reduced_cost, new_pattern, layout = self.solve_pricing_problem_milp(
                prices, time_limit=pricing_time_limit
            )
            
            print(f"  Reduced cost: {reduced_cost:.6f}")
            
            if new_pattern is None:
                print(f"\nOptimal solution found!")
                print(f"LP objective: {obj:.4f}")
                print(f"Integer bound: {int(np.ceil(obj))}\n")
                
                pattern_usage = {i: usage for i, usage in enumerate(solution) if usage > 1e-6}
                return obj, pattern_usage
            
            self.patterns.append(new_pattern)
            self.pattern_layouts.append(layout)
            print(f"  Added pattern: {new_pattern}\n")
        
        print("Warning: hit iteration limit")
        obj, solution, _ = self.solve_master_problem()
        pattern_usage = {i: usage for i, usage in enumerate(solution) if usage > 1e-6}
        return obj, pattern_usage
    
    def visualize_patterns(self, pattern_usage, save_path=None):
        """Visualize cutting patterns using ACTUAL positions from MILP"""
        patterns_to_show = sorted(pattern_usage.items(), key=lambda x: x[1], reverse=True)
        n_patterns = len(patterns_to_show)
        
        cols = min(3, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_patterns == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_patterns > 1 else [axes]
        
        for idx, (pattern_idx, usage) in enumerate(patterns_to_show):
            ax = axes[idx]
            pattern = self.patterns[pattern_idx]
            layout = self.pattern_layouts[pattern_idx]
            
            # Draw sheet boundary
            ax.add_patch(Rectangle((0, 0), self.L, self.W, 
                                   fill=False, edgecolor='black', linewidth=2))
            
            # Draw pieces using ACTUAL positions from MILP
            for x_pos, y_pos, length, width, piece_type in layout:
                color = self.colors[piece_type % len(self.colors)]
                
                ax.add_patch(Rectangle((x_pos, y_pos), length, width,
                                      facecolor=color, edgecolor='black', 
                                      linewidth=1.5, alpha=0.7))
                
                ax.text(x_pos + length/2, y_pos + width/2,
                       f'{int(length)}x{int(width)}',
                       ha='center', va='center', fontsize=8, weight='bold')
            
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, self.W)
            ax.set_aspect('equal')
            ax.set_title(f'Pattern {pattern_idx} (use {usage:.2f}x)', fontweight='bold')
            ax.set_xlabel('Length')
            ax.set_ylabel('Width')
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def print_solution(self, pattern_usage):
        """Display the cutting plan"""
        print("=" * 60)
        print("CUTTING PLAN")
        print("=" * 60)
        
        total = 0.0
        for idx, usage in pattern_usage.items():
            pattern = self.patterns[idx]
            print(f"\nPattern {idx} - use {usage:.2f} sheets:")
            for i, count in enumerate(pattern):
                if count > 0:
                    dims = self.pieces[i]
                    print(f"  {count}x {dims[0]} x {dims[1]}")
            total += usage
        
        print(f"\nTotal sheets (LP): {total:.2f}")
        print(f"Total sheets (rounded): {int(np.ceil(total))}")

if __name__ == "__main__":
    problem = TwoDStockCutting(
    sheet_length=100,
    sheet_width=100,
    piece_demands={
        (70, 60): 4,
        (50, 40): 8,
        (35, 30): 10,
        (25, 20): 15
    }
)
    obj, patterns = problem.solve(pricing_time_limit=60)
    if patterns:
        problem.print_solution(patterns)
        problem.visualize_patterns(patterns)
