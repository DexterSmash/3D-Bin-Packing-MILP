from docplex.mp.model import Model
import pandas as pd
import numpy as np
import cplex
from cplex.callbacks import IncumbentCallback
from cplex import Cplex

def unpack_scalar(val):
        if isinstance(val, (tuple, list, np.ndarray)):
            return float(val[0])
        else:
            return float(val)

def create_packing_model(n, m, Kmax, sku_df, dfc_df, fixOrient=None, heuristic_solution=None):
    # Create the model
    mdl = Model("SKU_Packing")

    # PARAMETERS
    coordPenalty = 1
    
    # Convert DataFrames to dictionaries for easier access
    L = {j: dfc_df.iloc[j-1]['L'] for j in range(1, m+1)}
    W = {j: dfc_df.iloc[j-1]['W'] for j in range(1, m+1)}
    H = {j: dfc_df.iloc[j-1]['H'] for j in range(1, m+1)}
    G = {j: dfc_df.iloc[j-1]['G'] for j in range(1, m+1)}
    
    l = {i: sku_df.iloc[i-1]['l'] for i in range(1, n+1)}
    w = {i: sku_df.iloc[i-1]['w'] for i in range(1, n+1)}
    h = {i: sku_df.iloc[i-1]['h'] for i in range(1, n+1)}
    g = {i: sku_df.iloc[i-1]['g'] for i in range(1, n+1)}
    
    if fixOrient is None:
        fixOrient = {i: sku_df.iloc[i-1]['upright'] for i in range(1, n+1)}

    # Sets
    J = range(1, m+1)
    I = range(1, n+1)
    K = {j: range(1, int(Kmax.iloc[j-1].iloc[0])+1) for j in J}
    M = max(max(L.values()), max(W.values()), max(H.values()))

    # VARIABLES
    x = mdl.continuous_var_dict(I, lb=0, ub=M, name='x')
    y = mdl.continuous_var_dict(I, lb=0, ub=M, name='y')
    z = mdl.continuous_var_dict(I, lb=0, ub=M, name='z')

    lx = mdl.binary_var_dict(I, name='lx')
    ly = mdl.binary_var_dict(I, name='ly')
    lz = mdl.binary_var_dict(I, name='lz')
    wx = mdl.binary_var_dict(I, name='wx')
    wy = mdl.binary_var_dict(I, name='wy')
    wz = mdl.binary_var_dict(I, name='wz')
    hx = mdl.binary_var_dict(I, name='hx')
    hy = mdl.binary_var_dict(I, name='hy')
    hz = mdl.binary_var_dict(I, name='hz')

    s = {(i, j, k): mdl.binary_var(name=f's_{i}_{j}_{k}') 
         for i in I for j in J for k in K[j]}
    u = {(j, k): mdl.binary_var(name=f'u_{j}_{k}') 
         for j in J for k in K[j]}
    
    # Relative position variables
    lr = {}
    rl = {}
    bt = {}
    tb = {}
    bf = {}
    fb = {}

    

    for i in I:
        for i_dash in I:
            if i < i_dash:
                lr[i, i_dash] = mdl.binary_var(name=f'lr_{i}_{i_dash}')
                rl[i, i_dash] = mdl.binary_var(name=f'rl_{i}_{i_dash}')
                bt[i, i_dash] = mdl.binary_var(name=f'bt_{i}_{i_dash}')
                tb[i, i_dash] = mdl.binary_var(name=f'tb_{i}_{i_dash}')
                bf[i, i_dash] = mdl.binary_var(name=f'bf_{i}_{i_dash}')
                fb[i, i_dash] = mdl.binary_var(name=f'fb_{i}_{i_dash}')

    if heuristic_solution:
        mip_start_dict = {}
        for (sku_id, bin_id, copy_id), assigned in heuristic_solution['s'].items():
            if assigned > 0 and (sku_id, bin_id, copy_id) in s:
                mip_start_dict[s[(sku_id, bin_id, copy_id)]] = int(assigned)
            elif (sku_id, bin_id, copy_id) not in s:
                print("IGNORED:", (sku_id, bin_id, copy_id))
                
        for sku_id in range(1, n+1):
            key_x = f"_{sku_id}"
            if key_x in heuristic_solution['x']:
                mip_start_dict[x[sku_id]] = unpack_scalar(heuristic_solution['x'][key_x])
            else:
                print(f"IGNORED missing key in x: {key_x}")

            key_y = f"_{sku_id}"
            if key_y in heuristic_solution['y']:
                mip_start_dict[y[sku_id]] = unpack_scalar(heuristic_solution['y'][key_y])
            else:
                print(f"IGNORED missing key in y: {key_y}")

            key_z = f"_{sku_id}"
            if key_z in heuristic_solution['z']:
                mip_start_dict[z[sku_id]] = unpack_scalar(heuristic_solution['z'][key_z])
            else:
                print(f"IGNORED missing key in z: {key_z}")


    
        
    # OBJECTIVE
    '''
    mdl.minimize(
        mdl.sum(u[j, k] * L[j] * W[j] * H[j] for j in J for k in K[j]) -
        mdl.sum(l[i] * w[i] * h[i] for i in I)
        + coordPenalty * mdl.sum(x[i] + y[i] + z[i] for i in I)
    )
    '''

    # Objective 1: Minimize volume unused (priority 1)
    objVol_expr = mdl.sum(u[j, k] * L[j] * W[j] * H[j] for j in J for k in K[j]) - mdl.sum(l[i] * w[i] * h[i] for i in I)

    # Objective 2: Minimize coordinate deviation (priority 2)
    objDev_expr = coordPenalty * mdl.sum(x[i] + y[i] + z[i] for i in I)

    # # Set multi-objective with priorities
    mdl.set_multi_objective(
        
        exprs=[objVol_expr, objDev_expr],
        priorities=[2, 1],
        # senses=[mdl.minimize, mdl.minimize]
        sense='min'
    )


    # CONSTRAINTS

    # c0 symmetry breaking
    for j in J:
        for k in range(1, int(Kmax.iloc[j-1].iloc[0])-1):
            mdl.add_constraint(u[j, k] >= u[j, k+1])

    # c1: fix orientation
    for i in I:
        if fixOrient.get(i, 0) == 1:
            mdl.add_constraint(lx[i] == 1)

    # c2: assignment
    for i in I:
        mdl.add_constraint(mdl.sum(s[i, j, k] for j in J for k in K[j]) == 1)

    # c3: active DFC
    for i in I:
        for j in J:
            for k in K[j]:
                mdl.add_constraint(u[j, k] >= s[i, j, k])

    # c4-c9: orientation constraints
    for i in I:
        mdl.add_constraints([
            lx[i] + ly[i] + lz[i] == 1,
            wx[i] + wy[i] + wz[i] == 1,
            hx[i] + hy[i] + hz[i] == 1,
            lx[i] + wx[i] + hx[i] == 1,
            ly[i] + wy[i] + hy[i] == 1,
            lz[i] + wz[i] + hz[i] == 1,
        ])

    # c10-c12: placement within DFCs
    for i in I:
        for j in J:
            for k in K[j]:
                mdl.add_constraints([
                    x[i] + lx[i]*l[i] + wx[i]*w[i] + hx[i]*h[i] <= L[j] + (1 - s[i, j, k]) * M,
                    y[i] + ly[i]*l[i] + wy[i]*w[i] + hy[i]*h[i] <= W[j] + (1 - s[i, j, k]) * M,
                    z[i] + lz[i]*l[i] + wz[i]*w[i] + hz[i]*h[i] <= H[j] + (1 - s[i, j, k]) * M,
                ])

    # c13: weight capacity
    for j in J:
        for k in K[j]:
            mdl.add_constraint(
                mdl.sum(s[i, j, k] * g[i] for i in I) <= G[j]
            )

    # c14-c19: relative positioning
    for i in I:
        for i_dash in I:
            if i < i_dash:
                mdl.add_constraints([
                    x[i] + lx[i]*l[i] + wx[i]*w[i] + hx[i]*h[i] <= x[i_dash] + (1 - lr[i, i_dash]) * M,
                    x[i_dash] + lx[i_dash]*l[i_dash] + wx[i_dash]*w[i_dash] + hx[i_dash]*h[i_dash] <= x[i] + (1 - rl[i, i_dash]) * M,
                    y[i] + ly[i]*l[i] + wy[i]*w[i] + hy[i]*h[i] <= y[i_dash] + (1 - bf[i, i_dash]) * M,
                    y[i_dash] + ly[i_dash]*l[i_dash] + wy[i_dash]*w[i_dash] + hy[i_dash]*h[i_dash] <= y[i] + (1 - fb[i, i_dash]) * M,
                    z[i] + lz[i]*l[i] + wz[i]*w[i] + hz[i]*h[i] <= z[i_dash] + (1 - bt[i, i_dash]) * M,
                    z[i_dash] + lz[i_dash]*l[i_dash] + wz[i_dash]*w[i_dash] + hz[i_dash]*h[i_dash] <= z[i] + (1 - tb[i, i_dash]) * M,
                ])

    # c20: non-overlap constraint if both SKUs are in same DFC
    for i in I:
        for i_dash in I:
            if i < i_dash:
                for j in J:
                    for k in K[j]:
                        mdl.add_constraint(
                            lr[i, i_dash] + rl[i, i_dash] + bt[i, i_dash] + tb[i, i_dash] +
                            bf[i, i_dash] + fb[i, i_dash] >= s[i, j, k] + s[i_dash, j, k] - 1
                        )

    # c21-c23: avoid conflicting relative positions
    for i in I:
        for i_dash in I:
            if i < i_dash:
                mdl.add_constraints([
                    lr[i, i_dash] + rl[i, i_dash] <= 1,
                    bt[i, i_dash] + tb[i, i_dash] <= 1,
                    bf[i, i_dash] + fb[i, i_dash] <= 1,
                ])

    return mdl

import os
import time
from docplex.mp.model import Model
from docplex.mp.progress import ProgressListener

class FileProgress(ProgressListener):
    """Logs node-level MIP progress safely to a file."""
    def __init__(self, log_file):
        super().__init__()
        self.log_file = open(log_file, "w")
        self.log_file.write("Node\tBestObj\tBestBound\tGap\tTime(s)\n")
        self.log_file.flush()

    def notify_progress(self, pdata):
        try:
            if hasattr(pdata, "has_incumbent") and pdata.has_incumbent:
                best_obj = getattr(pdata, "current_objective", None)
                best_bound = getattr(pdata, "best_bound", None)
                gap = getattr(pdata, "mip_gap", None)
                nodes = getattr(pdata, "current_nb_nodes", None)
                t = getattr(pdata, "time", 0.0)
                self.log_file.write(
                    f"{nodes}\t{best_obj}\t{best_bound}\t{gap}\t{t:.2f}\n"
                )
                self.log_file.flush()
        except Exception as e:
            # Catch and log but don't crash
            self.log_file.write(f"Exception in listener: {e}\n")
            self.log_file.flush()

    def __del__(self):
        try:
            self.log_file.close()
        except:
            pass

def solve_packing_model(mdl, time_limit=None, results_path=None):
    if results_path is None:
        results_path = os.getcwd()

    cpx = mdl.get_cplex()
    log_dir = os.path.join(results_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    mdl.export_as_lp(os.path.join(log_dir, "model.lp"))
    cpx.write(os.path.join(log_dir, "model.sav"))

    # Set detailed logging and tight parameters
    cpx.parameters.read.datacheck = 1
    cpx.parameters.mip.display = 4
    cpx.parameters.mip.interval = 1
    cpx.parameters.output.clonelog = 0

    if time_limit:
        cpx.parameters.timelimit = time_limit
        cpx.parameters.mip.tolerances.mipgap = 0.0

    # Attach safe progress listener
    progress_log = os.path.join(log_dir, "detailed_mip_progress.log")
    mdl.add_progress_listener(FileProgress(progress_log))

    start = time.time()
    solution = mdl.solve(log_output=os.path.join(log_dir, "cplex_verbose.log"))
    total_time = time.time() - start

    if solution:
        result = {
            'status': 'optimal',
            'objective_value': solution.objective_value,
            'total_time': total_time
        }
        
        # Extract all variable values safely
        var_types = ['x', 'y', 'z', 's', 'u', 'lx', 'ly', 'lz', 'wx', 'wy', 'wz', 'hx', 'hy', 'hz']
        
        for var_type in var_types:
            try:
                vars_dict = {}
                for var in mdl.iter_variables():
                    if var.name.startswith(var_type):
                        if var_type in ['s', 'u']:
                            parts = var.name.split('_')
                            if len(parts) >= 3:
                                if var_type == 's':
                                    key = (int(parts[1]), int(parts[2]), int(parts[3]))
                                else:
                                    key = (int(parts[1]), int(parts[2]))
                                vars_dict[key] = var.solution_value
                        else:
                            key = int(var.name[len(var_type):]) if var.name[len(var_type):].isdigit() else var.name[len(var_type):]
                            vars_dict[key] = var.solution_value
                
                result[var_type] = vars_dict
            except Exception as e:
                print(f"Warning: Could not extract {var_type} variables: {e}")
                result[var_type] = {}
         
        return result
    else:
        return {'status': 'infeasible', 'total_time': total_time}
