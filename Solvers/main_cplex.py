#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import os
import math
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from PIL import Image
from cplex_packing import create_packing_model, solve_packing_model
from Space_Defragmentation import Bin, Box as SDBox, try_placing_box_with_push
from layerBasedHeuristic import select_best_bin_type, BoxType, generate_feasible_solutions, generate_horizontal_layers, pack_balanced_bins
from peakFillingSlicePushHeuristic import PFSP3DPacker, Box as PFSPBox, Container, Position
from logger import PicklistLogger
from docplex.mp.model import Model


logger = PicklistLogger()

# function to read data from csv file
def prepareData(filename, k):
    sheets_dict = pd.read_csv(filename)
    DFC_data = pd.read_excel('Vmeasure_Data.xlsx', sheet_name='DFC Data')
    int_k = "Incorrect k"
    if (type(k)==int or all(map(str.isdigit, k))):
        int_k = int(k)
    else:
        int_k = k
    sheet = (sheets_dict[(sheets_dict['Picklist Barcode']==k) | (sheets_dict['Picklist Barcode'] == int_k)])
    l = []
    dfc = []
    qty = []

    boolean = False
    K1 = []


    '''
    Find the maximum number of DFCs of each type based on the ratio of sum of
    max. dimensions of all SKUs and the min. dimension of DFCs
    (this is a heuristic way; an upper bound would be better and make the
    mathematical model exact)
    '''


    sumOfSKUMaxDimension = 0
    for index, rows in sheet.iterrows():
        rep = int(rows['Qty'])
        qty.append(rep)
        for i in range(rep):
            l.append([len(l)+1, rows['Material Barcode'],rows['Length'],rows['Width'], rows['Height'], rows['SKU Weight'], rows['Packing Configuration']])
            if (np.isnan(l[-1][-1])):
                l[-1][-1]=0
            sumOfSKUMaxDimension += max(l[-1][2:-1])
    for i in range(2 , len(DFC_data)):
        if(boolean):
            dfc.append([i-1, DFC_data.iloc[i , 0], DFC_data.iloc[i , 1]/10, DFC_data.iloc[i , 2]/10, DFC_data.iloc[i , 3]/10, (DFC_data.iloc[i , 1]*DFC_data.iloc[i , 2])*7.5/100])
        else:
            dfc.append([i-1, DFC_data.iloc[i , 0], DFC_data.iloc[i , 1]/10, DFC_data.iloc[i , 2]/10, DFC_data.iloc[i , 3]/10, 100000.0])
        K1.append([i-1, math.ceil(sumOfSKUMaxDimension/min(dfc[-1][2:-1]))])
    if (len(l)==0):
        print("Picklist Not Found")
        return "Incorrect k", "Incorrect k", "Incorrect k", "Incorrect k", "Incorrect k", "Incorrect k"
    sku_df = pd.DataFrame(l, columns=['ind', 'name','l', 'w', 'h', 'g', 'upright']).set_index('ind')
    dfc_df = pd.DataFrame(dfc, columns=['ind', 'Name', 'L', 'W', 'H', 'G']).set_index('ind')
    # Why do we need the index here? Will some DFCs be excluded here sometimes?
    Kmax = pd.DataFrame(K1, columns=['ind', 'Kmax']).set_index('ind')
    #print(Kmax)
    #print(K1)
    return len(l), len(dfc), Kmax, sku_df, dfc_df, qty

'''
def masterData(k):
    xls = pd.ExcelFile('DLOR Report Template.xlsx')
    sheets_dict = pd.read_excel(xls, sheet_name=None)
    df = sheets_dict['DLOR Report']
    sku = []
    nw = df.loc[df['Delivery']==k]
    for i in range(nw.shape[0]):
        ji = nw.iloc[i]['Transfer order Quantity']
        while (ji>0):
            sku.append([nw.iloc[i]['Length cm'], nw.iloc[i]['Width cm'], nw.iloc[i]['Height cm']])
            ji-=1
    return sku
'''

def solveOpt(orig_n, n, m, Kmax, sku_df, dfc_df, argv, heuristic_solution=None):
    # Create and solve the model using CPLEX
    mdl = create_packing_model(n, m, Kmax, sku_df, dfc_df, heuristic_solution=heuristic_solution)
    print("Mock SolveOpt called")
    # Set CPLEX parameters
    if argv[4].lower() != 'cplex':
        print("Warning: This implementation only supports CPLEX solver. Forcing solver to CPLEX.")
    
    solution = solve_packing_model(mdl, time_limit=int(argv[5]))
    
    # if solution['status'] == 'optimal':
    #     print('\nSolution found!')
    #     print('Objective value:', solution['objective_value'])
    #     print('\nx,y,z coordinates of the SKUs in respective DFCs:')
    #     for i in range(1, n+1):
    #         print(f"SKU {i}: x={solution['x'][i]:.2f}, y={solution['y'][i]:.2f}, z={solution['z'][i]:.2f}")
    # else:
    #     print('No solution found')
    #     return None
    
    # return solution

    if solution['status'] == 'optimal':
        print('\nSolution found!')
        print('Objective value:', solution['objective_value'])
        print('\nx, y, z coordinates of the SKUs in respective DFCs:')
        for i in sorted(solution['x'].keys()):
            try:
                x = solution['x'][i]
                y = solution['y'][i]
                z = solution['z'][i]
                print(f"SKU {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            except KeyError as e:
                print(f" Missing coordinate for SKU {i}: {e}")
    else:
        print('No solution found')
        return None

    return solution

def plot_cube(lower, upper, ax, col, opacity):
    s = []
    for i in range(len(upper)):
        l = [i for i in range(3)]
        l.remove(i)
        for j in range(4):
            cl = upper.copy()
            cl[l[0]]= upper[l[0]] if (j&1)>0 else lower[l[0]]
            cl[l[1]]= upper[l[1]] if (j&2)>0 else lower[l[1]]
            cl2 = cl.copy()
            cl2[i] = lower[i]
            if list(cl2) not in s:
                s.append(list(cl2))
            if list(cl) not in s:
                s.append(list(cl))
            ax.plot3D(*zip(cl2, cl), color = col)
    faces = []
    for i in range(3):
        l = []
        u = []
        for j in s:
            if (upper[i]==j[i]):
                u.append(j)
            if lower[i]==j[i]:
                l.append(j)
        l.sort()
        l[0], l[1] = l[1], l[0]
        u.sort()
        u[0], u[1] = u[1], u[0]
        faces.append(l)
        faces.append(u)
    fc = art3d.Poly3DCollection(faces, alpha = opacity)
    fc.set_color(col)
    ax.add_collection(fc)
    return ax

def visualise(SKU_coordinate, sku, dfc, id, title, argv):
    mypath ='./results/' + argv[1] + '_' + argv[3] + '_' + argv[4] + '_' + argv[5]
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    fig = plt.figure("1")
    ax = fig.add_subplot(projection='3d')

    cmap = plt.cm.get_cmap('hsv', len(sku)+1)

    ax = plot_cube([0,0,0], dfc, ax, 'black', 0.05)
    Images = []
    for x in range(len(sku)):
        ax.set_title(title[x])
        ax = plot_cube(SKU_coordinate[x], np.array(SKU_coordinate[x])+np.array(sku[x]), ax, cmap(x+1), 0.2)
        # fig = plt.figure("1")
        fname = "/" + str(id) + "_" + str(x) + ".png"
        #print(fname, mypath, mypath+fname)
        fig.savefig(mypath+fname)
        print("\npng saved to file: %s"%mypath+fname)
        #Images.append(Image.open(str(id)+"_"+str(x)+".png"))
        Images.append(Image.open(mypath+fname))
    plt.close()
    gifname = "/" + str(id) + str(".gif")
    Images[0].save(mypath+gifname, save_all = True, append_images = Images[1:], duration = 1000, loop = 0)
    print("\ngif saved to file: %s"%mypath+gifname)

# To this:
def run_Space_Defrag_heuristic_pack(boxes, dfc_df):
    heuristic_solution = {'u': {}, 's': {}, 'x': {}, 'y': {}, 'z': {}}
    bins = []
    for j in range(len(dfc_df)):
        dfc = dfc_df.iloc[j]
        bin_inst = Bin(length=int(dfc['L']), width=int(dfc['W']), height=int(dfc['H']))
        bins.append(bin_inst)
    
    for box in boxes:
        placed = False
        for j, b in enumerate(bins):
            if try_placing_box_with_push(b, box):
                heuristic_solution['s'][(box.id, j+1, 1)] = 1
                heuristic_solution['u'][(j+1, 1)] = 1
                heuristic_solution['x'][f"_{box.id}"] = box.position[0]
                heuristic_solution['y'][f"_{box.id}"] = box.position[1]
                heuristic_solution['z'][f"_{box.id}"] = box.position[2]
                placed = True
                break
        if not placed:
            pass

    return heuristic_solution

def layer_based_heuristic_pack(sku_df, dfc_df):
    # Convert dfc_df to bin_types list with volume/weight
    bin_types = []
    for j in range(len(dfc_df)):
        dfc = dfc_df.iloc[j]
        bin_types.append(type('BinType', (), {
            'volume': dfc['L'] * dfc['W'] * dfc['H'],
            'max_weight': dfc['G'],
            'L': dfc['L'], 'W': dfc['W'], 'H': dfc['H']
        })())
    # Convert sku_df to box_types list
    box_types = []
    weights_by_type = {}
    n_list = []
    for i in range(len(sku_df)):
        sku = sku_df.iloc[i]
        box_types.append(BoxType(i, sku['l'], sku['w'], sku['h'], sku['g']))
        weights_by_type[i] = [sku['g']]
        n_list.append(1)
    
    best_bin = select_best_bin_type(box_types, bin_types)
    n_ip = generate_horizontal_layers(box_types, best_bin.L, best_bin.W)
    feasible_solutions = generate_feasible_solutions(best_bin.H, box_types)
    # This demo assumes all SKUs are of unique type—adjust logic if not.
    m, packed_bins = pack_balanced_bins(
    len(box_types),
    best_bin.max_weight,
    n_list,
    feasible_solutions,
    [weights_by_type[i] for i in range(len(box_types))],
    box_types,
    (best_bin.L, best_bin.W, best_bin.H)
    )

    # Now create return structure
    heuristic_solution = {'u': {}, 's': {}, 'x': {}, 'y': {}, 'z': {}}
    for jj, bin_boxes in enumerate(packed_bins):
        heuristic_solution['u'][(jj+1, 1)] = 1
        for box in bin_boxes:
            idx = box['box_type'] + 1
            heuristic_solution['s'][(idx, jj+1, 1)] = 1
            heuristic_solution['x'][f"_{idx}"] = box['position']
            heuristic_solution['y'][f"_{idx}"] = box['position'][1]
            heuristic_solution['z'][f"_{idx}"] = box['position'][2]
    return heuristic_solution


def peak_filling_heuristic_pack(boxes, dfc_df):
    # Assume single container - pick the first DFC or extend for multiple
    dfc = dfc_df.iloc[0]
    container = Container(w=int(dfc['L']), l=int(dfc['W']), h=int(dfc['H']))

    # Convert SKUs to Boxes - mapping column names carefully
    boxes = []
    for i, box in enumerate(boxes):
        box = PFSPBox(id=i+1, length=int(box.length), width=int(box.width), height=int(box.height))
        boxes.append(box)

    # Define slicing ratios, e.g. uniform slicing
    slicing_ratios = [0.2 for _ in range(5)]  # Adjust as needed

    packer = PFSP3DPacker(container, boxes, slicing_ratios)
    placed_boxes = packer.pack_boxes()

    # Prepare solution dictionary in expected format
    heuristic_solution = {'u': {}, 's': {}, 'x': {}, 'y': {}, 'z': {}}
    heuristic_solution['u'][(1,1)] = 1  # Single container/DFC

    for placed in placed_boxes:
        idx = placed.box.id
        heuristic_solution['s'][(idx, 1, 1)] = 1
        pos = placed.pos
        heuristic_solution['x'][f"_{idx}"] = pos.x
        heuristic_solution['y'][f"_{idx}"] = pos.y
        heuristic_solution['z'][f"_{idx}"] = pos.z

    return heuristic_solution


def get_Coordinates(n, m, solution, sku_df):
    sku_coor = []
    orien = []
    axes_orien = []
    arr = ['l', 'w', 'h']
    ind = [0, 1, 2]
    axi = ['x', 'y', 'z']
    
    for i in range(1, n+1):
        # sku_coor.append([solution['x'][i], solution['y'][i], solution['z'][i]])
        key = f"_{i}"
        sku_coor.append([solution['x'][key], solution['y'][key], solution['z'][key]])

        l = []
        nd = []
        for j in range(3):
            for k in range(3):
                # if solution[arr[k]+axi[j]][i] > 0:
                key = f"_{i}"
                if solution[arr[k]+axi[j]][key] > 0:

                    l.append(sku_df.iloc[i-1][arr[k]])
                    nd.append(ind[k])
        axes_orien.append(nd)
        orien.append(l)
    return sku_coor, orien, axes_orien

def preprocess(sku_df, n, k, max_Dimensions, DFC):
    # Convert DataFrame to numpy array for clustering
    X = []
    for i in range(n):
        l = [sku_df.iloc[i]['l'], sku_df.iloc[i]['w'], sku_df.iloc[i]['h']]
        if (sku_df.iloc[i]['upright']==0):
            l.sort()
        X.append(l)
    X = np.array(X)
    
    # Get unique points and their indices
    unique_points, unique_indices = np.unique(X, axis=0, return_inverse=True)
    
    # Adjust k if it's larger than the number of unique points
    actual_k = min(k, len(unique_points))
    
    # Only perform clustering if we have more than one unique point
    if len(unique_points) > 1:
        kMeans = KMeans(n_clusters=actual_k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
        kMeans.fit(unique_points)
        Sup_SKUs = kMeans.cluster_centers_
        x = kMeans.predict(unique_points)[unique_indices]
    else:
        # If we have only one unique point, create k copies of it
        Sup_SKUs = np.array([unique_points[0]] * actual_k)
        x = unique_indices
    
    # Convert cluster labels to list
    x = list(x)
    
    # Get the actual number of clusters found
    k_bar = len(set(x))
    Sup_SKUs = Sup_SKUs[0:k_bar]
    
    # If we need more clusters than found, duplicate the largest cluster
    while (k_bar < k):
        countSSku = []
        for i in range(k_bar):
            countSSku.append(x.count(i))
        maxSized_SSku = countSSku.index(max(countSSku))
        sizeOfmaxsized_SSku = max(countSSku)
        Sup_SKUs = np.concatenate((np.array(Sup_SKUs), np.array([Sup_SKUs[maxSized_SSku]])), axis=0)
        num_reassigned_SKU = 0
        for i in range(n):
            if x[i] == maxSized_SSku:
                x[i] = len(Sup_SKUs) - 1
                num_reassigned_SKU = num_reassigned_SKU + 1
            if num_reassigned_SKU >= int(sizeOfmaxsized_SSku/2):
                break
        k_bar = k_bar + 1

    temp = []
    p = []
    for i in range(len(Sup_SKUs)):
        p.append([1, x.count(i)])

    l = []
    plan = []
    for i in range(len(Sup_SKUs)):
        weight = 0
        dim = [0,0,0]
        ci = 0
        uprightConstraint = 0
        cur = []
        for j in range(n):
            if (x[j]==i):
                cur+=[j]
                weight+=sku_df.iloc[j]['g']
                curSKU = [sku_df.iloc[j]['l'], sku_df.iloc[j]['w'], sku_df.iloc[j]['h']]
                if (sku_df.iloc[j]['upright']==0):
                    curSKU.sort()
                else:
                    uprightConstraint = 1
                dim[0]+=curSKU[0]
                dim[1] = max(curSKU[1], dim[1])
                dim[2] = max(curSKU[2], dim[2])
                ci+=1
                if ci==p[x[j]][1]//p[x[j]][0]:
                    p[x[j]][1]-=p[x[j]][1]//p[x[j]][0]
                    p[x[j]][0]-=1
                    l.append([len(l)+1, dim[0], dim[1], dim[2], weight, uprightConstraint])
                    plan.append(cur)
                    cur = []
                    weight = 0
                    dim = [0,0,0]
                    ci = 0
                    uprightConstraint = 0
        if weight>0:
            l.append([len(l)+1, dim[0], dim[1], dim[2], weight, uprightConstraint])
            plan.append(cur)
            uprightConstraint = 0

    superSKUs = pd.DataFrame(l, columns=['ind', 'l', 'w', 'h', 'g', 'upright']).set_index('ind')
    return len(l), superSKUs, plan


def validate_solution(solution, n, m, sku_df, dfc_df):
    """Validate that the packing solution is feasible"""
    validation_results = {}
    
    # Check 1: All SKUs are assigned to a DFC
    assigned_skus = set()
    for i in range(1, n+1):
        for j in range(1, m+1):
            for k in range(1, int(Kmax.iloc[j-1].iloc[0])+1):
                if solution['s'].get((i, j, k), 0) > 0:
                    assigned_skus.add(i)
    
    all_skus = set(range(1, n+1))
    unassigned = all_skus - assigned_skus
    
    validation_results['all_skus_assigned'] = {
        'passed': len(unassigned) == 0,
        'details': f"Unassigned SKUs: {unassigned}" if unassigned else "All SKUs assigned"
    }
    
    # Check 2: Coordinates are within DFC boundaries
    boundary_violations = []
    for i in range(1, n+1):
        for j in range(1, m+1):
            for k in range(1, int(Kmax.iloc[j-1].iloc[0])+1):
                if solution['s'].get((i, j, k), 0) > 0:
                    sku = sku_df.iloc[i-1]
                    dfc = dfc_df.iloc[j-1]
                    
                    # Get position
                    key = f"_{i}"
                    x = solution['x'].get(key, 0)
                    y = solution['y'].get(key, 0)
                    z = solution['z'].get(key, 0)
                    
                    # Check orientation and dimensions
                    l_dim = 0
                    w_dim = 0
                    h_dim = 0
                    
                    if solution.get('lx', {}).get(key, 0) > 0:
                        l_dim = sku['l']
                    elif solution.get('wx', {}).get(key, 0) > 0:
                        l_dim = sku['w']
                    elif solution.get('hx', {}).get(key, 0) > 0:
                        l_dim = sku['h']
                    
                    if solution.get('ly', {}).get(key, 0) > 0:
                        w_dim = sku['l']
                    elif solution.get('wy', {}).get(key, 0) > 0:
                        w_dim = sku['w']
                    elif solution.get('hy', {}).get(key, 0) > 0:
                        w_dim = sku['h']
                    
                    if solution.get('lz', {}).get(key, 0) > 0:
                        h_dim = sku['l']
                    elif solution.get('wz', {}).get(key, 0) > 0:
                        h_dim = sku['w']
                    elif solution.get('hz', {}).get(key, 0) > 0:
                        h_dim = sku['h']
                    
                    # Check boundaries
                    if x + l_dim > dfc['L'] or y + w_dim > dfc['W'] or z + h_dim > dfc['H']:
                        boundary_violations.append(f"SKU {i} exceeds DFC {j} boundaries")
    
    validation_results['within_boundaries'] = {
        'passed': len(boundary_violations) == 0,
        'details': '; '.join(boundary_violations) if boundary_violations else "All SKUs within boundaries"
    }
    
    return validation_results

def main(argc, argv):
    # Validate command line arguments
    if argc < 6:
        print("Usage: python main_cplex.py <order_id> <picklist_file.csv> <num_superSKUs> <solver> <time_limit>")
        return None
        
    print('Order id: ', argv[1])
    print('Picklist file (.csv): ', argv[2])
    print('Number of superSKUs: ', argv[3])
    print('Solver: ', argv[4])
    print('Time limit: ', argv[5])

    n, m, Kmax, sku_df, dfc_df, qty = prepareData(argv[2], argv[1])

    picklist_id = logger.auto_log_start(list(range(n)))
    print('\nNumber of different SKUs: ', n)
    print('Number of different DFCs: ', m)
    print('\n Super SKU Details:')
    print(sku_df)

    nSKUs = int(argv[3])
    initial_dfcdf = dfc_df
    initial_skudf = sku_df
    max_Dimensions = [0, 0, 0]
    dfc_vol = 0
    sku_vol = 0
    DFCs = []
    l = [dict() for i in range(len(qty))]
    name = []
    index = 0
    for i in range(len(qty)):
        name.append(sku_df.iloc[index]['name'])
        index += qty[i]
    ind = [0]*(sum(qty))
    cur = 0
    for i in range(len(qty)):
        ind[cur] = i
        cur+=qty[i]
    for i in range(1, len(ind)):
        ind[i] = max(ind[i], ind[i-1])

    for j in range(m):
        DFCs.append([dfc_df.iloc[j]['L'], dfc_df.iloc[j]['W'], dfc_df.iloc[j]['H']])
        max_Dimensions[0] = max(max_Dimensions[0], dfc_df.iloc[j]['L'])
        max_Dimensions[1] = max(max_Dimensions[1], dfc_df.iloc[j]['W'])
        max_Dimensions[2] = max(max_Dimensions[2], dfc_df.iloc[j]['H'])
    
    k = min(n, nSKUs)
    for i in range(0, n):
        sku_vol+=(sku_df.iloc[i]['l']*sku_df.iloc[i]['w']*sku_df.iloc[i]['h'])
    
    plan = [[i] for i in range(n)]
    DataPreProcessed = False
    new_n = n
    
    if (nSKUs > 0 and n > nSKUs):
        DataPreProcessed = True
        print('Number of SuperSKUs to be generated: ', nSKUs)
        new_n, sku_df, plan = preprocess(sku_df, n, k, max_Dimensions, DFCs)
    
    if DataPreProcessed == True:
        print('\n Super SKU Details:')
        print(sku_df)
    
    heuristic_solution = None
    if nSKUs == 0:
        print('Running heuristic pre-packing...')
        if len(argv) > 6:
            heuristic_type= argv[6].lower()
        else:
            heuristic_type= ''

        if heuristic_type == 'space_defrag':
            boxes_for_defrag = [SDBox(id=i+1, length=int(row['l']), width=int(row['w']), height=int(row['h'])) for i, row in sku_df.iterrows()]
            heuristic_solution = run_Space_Defrag_heuristic_pack(boxes_for_defrag, dfc_df)
        elif heuristic_type == 'peakfilling':
            boxes_for_pfsp = [PFSPBox(id=i+1, w=int(row['w']), l=int(row['l']), h=int(row['h'])) for i, row in sku_df.iterrows()]
            heuristic_solution = peak_filling_heuristic_pack(boxes_for_pfsp, dfc_df)
        elif heuristic_type == 'layer':
            heuristic_solution = layer_based_heuristic_pack(sku_df, dfc_df)
        else:
            print('some error')

        print("Output from heuristic_solution:", heuristic_solution)
        if heuristic_solution is None or not heuristic_solution.get('s'):
            print("Heuristic failed or returned no valid packing. Proceeding without heuristic.")
            heuristic_solution = None

    print('\nSolving the assignment and packing mathematical model\n')
    print("heur soln is NOne?", heuristic_solution is None)
    solution = solveOpt(n, new_n, m, Kmax, sku_df, dfc_df, argv, heuristic_solution=heuristic_solution)
    
    logger.auto_log_result(picklist_id, solution)

    if solution is None:
        print("No solution found")
        return None

    validation_results= validate_solution(solution, new_n, m, sku_df, dfc_df)
    logger.log_validation_results(picklist_id, validation_results)

    if solution and solution.get('first_incumbent'):
        logger.log_incumbent_comparison(
            picklist_id,
            solution['first_incumbent'],
            solution['objective_value'],
            solution['total_time']
        )

    sku_coor, orientation, axes_orien = get_Coordinates(new_n, m, solution, sku_df)

    suggestedDFC = []
    DFC_co = dict()
    bubble_wrap = dict()
    dfc_count = {}
    
    for j in range(1, m+1):
        for k in range(1, int(Kmax.iloc[j-1].iloc[0])+1):
            if solution['u'][(j, k)] > 0:
                dfc_vol+=(dfc_df.iloc[j-1]['L']*dfc_df.iloc[j-1]['W']*dfc_df.iloc[j-1]['H'])
                dfc_name = dfc_df.iloc[j-1]['Name']
                
                if dfc_name in dfc_count:
                    dfc_count[dfc_name] += 1
                else:
                    dfc_count[dfc_name] = 1
                
                dfc_instance_name = f"{dfc_name}_{dfc_count[dfc_name]}"
                suggestedDFC.append(dfc_name)
                
                currentDFC = [dfc_df.iloc[j-1]['L'], dfc_df.iloc[j-1]['W'], dfc_df.iloc[j-1]['H']]
                currentSKU_Coordinates = []
                currentSKU_Dimensions = []
                title = []
                wt = 0
                bubble = 2
                vol = dfc_df.iloc[j-1]['L']*dfc_df.iloc[j-1]['W']*(7.5)/100
                if (vol>=wt):
                    bubble-=1
                bubble_wrap[dfc_instance_name] = bubble

                for i in range(0, new_n):
                    if solution['s'][(i+1, j, k)] > 0:
                        wt+=sku_df.iloc[i]['g']
                        coor = []
                        if (DataPreProcessed == False):
                            currentSKU_Dimensions.append(orientation[i])
                            currentSKU_Coordinates.append(sku_coor[i])
                            title.append(initial_skudf.iloc[i]['name'])
                        else:
                            coor = copy.deepcopy(sku_coor[i])
                        for ski in plan[i]:
                            if DataPreProcessed:
                                edges = [initial_skudf.iloc[ski]['l'], initial_skudf.iloc[ski]['w'], initial_skudf.iloc[ski]['h']]
                                edges.sort()
                                wh = [0 for _ in range(3)]
                                title.append(initial_skudf.iloc[ski]['name'])
                                currentSKU_Dimensions.append(wh)
                                currentSKU_Coordinates.append(coor.copy())
                                for ax in range(3):
                                    if (axes_orien[i][ax]==0):
                                        coor[ax]+=edges[0]
                                    wh[ax] = edges[axes_orien[i][ax]]
                                    
                            if dfc_instance_name in l[ind[ski]]:
                                l[ind[ski]][dfc_instance_name] += 1
                            else:
                                l[ind[ski]][dfc_instance_name] = 1

                visualise(currentSKU_Coordinates, currentSKU_Dimensions, currentDFC,
                        f"{argv[1]}_{dfc_instance_name}", title, sys.argv)

    try:
        sheets_dict = pd.read_csv(argv[2])
    except FileNotFoundError:
        print(f"Error: Could not find the picklist file '{argv[2]}'")
        return None
    if ('DFCs Used' not in sheets_dict):
        sheets_dict['DFCs Used'] = pd.Series([0 for _ in range(len(sheets_dict['Length']))], index = sheets_dict.index)
    sheet = (sheets_dict.loc[sheets_dict['Picklist Barcode']==argv[1]])
    print(sheet)

    print("\nSuggested DFCs with required copies:")
    printed = set()
    for i in suggestedDFC:
        if (i not in printed):
            print(str(dfc_count[i])+" "+("copy" if dfc_count[i]==1 else "copies")+" of the DFC "+i)
            printed.add(i)

    o = []
    print("\nPacking plan,")
    for i in range(len(l)):
        temp = []
        for j in l[i]:
            temp+=[str(l[i][j])+(" quantity of SKU:" if l[i][j]==1 else " quantities of SKU:")+str(name[i])+" in DFC "+j]
        print(', '.join(map(str, temp)))
        o.append(', '.join(map(str, temp)))
    
    cur = 0
    for index, rows in sheet.iterrows():
        sheets_dict.loc[index, 'DFCs Used'] = o[cur]
        cur+=1
    
    sheets_dict.to_csv(argv[2], index = False)
    print("\nVolumetric Efficiency:", str(round(100*sku_vol/max(1, dfc_vol), 3))+"%")
    return sheets_dict

if __name__ == "__main__":
    try:
        main(len(sys.argv), sys.argv)
    except Exception as e:
        print(e)
        raise 
