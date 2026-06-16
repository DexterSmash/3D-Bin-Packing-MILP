import sys
import math
import copy
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from main_cplex import prepareData
import itertools


def select_best_bin_type(boxes, bin_types):
    ''' Option A: Choose the smallest DFC such that, 
        bin.volume >= total_box_volume * volume_buffer 
        where volume_buffer is a multiplier (e.g., 1.1 to allow for packing inefficiency)

        Option B: Pick the bin with the lowest score (α and β are hyperparameters):
        score = α * (unused_volume / bin_volume) + β * (unused_weight / bin_weight)
    
        Option C: Match generated layers (from Stage 1) to bin dimensions.
        Prefer bins that closely match the base area of promising layers.
    '''
    best_bin = None
    best_score = float('inf')

    total_volume = sum(b.volume for b in boxes)
    total_weight = sum(b.weight for b in boxes)

    # volume_buffer = 1.1
    # for bin_type in bin_types:
    #     if bin_type.volume < total_volume * volume_buffer:
    #         continue

    #     else:
    #         if best_bin is None or bin_type.volume < best_bin.volume:
    #             best_bin = bin_type 

    for bin_type in bin_types:
        if bin_type.volume < total_volume or bin_type.max_weight < total_weight:
            continue

        volume_fit = (bin_type.volume - total_volume) / bin_type.volume
        weight_fit = (bin_type.max_weight - total_weight) / bin_type.max_weight
        score = 0.6 * volume_fit + 0.4 * weight_fit

        if score < best_score:
            best_score = score
            best_bin = bin_type

    return best_bin

# Step 1: Generate packing layers that fit within the base of a selected bin

from itertools import permutations
from math import floor

@dataclass
class Box:
    id: int
    length: int
    width: int
    height: int
    weight: float = 0
    position: Optional[Tuple[int, int, int]] = None
    orientation: Optional[Tuple[int, int, int]] = None  # to store orientation if needed

    def volume(self):
        return self.length * self.width * self.height

@dataclass
class Bin:
    length: int
    width: int
    height: int
    max_weight: float = float('inf')
    boxes: List[Box] = field(default_factory=list)

    def add_box(self, box: Box, position: Tuple[int, int, int], orientation: Tuple[int,int,int]):
        box.position = position
        box.orientation = orientation
        self.boxes.append(box)

    def current_weight(self):
        return sum(box.weight for box in self.boxes)


def get_orientations(box_dims):
    l, w, h = box_dims
    return [
        (l, w, h),
        (l, h, w),
        (w, l, h),
        (w, h, l),
        (h, l, w),
        (h, w, l)
    ]

def generate_layers(sku_df, bin_dims):
    layers = []  # Each element: dict with keys: 'box_type', 'orientation', 'count', 'layer_height', 'layer_dims'
    t = len(sku_df)
    L, D, H = bin_dims  # bin dimensions (lwngth, depth, height)
    for i in range(t):
        box = sku_df.iloc[i]
        box_dims = (box['l'], box['w'], box['h'])
        box_weight = box['g']
        for ori in get_orientations(box_dims):
            or_l, or_w, or_h = ori
            max_in_length = math.floor(L / or_l)
            max_in_depth = math.floor(D / or_w)
            count_in_layer = max_in_length * max_in_depth
            if count_in_layer > 0:
                layer_info = {
                    'box_type': i,
                    'orientation': ori,
                    'count': count_in_layer,
                    'layer_height': or_h,
                    'box_weight': box_weight,
                    'layer_volume': count_in_layer * or_l * or_w * or_h
                }
                layers.append(layer_info)
    return layers

# Stage 2: Generate feasible stacking solutions of layers per bin height
def generate_feasible_solutions(layers, bin_height):
    feasible_solutions = []
    max_layers_stack = 2  # example limit
    for combination in itertools.combinations(layers, max_layers_stack):
        total_height = sum(layer['layer_height'] for layer in combination)
        if total_height <= bin_height:
            feasible_solutions.append(list(combination))
    # Also include single-layer packings
    for layer in layers:
        if len(feasible_solutions) > 20000:
            break
        elif layer['layer_height'] <= bin_height:
            feasible_solutions.append([layer])

    return feasible_solutions

# Stage 3: Pack boxes into bins via best feasible solution candidates
def pack_bins(box_counts, layers, feasible_solutions, sku_df, bin_dims, max_weight):
    bins = []
    remaining_counts = box_counts.copy()
    total_boxes = sum(box_counts)
    print(f"Starting packing with {sum(remaining_counts)} boxes remaining.")
    print(f"Number of feasible solutions: {len(feasible_solutions)}")

    while sum(remaining_counts) > 0:
        # Filter feasible solutions respecting current remaining boxes and weight
        candidates = []
        for sol in feasible_solutions:
            # Check if enough boxes remaining for each layer in solution
            enough_boxes = True
            total_weight = 0
            total_volume = 0
            for layer in sol:
                box_type = layer['box_type']
                needed = min(layer['count'],remaining_counts[box_type])
                if needed==0:
                    enough_boxes=False
                    break
                if remaining_counts[box_type] < needed:
                    enough_boxes = False
                    break
                total_weight += needed * layer['box_weight']
                total_volume += layer['layer_volume']
            if enough_boxes:
                candidates.append((sol, total_volume))
        if not candidates:
            # No feasible solution fits remaining, break loop
            break
        
        print(f"Checking candidate solution with {len(sol)} layers")
# Select best candidate: max volume used
        best_sol, _ = max(candidates, key=lambda x: x[1])
        bin_inst = Bin(length=bin_dims[0], width=bin_dims[1], height=bin_dims[2], max_weight=max_weight)
        z_pos = 0
        for layer in best_sol:
            box_type = layer['box_type']
            ori = layer['orientation']
            count = min(layer['count'], remaining_counts[box_type])
            or_l, or_w, or_h = ori
            # Place boxes in grid in this layer
            num_length = math.floor(bin_inst.length / or_l)
            num_width = math.floor(bin_inst.width / or_w)
            for b in range(count):
                x = (b % num_length) * or_l
                y = (b // num_length) * or_w
                if remaining_counts[box_type] == 0:
                    break
                box_data = sku_df.iloc[box_type]
                box = Box(
                    id=None,
                    length=or_l,
                    width=or_w,
                    height=or_h,
                    weight=box_data['g']
                )
                bin_inst.add_box(box, (x, y, z_pos), ori)
                remaining_counts[box_type] -= 1
            z_pos += or_h
        bins.append(bin_inst)
    # return bins and remaining_counts for unpacked info
    return bins, remaining_counts

def main(argc, argv):
    if argc < 6:
        print("Usage: python Layer_based_main.py <order_id> <picklist_file.csv> <num_superSKUs> <solver> <time_limit>")
        return
    order_id = argv[1]
    picklist_file = argv[2]
    n_super_skus = int(argv[3])
    solver = argv[4]
    time_limit = argv[5]

    n, m, Kmax, sku_df, dfc_df, qty = prepareData(picklist_file, order_id)

    print(f"Number of SKUs: {n}, Number of DFCs: {m}")

    # Define bin dimension and max weight for packing (assumed all bins same size/weight from first DFC)
    bin_length = int(dfc_df.iloc[0]['L'])
    bin_width = int(dfc_df.iloc[0]['W'])
    bin_height = int(dfc_df.iloc[0]['H'])
    max_weight = float(dfc_df.iloc[0].get('Max Weight', 1e10))  # Set per paper or input - you can extend to read from dfc_df


    # Count of boxes per SKU type, can be from qty or 1:1
    box_counts = [qty[i] if i < len(qty) else 0 for i in range(n)]

    # Stage 1: generate layers
    layers = generate_layers(sku_df, (bin_length, bin_width, bin_height))
    for layer in layers:
        print("Layer:", layer)

    # Stage 2: generate solutions
    feasible_solutions = generate_feasible_solutions(layers, bin_height)

    # Stage 3: pack bins
    bins, remaining_counts = pack_bins(box_counts, layers, feasible_solutions, sku_df, (bin_length, bin_width, bin_height), max_weight)

    total_packed_volume = sum(sum(box.volume() for box in bin.boxes) for bin in bins)
    total_bin_volume = sum(bin.length * bin.width * bin.height for bin in bins if bin.boxes)
    efficiency = 100 * total_packed_volume / max(1, total_bin_volume)

    print(f"Volumetric Efficiency: {efficiency:.3f}%")
    print(f"Total bins used: {len(bins)}")
    total_unpacked = sum(remaining_counts)
    print(f"Total unpacked boxes: {total_unpacked} (IDs/types: {remaining_counts})")

if __name__=="__main__":
    main(len(sys.argv), sys.argv)
