# Step 0: Selecting a DFC from available DFCs using a greedy strategy

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

class BoxType:
    def __init__(self, type_id, d, l, h, weight):
        self.type_id = type_id
        self.original_dims = (d, l, h)
        self.weight = weight
        self.orientations = list(set(permutations([d, l, h])))  # 6 orientations
    @property
    def volume(self):
        d,l,h = self.original_dims
        return d*l*h

def generate_horizontal_layers(box_types, bin_D, bin_L):
    """
    For each box type i and each orientation p (1 to 6),
    compute how many boxes can be packed in one horizontal layer.

    Returns a dict of the form:
        n_ip[(i, orientation)] = number of boxes that fit in one layer
    """
    n_ip = {}  # Key: (type_id, orientation), Value: box count per layer

    for box_type in box_types:
        for orientation in box_type.orientations:
            d_ip, l_ip, h_ip = orientation

            boxes_along_depth = floor(bin_D / d_ip)
            boxes_along_length = floor(bin_L / l_ip)
            count = boxes_along_depth * boxes_along_length

            if count > 0:
                n_ip[(box_type.type_id, orientation)] = count

    return n_ip

# Example usage for step 1:

box_types = [
    BoxType(type_id=1, d=2, l=2, h=1, weight=5),
    BoxType(type_id=2, d=1, l=3, h=2, weight=3)
]

bin_D = 5  # bin depth (Z-axis)
bin_L = 5  # bin length (X-axis)

n_ip = generate_horizontal_layers(box_types, bin_D, bin_L)

for key, val in n_ip.items():
    print(f"Box Type {key[0]}, Orientation {key[1]} → Max in layer: {val}")

# Step 2: Generate layers based on the number of boxes that fit in one layer

from itertools import product
from math import floor

def generate_feasible_solutions(bin_H, box_types):
    """
    Generate all feasible combinations of layers (ignoring weight constraint).
    
    Args:
        bin_H: height of the bin (Z direction)
        box_types: list of BoxType objects (from Stage 1)
    
    Returns:
        List of feasible solutions.
        Each solution is a list of tuples: (box_type_id, orientation, num_layers)
    """
    # First: compute all (type_id, orientation) → hᵢₚ and max_layers
    orientation_data = []
    for box_type in box_types:
        for orientation in box_type.orientations:
            _, _, h_ip = orientation  # vertical dimension
            max_layers = floor(bin_H / h_ip)
            if max_layers > 0:
                orientation_data.append((box_type.type_id, orientation, h_ip, max_layers))

    feasible_solutions = []

    # Try all combinations of number of layers (rᵢₚ) for each orientation (0 to max_layers)
    ranges = [range(0, max_layers + 1) for (_, _, _, max_layers) in orientation_data]
    
    for r_combination in product(*ranges):
        total_height = sum(r * h_ip for r, (_, _, h_ip, _) in zip(r_combination, orientation_data))
        if total_height <= bin_H and total_height > 0:  # Non-zero feasible solution
            solution = []
            for r, (type_id, orientation, h_ip, _) in zip(r_combination, orientation_data):
                if r > 0:
                    solution.append((type_id, orientation, r))  # r layers of this type/orientation
            feasible_solutions.append(solution)

    return feasible_solutions

# Example usage for step 2:

# Assume you’ve already defined BoxType class and have the same box_types list from Stage 1

bin_H = 5

solutions = generate_feasible_solutions(bin_H, box_types)

for idx, sol in enumerate(solutions[:5]):  # print first 5
    print(f"Solution {idx+1}:")
    for (type_id, orientation, r) in sol:
        print(f"  - Box Type {type_id}, Orientation {orientation}, Layers: {r}")

# Step 3: Select the best solution based on volume and weight constraints

def pack_balanced_bins(t, W, n_list, feasible_solutions, weights_by_type, box_types, bin_dims):
    m = 0
    n_remaining = n_list[:]
    packed_bins = []

    def volumetric_utilization(solution):
        return sum(r for (_, _, r) in solution)

    feasible_solutions.sort(key=volumetric_utilization, reverse=True)

    while any(n > 0 for n in n_remaining):
        sorted_weights = [
            sorted(weights_by_type[i][:n_remaining[i]], reverse=True)
            for i in range(t)
        ]

        packed_this_round = False

        for solution in feasible_solutions:
            W_s = 0
            valid = True
            layer_plan = []

            for type_id, orientation, r in solution:
                if n_remaining[type_id] < r:
                    valid = False
                    break

                weights = sorted_weights[type_id]
                half = r // 2
                heaviest = weights[:half]
                lightest = weights[-half:] if half > 0 else []
                W_s += sum(heaviest) + sum(lightest)

            if valid and W_s <= W:
                # Pack boxes & track positions for visualization
                bin_boxes = []
                current_z = 0
                for type_id, orientation, r in solution:
                    box_type = box_types[type_id]
                    d, l, h = orientation
                    boxes_in_layer = floor(bin_dims[0] / d) * floor(bin_dims[1] / l)
                    placed = 0

                    for layer in range(r):
                        layer_boxes = 0
                        for i in range(floor(bin_dims[0] / d)):
                            for j in range(floor(bin_dims[1] / l)):
                                if placed >= n_remaining[type_id]:
                                    break
                                box = {
                                    "box_type": type_id,
                                    "dims": (d, l, h),
                                    "position": (i * d, j * l, current_z)
                                }
                                bin_boxes.append(box)
                                layer_boxes += 1
                                placed += 1
                            if placed >= n_remaining[type_id]:
                                break
                        current_z += h

                    n_remaining[type_id] -= placed
                    weights_by_type[type_id] = weights_by_type[type_id][placed:]

                packed_bins.append(bin_boxes)
                m += 1
                packed_this_round = True
                break

        if not packed_this_round:
            break

    return m, packed_bins
