import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple, Optional
import random
from main_cplex import prepareData
import sys
from main_cplex import visualise

@dataclass
class Box:
    id: int
    width: int
    length: int
    height: int
    position: Optional[Tuple[int, int, int]] = None 

    def volume(self):
        return self.length * self.width * self.height

@dataclass
class EndPoint:
    x: int
    y: int
    z: int

@dataclass
class Bin:
    length: int
    width: int
    height: int
    boxes: List[Box] = field(default_factory=list)
    extreme_points: List[EndPoint] = field(default_factory=lambda: [EndPoint(0, 0, 0)])

    def add_box(self, box: Box, position: Tuple[int, int, int]):
        box.position = position
        self.boxes.append(box)
        self.update_extreme_points(box)

    def can_place(self, box: Box, pos: Tuple[int, int, int]) -> bool:
        x, y, z = pos
        # Checking bounds
        if (x + box.length > self.length) or (y + box.height > self.height) or (z + box.width > self.width):
            return False
        # Check if overlapping
        for b in self.boxes:
            bx, by, bz = b.position
            if (x < bx + b.length and x + box.length > bx and
                y < by + b.height and y + box.height > by and
                z < bz + b.width and z + box.width > bz):
                return False
        return True
    
    def update_extreme_points(self, box: Box):
        x, y, z = box.position
        # Remove the extreme point where box is placed
        self.extreme_points = [p for p in self.extreme_points if not (p.x == x and p.y == y and p.z == z)]
        # Add new extreme points around the box
        candidates = [
            EndPoint(x + box.length, y, z),
            EndPoint(x, y + box.height, z),
            EndPoint(x, y, z + box.width),
        ]
        # Add candidates if inside DFC bounds and not overlapping
        for p in candidates:
            if (p.x <= self.length and p.y <= self.height and p.z <= self.width and
                all(not(self.can_place(box, (p.x, p.y, p.z)) == False) for box in self.boxes)):
                if p not in self.extreme_points:
                    self.extreme_points.append(p)
        #This just updates the extreme points with new ones
    
#Part 2: Space Defragmentation PUSH logic
def compute_push_distances(bin:Bin, axis: str):
#This func calculates the max distances a box can be pushed within the Bin along the given axis without overlapping or exceeding size.
#Axis should be x, y or z
#Returns a dictionary {box_id: ,max_push_distance}

    distances={}
    for box in bin.boxes:
        max_push = None
        if axis == 'x':
            current_end = box.position[0] + box.length
            # Find the nearest obstacle ahead on X-axis
            obstacles = [b for b in bin.boxes if b.position[1] < box.position[1] + box.height and b.position[1] + b.height > box.position[1] and b.position[2] < box.position[2] + box.width and b.position[2] + b.width > box.position[2] and b.position[0] > box.position[0]]
            next_obstacle = min((b.position[0] for b in obstacles), default=bin.length)
            max_push = next_obstacle - current_end
        elif axis == 'y':
            current_end = box.position[1] + box.height
            obstacles = [b for b in bin.boxes if b.position[0] < box.position[0] + box.length and b.position[0] + b.length > box.position[0] and b.position[2] < box.position[2] + box.width and b.position[2] + b.width > box.position[2] and b.position[1] > box.position[1]]
            next_obstacle = min((b.position[1] for b in obstacles), default=bin.height)
            max_push = next_obstacle - current_end
        else:  #z-axis
            current_end = box.position[2] + box.width
            obstacles = [b for b in bin.boxes if b.position[0] < box.position[0] + box.length and b.position[0] + b.length > box.position[0] and b.position[1] < box.position[1] + box.height and b.position[1] + b.height > box.position[1] and b.position[2] > box.position[2]]
            next_obstacle = min((b.position[2] for b in obstacles), default=bin.width)
            max_push = next_obstacle - current_end
        distances[box.id] = max(0, max_push)
    return distances

def push_boxes(bin:Box, axis:str, distance_map: dict):
    #this pushes boxes along the specified axis by the distance in the distance_map.
    #use this after compute_push_distances
    for box in bin.boxes:
        dist= distance_map.get(box.id,0)
        if dist>0:
            x,y,z = box.position
            if axis =='x':
                new_pos=(x+dist,y,z)
            elif axis =='y':
                new_pos=(x,y+dist,z)
            else:
                new_pos=(x,y,z+dist)
            
            box.position=new_pos


def try_placing_box_with_push(bin: Bin, box: Box) -> bool:
    # Try each extreme point to place the box
    for p in bin.extreme_points:
        if bin.can_place(box, (p.x, p.y, p.z)):
            # Place directly if it fits
            bin.add_box(box, (p.x, p.y, p.z))
            normalize_bins(bin)
            return True
        else:
            # Try push-out strategy: compute push distances
            x_dists = compute_push_distances(bin, 'x')
            y_dists = compute_push_distances(bin, 'y')
            z_dists = compute_push_distances(bin, 'z')
            # Push boxes away along all axes
            push_boxes(bin, 'x', x_dists)
            push_boxes(bin, 'y', y_dists)
            push_boxes(bin, 'z', z_dists)
            # Check if box fits now
            if bin.can_place(box, (p.x, p.y, p.z)):
                bin.add_box(box, (p.x, p.y, p.z))
                normalize_bins(bin)
                return True
            # If not, revert
    return False

def normalize_bins(bin: Bin):
# Repeatedly push all boxes along x, then y, then z toward origin as much as possible without overlap
    changed = True
    while changed:
        changed = False
        for axis in ['x', 'y', 'z']:
            for box in bin.boxes:
                dist = 0
                x, y, z = box.position
                while True:
                    if axis == 'x' and x - 1 >= 0 and bin.can_place(box, (x - 1, y, z)):
                        x -= 1
                        dist += 1
                    elif axis == 'y' and y - 1 >= 0 and bin.can_place(box, (x, y - 1, z)):
                        y -= 1
                        dist += 1
                    elif axis == 'z' and z - 1 >= 0 and bin.can_place(box, (x, y, z - 1)):
                        z -= 1
                        dist += 1
                    else:
                        break
                if dist > 0:
                    box.position = (x, y, z)
                    changed = True


def inflate_replace(bin: Bin, box_to_place: Box) -> bool:
    for existing_box in bin.boxes:
        if existing_box.volume() < box_to_place.volume():
            # "Inflate" existing_box by pushing boxes outwards as much as possible
            x_dists = compute_push_distances(bin, 'x')
            y_dists = compute_push_distances(bin, 'y')
            z_dists = compute_push_distances(bin, 'z')
            push_boxes(bin, 'x', x_dists)
            push_boxes(bin, 'y', y_dists)
            push_boxes(bin, 'z', z_dists)
            # Check if box_to_place fits into space freed by removing existing_box
            
            bin.boxes.remove(existing_box)
            if any(bin.can_place(box_to_place, ep) for ep in bin.extreme_points):
                # Place box_to_place in first fitting extreme point
                for p in bin.extreme_points:
                    if bin.can_place(box_to_place, (p.x, p.y, p.z)):
                        bin.add_box(box_to_place, (p.x, p.y, p.z))
                        normalize_bins(bin)
                        return True
            else:
                # revert if fails
                bin.boxes.append(existing_box)
    return False

def bin_shuffling(bins: List[Bin], K=200):
#This is an improvement heuristic and takes a lot of time. Increases volume efficiency.
    while True:
        # Find bin with lowest volume utilization
        volumes = [(sum(box.volume() for box in b.boxes), b) for b in bins]
        volumes.sort(key=lambda x: x[0])
        low_bin = volumes[0][1]
        if not low_bin.boxes:
            break

        unload_boxes = low_bin.boxes[:]
        bins.remove(low_bin)
        items_to_repack = unload_boxes[:]
        
        attempts = 0
        while items_to_repack and attempts < K:
            # Shuffle bins order, flatten boxes into sequence
            random.shuffle(bins)
            sequence = []
            for b in bins:
                sequence.extend(b.boxes)
            # Insert one item from items_to_repack randomly in sequence
            item = items_to_repack.pop(0)
            pos = random.randint(0, len(sequence))
            sequence.insert(pos, item)
            # Could repack sequentially using Spatial Defrag heur instead but it might take too much time. Not using this function in the main code as of rn.
            attempts += 1
        break

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

    boxes_for_defrag = [Box(id=i+1, length=int(row['l']), width=int(row['w']), height=int(row['h'])) for i, row in sku_df.iterrows()]
    boxes_for_defrag.sort(key=lambda b: b.volume(), reverse=True)

    bins = []
    for j in range(m):
        # assuming Kmax[j] tells how many bins of this DFC type
        count = int(Kmax.iloc[j, 0])
        for _ in range(count):
            bins.append(Bin(length=int(dfc_df.iloc[j]['L']), width=int(dfc_df.iloc[j]['W']), height=int(dfc_df.iloc[j]['H'])))
    boxes = [Box(id=i+1, length=int(row['l']), width=int(row['w']), height=int(row['h'])) for i, row in sku_df.iterrows()]

    
    unpacked_boxes = []
    for box in boxes:
        placed = False
        for bin in bins:
            if try_placing_box_with_push(bin, box):
                placed = True
                break
        if not placed:
            # Optional: try inflate_replace with all bins here
            for bin in bins:
                if inflate_replace(bin, box):
                    placed = True
                    break
        if not placed:
            unpacked_boxes.append(box.id)

    total_packed_volume = sum(sum(b.volume() for b in bin_inst.boxes) for bin_inst in bins)
    total_bin_volume = sum(bin_inst.length * bin_inst.width * bin_inst.height for bin_inst in bins if bin_inst.boxes)

    efficiency = 100 * total_packed_volume / max(1, total_bin_volume)
    print(f"\nVolumetric Efficiency: {efficiency:.3f}%")
    print(f"Total boxes unpacked: {len(unpacked_boxes)} (IDs: {unpacked_boxes})")
    
    dfc_count = {}
    dfc_type_map = {}  # Map bin to its DFC type index
    
    # First, identify which DFC type each bin corresponds to
    bin_idx = 0
    for j in range(m):
        count = int(Kmax.iloc[j, 0])
        for k in range(count):
            if bin_idx < len(bins):
                dfc_type_map[bin_idx] = j
                bin_idx += 1
    
    # Create visualization for each bin that has boxes
    for bin_idx, bin_inst in enumerate(bins):
        if not bin_inst.boxes:  # Skip empty bins
            continue
        
        # Get DFC type for this bin
        dfc_idx = dfc_type_map.get(bin_idx, 0)
        dfc_name = dfc_df.iloc[dfc_idx]['Name']
        
        # Track instance count for this DFC type
        if dfc_name not in dfc_count:
            dfc_count[dfc_name] = 0
        dfc_count[dfc_name] += 1
        
        dfc_instance_name = f"{dfc_name}_{dfc_count[dfc_name]}"
        
        # Get DFC dimensions as a list [L, W, H]
        currentDFC = [
            bin_inst.length,
            bin_inst.width,
            bin_inst.height
        ]
        
        # Prepare data for visualization
        # These lists will contain ALL boxes, to be added progressively in visualise()
        currentSKU_Coordinates = []
        currentSKU_Dimensions = []
        title = []
        
        # Extract each box's position and dimensions
        for box in bin_inst.boxes:
            x, y, z = box.position
            currentSKU_Coordinates.append([x, y, z])
            currentSKU_Dimensions.append([box.length, box.width, box.height])
            
            # Get box name from original SKU dataframe (box.id is 1-indexed)
            box_name = sku_df.iloc[box.id - 1]['name']
            title.append(box_name)
        
        # Call visualise function - it will create animated GIF
        visualise(
            currentSKU_Coordinates,
            currentSKU_Dimensions,
            currentDFC,
            f"{argv[1]}_{dfc_instance_name}",
            title,
            argv
        )
    
    # Print summary of DFCs used
    print("\nSuggested DFCs with required copies:")
    for dfc_name in sorted(set(dfc_df.iloc[dfc_type_map[i]]['Name'] 
                                for i, b in enumerate(bins) if b.boxes)):
        count = sum(1 for i, b in enumerate(bins) 
                   if b.boxes and dfc_df.iloc[dfc_type_map[i]]['Name'] == dfc_name)
        print(f"{count} {'copy' if count == 1 else 'copies'} of the DFC {dfc_name}")
    
    # Create packing plan output
    print("\nPacking plan:")
    for bin_idx, bin_inst in enumerate(bins):
        if not bin_inst.boxes:
            continue
        
        dfc_idx = dfc_type_map.get(bin_idx, 0)
        dfc_name = dfc_df.iloc[dfc_idx]['Name']
        
        # Count instance number
        instance_num = sum(1 for i in range(bin_idx + 1) 
                          if bins[i].boxes and dfc_type_map.get(i, 0) == dfc_idx)
        dfc_instance_name = f"{dfc_name}_{instance_num}"
        
        # Group boxes by SKU name
        sku_counts = {}
        for box in bin_inst.boxes:
            box_name = sku_df.iloc[box.id - 1]['name']
            sku_counts[box_name] = sku_counts.get(box_name, 0) + 1
        
        # Format output
        packing_details = []
        for sku_name, count in sku_counts.items():
            qty_text = "quantity" if count == 1 else "quantities"
            packing_details.append(f"{count} {qty_text} of SKU:{sku_name} in DFC {dfc_instance_name}")
        
        print(', '.join(packing_details))
    
    return None

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
