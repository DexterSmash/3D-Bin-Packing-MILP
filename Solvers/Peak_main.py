import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from main_cplex import prepareData

@dataclass
class Box:
    id: int
    w: int
    l: int
    h: int

@dataclass
class Container:
    w: int
    l: int
    h: int

@dataclass
class Position:
    x: int
    y: int
    z: int

@dataclass
class PlacedBox:
    box: Box
    pos: Position

class PFSP3DPacker:
    def __init__(self, container: Container, boxes: List[Box], slicing_ratios: List[float]):
        self.container = container
        self.original_boxes = boxes.copy()
        self.boxes = sorted(boxes, key=lambda b: (-b.h, -b.w, -b.l))
        self.placed_boxes: List[PlacedBox] = []
        self.slicing_ratios = slicing_ratios

    def slice_container(self):
        slices = []
        for ratio in self.slicing_ratios:
            slice_height = int(self.container.h * ratio)
            if slice_height > 0:
                slices.append(slice_height)
        return slices

    def create_2D_representation(self):
        return [[0] * self.container.l for _ in range(self.container.w)]

    def can_place(self, grid, box, x, y):
        for i in range(x, x + box.w):
            for j in range(y, y + box.l):
                if i >= self.container.w or j >= self.container.l or grid[i][j] != 0:
                    return False
        return True

    def mark_occupied(self, grid, box, x, y):
        for i in range(x, x + box.w):
            for j in range(y, y + box.l):
                grid[i][j] = 1

    def place_box_in_slice(self, slice_grid, box: Box) -> Optional[Tuple[int, int]]:
        for x in range(self.container.w - box.w + 1):
            for y in range(self.container.l - box.l + 1):
                if self.can_place(slice_grid, box, x, y):
                    self.mark_occupied(slice_grid, box, x, y)
                    return x, y
        return None
    
    def peak_fill_slice(self, z_start: int, slice_height: int):
        slice_grid = self.create_2D_representation()
        temp_boxes = self.boxes.copy()
        for box in temp_boxes:
            if box.h <= slice_height:
                pos = self.place_box_in_slice(slice_grid, box)
                if pos:
                    x, y = pos
                    self.placed_boxes.append(PlacedBox(box, Position(x, y, z_start)))
                    self.boxes.remove(box)

    def pack_boxes(self):
        current_z = 0
        for slice_height in self.slice_container():
            if current_z + slice_height <= self.container.h:
                self.peak_fill_slice(current_z, slice_height)
                current_z += slice_height
        return self.placed_boxes

def main(argc, argv):
    if argc < 6:
        print("Usage: python Peak_main.py <order_id> <picklist_file.csv> <num_superSKUs> <solver> <time_limit>")
        return
    order_id = argv[1]
    picklist_file = argv[2]
    n_super_skus = int(argv[3])
    
    # Use your prepareData or equivalent CSV reader here
    n, m, Kmax, sku_df, dfc_df, qty = prepareData(picklist_file, order_id)
    
    # Create Boxes list
    boxes = []
    num_boxes = min(len(sku_df), len(qty))
    for i in range(num_boxes):
        row = sku_df.iloc[i]
        for _ in range(qty[i]):
            boxes.append(Box(
                id=i + 1,  # or unique id
                w=int(row['w']),
                l=int(row['l']),
                h=int(row['h'])
            ))
    
    # Create container from first DFC entry or your choice
    container = Container(
        w=int(dfc_df.iloc[0]['W']),
        l=int(dfc_df.iloc[0]['L']),
        h=int(dfc_df.iloc[0]['H'])
    )
    
    # Slicing ratios (e.g., split container height equally into slices)
    slicing_ratios = [0.25, 0.25, 0.25, 0.25]  # 4 equal height slices as example
    
    pfsp = PFSP3DPacker(container, boxes, slicing_ratios)
    placed_boxes = pfsp.pack_boxes()
    
    packed_volume = sum(b.box.w * b.box.l * b.box.h for b in placed_boxes)
    bin_volume = container.w * container.l * container.h
    
    volumetric_efficiency = 100 * packed_volume / bin_volume if bin_volume > 0 else 0
    
    print(f"Packed {len(placed_boxes)} boxes out of {len(boxes)}")
    print(f"Volumetric Efficiency: {volumetric_efficiency:.2f}%")
    
    # Optionally print more details or output packing report
    
    
if __name__ == "__main__":
    import sys
    main(len(sys.argv), sys.argv)
