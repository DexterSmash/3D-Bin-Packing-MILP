import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple, Optional
import random


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


#Example to check implementation
def main():
    bin=Bin(length=680, width=445, height=445)

    boxes= [
        Box(id=1, length=27, width=27, height=16),
        Box(id=2, length=34, width=18, height=13),
        Box(id=3, length=20, width=17, height=13),
        Box(id=4, length=28, width=23, height=20),
        Box(id=5, length=30, width=18, height=13),
    ]

    boxes.sort(key=lambda b:b.volume(), reverse=True)
    #sorts boxes in descending order of volume

    for box in boxes:
        if not try_placing_box_with_push(bin, box):
            if not inflate_replace(bin, box):
                print(f"{box.id} couldn't be placed.")

    print("Packed boxes:")
    for b in bin.boxes:
        print(f"Box {b.id} at position {b.position}, size=({b.length},{b.width},{b.height})")


if __name__ == "__main__":
    main()
        