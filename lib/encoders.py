from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import copy
from typing import Any
from numpy.typing import NDArray

from hilbert import decode
from hilbert import encode

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product

__all__ = ["BAdicRange", "BAdicCube", "NumericRange", "minimal_b_adic_cover",
            "sort_b_adic_ranges", "minimal_spatial_b_adic_cover", "get_border_coordinates", 
            "check_jump", "get_hilbert_ranges", "draw_curve", "draw_hilbert_range", "check_query_coverage"]

class BAdicRange:
        
    def __init__(self, base, level, index):
        """
        Initialize a single b-adic range [base**level * index, base**level * (index + 1)).
        :param base: The base for the b-adic range (base >= 1).
        :param level: Power of the base defining the range size.
        :param index: Integer factor for the range.
        """
        if base < 1:
            raise ValueError("Base must be greater or equal 1.")
        # if k < 0:
        #     raise ValueError("The power k must be non-negative.")
        
        self.base = base
        self.level = level
        self.index = index
        self.low = base**level * index
        self.high = base**level * (index + 1)

    def downgrade_b_adic_range(self, new_level) -> list:
        """
        Downgrade the b-adic range to a lower level.
        :param new_level: The new level to downgrade to.
        :return: A list of BAdicRange objects with the same base but at the new level.
        """
        if new_level == self.level:
            return [copy.deepcopy(self)]
        if new_level > self.level:
            raise ValueError("Cannot downgrade to a higher level.")
        
        level_diff = self.level - new_level
        scale = self.base ** level_diff
        temp_index = self.index * scale
        new_ranges = []
        for i in range(scale):
            new_ranges.append(BAdicRange(self.base, new_level, temp_index))
            temp_index += 1

        return new_ranges

    
    def __str__(self):
        return (
            f"[{self.low}, {self.high}) ; base = {self.base} ; "
            f"level = {self.level} ; index = {self.index}"
        )

    def __eq__(self, other):
        """
        Check if two b-adic ranges are equal.
        :param other: Another BAdicRange instance.
        :return: True if the ranges are equal, False otherwise.
        """
        if not isinstance(other, BAdicRange):
            return False
        return (
            self.base == other.base
            and self.index == other.index
            and self.level == other.level
        )
            
class BAdicCube:

    def __init__(self, b_adic_ranges: list, level, index) -> BAdicCube:
        """
        Initialize a b-adic cube with a list of b-adic ranges.
        :param b_adic_ranges: A numpy array of BAdicRange objects.
        :param level: The level of the b-adic cube.
        :param index: The index of the b-adic cube.
        """
        self.b_adic_ranges = b_adic_ranges
        self.level = level
        self.index = index

    def __str__(self) -> str:
        return (
            f"Level {self.level} Cube {self.index}:\n"
            + "\n".join([str(r) for r in self.b_adic_ranges])
        )
    
    def __eq__(self, other: BAdicCube) -> bool:
        """
        Check if two b-adic cubes are equal.
        :param other: Another BAdicCube instance.
        :return: True if the cubes are equal, False otherwise.
        """
        if not isinstance(other, BAdicCube):
            return False
        return (
            self.level == other.level
            and self.index == other.index
            and all([r1 == r2 for r1, r2 in zip(self.b_adic_ranges, other.b_adic_ranges)])
        )
    
class NumericRange:
    
    def __init__(self, low, high):
            self.low = low
            self.high = high
            
    def __str__(self):
        return "[ "+str(self.low)+", "+str(self.high)+" ]"


def minimal_b_adic_cover(base, low, high, lowest_level = 0):
    """
    Compute the minimal b-adic cover of the range [low, high].
    :param base: The base for the b-adic range.
    :param low: The start of the range.
    :param high: The end of the range (inclusive).
    :return: A list of BAdicRange objects covering [low, high].
    """
    if base < 1:
        raise ValueError("Base must be greater than or equal to 1.")

    if base == 1:
        return np.asarray([BAdicRange(1, i, 0) for i in range(low, high + 1)])
            
    D = []
    level = lowest_level  # Start from the smallest level (b^0)

    # TODO: Adapt to floating point numbers 
    # make sure that lowest level can capture exactly a bucket of the lowest level
    # if 
    
    # Find the b-adic intervals for the given range starting from the bounds
    while low <= high:
        # Check if the next level can cover exactly the value of the current low 
        low_level_limit = math.floor(low/base**(level+1)) * base**(level+1)
        if low_level_limit != low:
            # When not, add as many intervals of the current level as needed to reach 
            # the bound of the nect level
            low_level_limit += base**(level+1)
            while low_level_limit != low:
                if low > high:
                    break
                index = math.floor(low/base**level)
                D.append(BAdicRange(base, level, index))
                low = low + base**level

        # Check if the next level can cover exactly the value of the current high
        high_level_limit = (math.floor(high/base**(level+1))+1) * base**(level+1) -1
        if high_level_limit != high:
            # When not, add as many intervals of the current level as needed to reach 
            # the bound of the nect level
            high_level_limit -= base**(level+1)
            while high_level_limit != high:
                if low > high:
                    break
                index = math.floor(high/base**level)
                D.append(BAdicRange(base, level, index))
                high = high - base**level
        
        level += 1
    return np.asarray(sorted(D, key=lambda x: x.low))

def sort_b_adic_ranges(b_adic_ranges):
    """
    Sort an array of BAdicRange objects by their lower bound (low attribute).

    :param b_adic_ranges: A numpy array of BAdicRange objects.
    :return: A numpy array of BAdicRange objects sorted by their low attribute.
    """
    return np.array(sorted(b_adic_ranges, key=lambda x: x.low))

def minimal_spatial_b_adic_cover(bounds: list[tuple], bases: list) -> np.ndarray:
        assert len(bounds) == len(bases)

        minimal_b_adic_covers = []
        for i in range(len(bounds)):
            minimal_b_adic_covers.append(minimal_b_adic_cover(bases[i], bounds[i][0], bounds[i][1]))

        combinations = product(*minimal_b_adic_covers)

        D = []
        for combination in combinations:
            # Find the minimum level
            min_level = combination[0].level
            for i in range(len(combination)):
                if combination[i].level < min_level:
                    min_level = combination[i].level

            # Downgrade all the ranges to the minimum level in the combination
            new_b_adic_ranges = []
            for i in range(len(combination)):
                new_b_adic_ranges.append(combination[i].downgrade_b_adic_range(min_level))

            # Generate all local combinations for the new ranges and create the BAdicCubes
            local_combinations = product(*new_b_adic_ranges)
            for local_combination in local_combinations:
                D.append(BAdicCube(local_combination, min_level, 0))
        return np.asarray(D)

def get_border_coordinates(query):
    dimensions = len(query)
    ranges = [list(range(interval[0], interval[1] + 1)) for interval in query]
    
    border_points = []
    
    for point in product(*ranges):
        for i in range(dimensions):
            if point[i] == query[i][0] or point[i] == query[i][1]:
                border_points.append(list(point))
                break
            
    return np.asarray(border_points)

def check_jump(hilbert_1, hilbert_2):
    return hilbert_1 != hilbert_2 - 1

def get_hilbert_ranges_OLD(query, num_dims, num_bits):
    query = np.asarray(query)
    border_coords = get_border_coordinates(query)
    
    if len(border_coords) == 1:
        hilbert_index = encode(border_coords, num_dims, num_bits)
        return [(hilbert_index, hilbert_index)]
        
    hilbert_indices =  np.asarray(sorted(encode(border_coords, num_dims, num_bits)))

    ranges = []
    i = 0
    start = hilbert_indices[i]
    
    while i < len(hilbert_indices):
        # Case 1: Check if in the current position we have a point range
        # Special case where the point range is at the beginning
        if start == hilbert_indices[i]:
            if i == 0 and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
                end = hilbert_indices[i]
                ranges.append((start, end))
                if i + 1 < len(hilbert_indices):
                    start = hilbert_indices[i+1]
            # Special case where the point range is at the end
            elif i == len(hilbert_indices) - 1 and check_jump(hilbert_indices[i-1], hilbert_indices[i]):
                end = hilbert_indices[i]
                ranges.append((start, end))
            # General case
            elif i > 0 and i < len(hilbert_indices) - 1 and check_jump(hilbert_indices[i-1], hilbert_indices[i]) and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
                # Confirm that it is a point range
                next_coords = decode(hilbert_indices[i]+1, num_dims, num_bits)
                if not np.all((next_coords >= query[:, 0]) & (next_coords <= query[:, 1]), axis=1)[0]:
                    end = hilbert_indices[i]
                    ranges.append((start, end))
                    start = hilbert_indices[i+1]
                

        # Case 2: Build a normal range
        # Next boder point is not contiguous in the Hilbert curve
        elif i < len(hilbert_indices) - 1 and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
            # Check if the next Hilbert index is outside the query's bounds
            next_coords = decode(hilbert_indices[i]+1, num_dims, num_bits)
            if not np.all((next_coords >= query[:, 0]) & (next_coords <= query[:, 1]), axis=1)[0]:
                end = hilbert_indices[i]
                ranges.append((start, end))
                start = hilbert_indices[i+1]
        elif i == len(hilbert_indices) - 1:
            end = hilbert_indices[i]
            ranges.append((start, end))
                
        i += 1

    return ranges


def get_hilbert_ranges(query, num_dims, num_bits):
    query = np.asarray(query)
    border_coords = get_border_coordinates(query)
    
    if len(border_coords) == 1:
        hilbert_index = encode(border_coords, num_dims, num_bits)
        return [(hilbert_index, hilbert_index)]
        
    hilbert_indices =  np.asarray(sorted(encode(border_coords, num_dims, num_bits)))
    next_points = decode(hilbert_indices+1, num_dims, num_bits)

    ranges = []
    i = 0
    start = hilbert_indices[i]
    
    while i < len(hilbert_indices):
        # Case 1: Check if in the current position we have a point range
        # Special case where the point range is at the beginning
        if start == hilbert_indices[i]:
            if i == 0 and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
                end = hilbert_indices[i]
                ranges.append((start, end))
                if i + 1 < len(hilbert_indices):
                    start = hilbert_indices[i+1]
            # Special case where the point range is at the end
            elif i == len(hilbert_indices) - 1 and check_jump(hilbert_indices[i-1], hilbert_indices[i]):
                end = hilbert_indices[i]
                ranges.append((start, end))
            # General case
            elif i > 0 and i < len(hilbert_indices) - 1 and check_jump(hilbert_indices[i-1], hilbert_indices[i]) and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
                # Confirm that it is a point range
                next_coords = next_points[i]
                if not np.all((next_coords >= query[:, 0]) & (next_coords <= query[:, 1]), axis=1)[0]:
                    end = hilbert_indices[i]
                    ranges.append((start, end))
                    start = hilbert_indices[i+1]
                

        # Case 2: Build a normal range
        # Next boder point is not contiguous in the Hilbert curve
        elif i < len(hilbert_indices) - 1 and check_jump(hilbert_indices[i], hilbert_indices[i+1]):
            # Check if the next Hilbert index is outside the query's bounds
            next_coords = np.asarray([next_points[i]])
            if not np.all((next_coords >= query[:, 0]) & (next_coords <= query[:, 1]), axis=1)[0]:
                end = hilbert_indices[i]
                ranges.append((start, end))
                start = hilbert_indices[i+1]
        elif i == len(hilbert_indices) - 1:
            end = hilbert_indices[i]
            ranges.append((start, end))
                
        i += 1

    return ranges

def draw_curve(ax, num_dims, num_bits):

    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)

    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)

    # Compute the 2-dimensional locations.
    locs = decode(hilberts, num_dims, num_bits)

    # Draw
    ax.plot(locs[:,0], locs[:,1], '.-')
    ax.set_aspect('equal')
    ax.set_title('%d bits per dimension' % (num_bits))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')

def draw_hilbert_range(ax, range, num_dims, num_bits):
    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(range[0], range[1]+1)

    # Compute the 2-dimensional locations.
    locs = decode(hilberts, num_dims, num_bits)

    # Draw
    ax.plot(locs[:,0], locs[:,1], 'r.-', )


def check_query_coverage(query, ranges, num_dims, num_bits, deep_check=False):
    query_size = np.prod([q[1] - q[0] + 1 for q in query])
    range_cover = np.sum([r[1] - r[0] +1 for r in ranges])
    if query_size != range_cover:
        print(f"Size mismatch: query size = {query_size}, range cover = {range_cover}")
        return False
    if deep_check:
        hilbert_points = []
        for r in ranges:
            hilbert_points.extend([p for p in range(r[0], r[1]+1)])
        if len(hilbert_points) != len(set(hilbert_points)):
            print("Duplicate points in the ranges")
            return False
        decoded_points = decode(hilbert_points, num_dims, num_bits)
        for p in decoded_points:
            if not all([q[0] <= p[i] <= q[1] for i, q in enumerate(query)]):
                print(f"Point {p} is out of the query range")
                return False
    return True