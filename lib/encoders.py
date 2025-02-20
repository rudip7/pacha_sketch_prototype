from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import copy
from typing import Any
from numpy.typing import NDArray

import hashlib
import random

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product

__all__ = ["BAdicRange", "BAdicCube", "NumericRange", "minimal_b_adic_cover", "sort_b_adic_ranges", "minimal_spatial_b_adic_cover"]





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