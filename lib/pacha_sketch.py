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

from hilbert import decode
from hilbert import encode

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product
from .sketches import BaseSketch, CountMinSketch, BloomFilter
from .encoders import BAdicRange, BAdicCube, NumericRange, minimal_b_adic_cover, sort_b_adic_ranges, minimal_spatial_b_adic_cover, get_hilbert_ranges

__all__ = ["PachaSketch", "SpatialPachaSketch", "HilbertPachaSketch", "CMParameters", "BFParameters"]

class CMParameters:
    def __init__(self, width=-1, depth=-1, eps=0.0, delta=0.0):
        if width > 0 and depth > 0:
            self.width = width
            self.depth = depth
            self.eps = 0.0
            self.delta = 0.0
        elif eps >= 0 and delta >= 0:
            self.eps = eps
            self.delta = delta
            self.width = -1
            self.depth = -1
        else:
            raise ValueError("Invalid parameters for Count-Min Sketch.")

    def build_cm_sketch(self):
        return CountMinSketch(width=self.width, depth=self.depth, eps=self.eps, delta=self.delta)

class BFParameters:
    def __init__(self, size=None, hash_count=None, n_values=None, p=None):
        if size is not None and hash_count is not None:
            self.size = size
            self.hash_count = hash_count
            self.n_values = None
            self.p = None
        elif n_values is not None and p is not None:
            self.size = None
            self.hash_count = None
            self.n_values = n_values
            self.p = p
        else:
            raise ValueError("Invalid parameters for Bloom Filter.")

    def build_bf_sketch(self):
        return BloomFilter(size=self.size, hash_count=self.hash_count, n_values=self.n_values, p=self.p)

class PachaSketch:
    def update(self, element: tuple):
        pass

    def query(self, query: list[tuple]) -> int:
        pass

    def update_data_frame(self, df: pd.DataFrame):
        pass

class SpatialPachaSketch(PachaSketch):
    def __init__(self, num_dimensions: int, base: int, levels: int, cm_params: CMParameters, bf_params: BFParameters, sketch_type: list[str] = None):
        self.base = base
        self.levels = levels
        self.cm_params = cm_params
        self.bf_params = bf_params
        self.num_dimensions = num_dimensions

        self.base_sketches = []

        if sketch_type is None:
            sketch_type = ["cm"] * levels

        assert len(sketch_type) == levels and all([t in ["cm", "bf"] for t in sketch_type])
        
        for i in range(levels):
            if sketch_type[i] == "cm":
                self.base_sketches.append(cm_params.build_cm_sketch())
            elif sketch_type[i] == "bf":
                self.base_sketches.append(bf_params.build_bf_sketch())

    def update(self, element: tuple):
        # For now using B-Adic Cubes
        for l in range(self.levels):
            cube = []
            for d in range(len(element)):
                cube.append(element[d] // self.base**l)
            cube = tuple(cube)
            self.base_sketches[l].update(cube)

    def query(self, query: list[tuple]) -> int:
        # Working with B-Adic Cubes
        # For now, we are assuming that the query contains all dimensions
        bases = [self.base] * self.num_dimensions
        cubes = minimal_spatial_b_adic_cover(query, bases)
        estimate = 0
        for cube in cubes:
            # if cube.level == 0:
            #     n_level_0 += 1
            tuple_value = tuple([r.index for r in cube.b_adic_ranges])               
            cube_extimate = self.base_sketches[cube.level].query(tuple_value)
            estimate += cube_extimate
        return estimate
    
    def update_data_frame(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            row = tuple(row)
            self.update(row) 
        return self
    

class HilbertPachaSketch(PachaSketch):
    def __init__(self, num_dimensions: int, num_bits: int, base: int, levels: int, cm_params: CMParameters, bf_params: BFParameters, sketch_type: list[str] = None):
        self.base = base
        self.levels = levels
        self.cm_params = cm_params
        self.bf_params = bf_params
        self.num_dimensions = num_dimensions
        self.num_bits = num_bits

        self.base_sketches = []

        if sketch_type is None:
            sketch_type = ["cm"] * levels

        assert len(sketch_type) == levels and all([t in ["cm", "bf"] for t in sketch_type])
        
        for i in range(levels):
            if sketch_type[i] == "cm":
                self.base_sketches.append(cm_params.build_cm_sketch())
            elif sketch_type[i] == "bf":
                self.base_sketches.append(bf_params.build_bf_sketch())

    def update(self, element: tuple):
        # Using Hilbert Curves
        hilbert_value = encode(np.asarray(element), self.num_dimensions, self.num_bits)
        for l in range(self.levels):
            level_index = hilbert_value // (self.base ** l)
            self.base_sketches[l].update(level_index)

    def query(self, query: list[tuple]) -> int:
        # Using Hilbert Curves
        # For now, we are assuming that the query contains all dimensions
        ranges = get_hilbert_ranges(query, self.num_dimensions, self.num_bits)
        
        estimate = 0
        for r in ranges:
            # if r[0] == r[1]:
            #     n_level_0 += 1
            b_adic_cover = minimal_b_adic_cover(base=self.base, low=int(r[0]), high=int(r[1]))
            range_estimate = 0
            for b_adic_range in b_adic_cover:
                range_estimate += self.base_sketches[b_adic_range.level].query(b_adic_range.index)
            estimate += range_estimate
        return estimate
    
    def update_data_frame(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            row = tuple(row)
            self.update(row) 
        return self
    

class BaselineSketch(PachaSketch):
    def __init__(self, num_dimensions: int, cm_params: CMParameters = None, bf_params: BFParameters = None, sketch_type: str = None):
        self.cm_params = cm_params
        self.bf_params = bf_params
        self.num_dimensions = num_dimensions

        if sketch_type is None:
            sketch_type = "cm"
        
        if sketch_type == "cm":
            self.base_sketch = (cm_params.build_cm_sketch())
        elif sketch_type == "bf":
            self.base_sketch = (bf_params.build_bf_sketch())

    def update(self, element: tuple):
        self.base_sketch.update(element)


    def query(self, query: list[tuple]) -> int:
        ranges = [list(range(interval[0], interval[1] + 1)) for interval in query]

        estimate = 0    
        for point in product(*ranges):
            estimate += self.base_sketch.query(point)

        return estimate
    
    def update_data_frame(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            row = tuple(row)
            self.update(row) 
        return self