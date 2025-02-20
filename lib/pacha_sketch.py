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
from .sketches import BaseSketch, CountMinSketch, BloomFilter
from .encoders import BAdicRange, BAdicCube, NumericRange, minimal_b_adic_cover, sort_b_adic_ranges, minimal_spatial_b_adic_cover

__all__ = ["PachaSketch"]

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
    def __init__(self, dimensions: int, base: int, levels: int, cm_params: CMParameters, bf_params: BFParameters, sketch_type: list[str] = None):
        self.base = base
        self.levels = levels
        self.cm_params = cm_params
        self.bf_params = bf_params
        self.dimensions = dimensions

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
        bases = [self.base] * self.dimensions
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