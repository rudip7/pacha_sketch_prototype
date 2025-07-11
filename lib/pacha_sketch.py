from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import copy
from typing import Any
from numpy.typing import NDArray

from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib
import random

from hilbert import decode
from hilbert import encode

import orjson
import gzip

from itertools import islice
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pympler import asizeof

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product
from .sketches import BaseSketch, CountMinSketch, BloomFilter
from .encoders import BAdicRange, BAdicCube, NumericRange, minimal_b_adic_cover, sort_b_adic_ranges,\
      minimal_spatial_b_adic_cover, get_hilbert_ranges, minimal_b_adic_cover_array, downgrade_b_adic_range_indices

from typing import List, Tuple, Dict, Any, Set


__all__ = ["PachaSketch", "ADTree", "NumericalBitmap", "CMParameters", "BFParameters"]


def cartesian_product(arrays):
    """
    Compute the Cartesian product of input 1D arrays.
    Returns an array of shape (len(arrays[0]) * ... * len(arrays[-1]), len(arrays)).
    """
    # arrays = [np.asarray(a) for a in arrays]
    mesh = np.meshgrid(*arrays, indexing='ij')
    return np.stack(mesh, axis=-1).reshape(-1, len(arrays))

def region_cross_product(cat_regions: np.ndarray, num_regions: np.ndarray) -> np.ndarray:
    n = cat_regions.shape[0]
    m = num_regions.shape[0]

    # Repeat cat_mappings for each num_mappings row
    cat_repeated = np.repeat(cat_regions, m, axis=0)  # shape (n*m, n_cat)
    # Tile num_mappings for each cat_mappings row
    num_tiled = np.tile(num_regions, (n, 1))          # shape (n*m, n_num + 1)

    # Concatenate along columns
    return np.concatenate([cat_repeated, num_tiled], axis=1)

def make_cube(combination):
    return BAdicCube(combination)

class ADTree:
    def __init__(self, json_dict: dict = None):
        if json_dict is None:
            self.num_dimensions = 0
            self.possible_values = []
            self.names = []
        else:
            self.num_dimensions = json_dict["num_dimensions"]
            self.possible_values = []
            for value_sets in json_dict["possible_values"]:
                self.possible_values.append(set(value_sets))
            self.names = json_dict["names"]

    def add_dimension(self, possible_values: Set[Any], name: str = None):
        if not isinstance(possible_values, set):
            raise TypeError("Possible values must be a set.")
        self.possible_values.append(possible_values)
        self.names.append(name if name is not None else f"Dimension {self.num_dimensions + 1}")
        self.num_dimensions += 1

    def get_mapping(self, element: tuple) -> np.ndarray:
        if len(element) != self.num_dimensions:
            raise ValueError("Element length does not match the number of dimensions.")
        mappings = []
        mappings.append(tuple("*" for _ in range(self.num_dimensions)))
        template = []
        for i, value in enumerate(element):
            if value not in self.possible_values[i]:
                raise ValueError(f"Value {value} at index {i} is not in the possible values.")
            template.append(value)
            mapping = template + (["*"] * (self.num_dimensions - 1 - i))
            mappings.append(tuple(mapping))
        return np.asarray(mappings)
    
    def get_level(self, mapping: tuple) -> int:
        return self.num_dimensions - mapping.index("*") if "*" in mapping else 0

    def get_relevant_nodes(self, predicates: List[Set[Any]], for_query=False) -> np.ndarray:
        if len(predicates) != self.num_dimensions:
            raise ValueError("Predicates length does not match the number of dimensions.")

        # Quick path for all wildcards
        if all(p == {"*"} for p in predicates):
            return np.asarray([tuple("*" for _ in range(self.num_dimensions))])

        # Identify last non-wildcard dimension
        last_predicate = self.num_dimensions - 1
        for predicate in reversed(predicates):
            if predicate != {"*"}:
                break
            last_predicate -= 1

        # Materialize wildcard dimensions with possible values
        for i in range(self.num_dimensions):
            if predicates[i] == {"*"}:
                if i < last_predicate:
                    predicates[i] = self.possible_values[i]
                continue
            elif not predicates[i].issubset(self.possible_values[i]):
                raise ValueError(f"Predicate {predicates[i]} at index {i} is not in the possible values.")

        # Use NumPy for cartesian product
        arrays = [np.array(list(p)) for p in predicates]
        relevant_nodes = cartesian_product(arrays)

        if not for_query:
            relevant_nodes = np.vstack([tuple("*" for _ in range(self.num_dimensions)), relevant_nodes])

        return relevant_nodes  
    
    def to_json(self) -> str:
        return {
            "num_dimensions": self.num_dimensions,
            "possible_values": [list(values) for values in self.possible_values],
            "names": self.names
        }
    
    def get_size(self, unit: str = "MB") -> int:
        n_bytes = asizeof.asizeof(self)
        if unit == "MB":
            return n_bytes / (1024 * 1024)
        elif unit == "KB":
            return n_bytes / 1024
        elif unit == "B":
            return n_bytes
        else:            
            raise ValueError("Unit must be 'MB', 'KB', or 'B'.")
    
    def save_to_file(self, file_path: str):
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "wb") as f:
                f.write(orjson.dumps(self.to_json()))
        else:
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(self.to_json()))

    def __eq__(self, other: ADTree) -> bool:
        return self.num_dimensions == other.num_dimensions and self.possible_values == other.possible_values
    
    @staticmethod
    def from_json(file_path: str) -> ADTree:
        """
        Build an AD Tree from a JSON file.
        """
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "rb") as f:
                data_bytes = f.read()
                json_dict = orjson.loads(data_bytes)
        else:
            with open(file_path, 'rb') as f:
                json_dict = orjson.loads(f.read())

        return ADTree(json_dict=json_dict)
    
class NumericalBitmap:
    def __init__(self, base: int, size_per_side: int = 10000):
        self.base = base
        self.exponent = 0
        self.bucket_size = 1
        size_per_side += size_per_side % base
        self.size_per_side = size_per_side
        self.limit = size_per_side
        self.positive_bitmap = np.zeros(size_per_side, dtype=bool)
        self.negative_bitmap = np.zeros(size_per_side, dtype=bool)
    
    def _increase_exponent(self):
        """
        Increase the exponent of the base, effectively doubling the size of the bitmap.
        """
        self.exponent += 1
        self.bucket_size *= self.base
        self.limit *= self.base
        # Group the positive_bitmap in groups of size base and OR them together
        n_groups = int(np.ceil(len(self.positive_bitmap) / self.base))
        compressed_positive_bitmap = np.zeros(n_groups, dtype=bool)
        for i in range(n_groups):
            start = i * self.base
            end = min((i + 1) * self.base, len(self.positive_bitmap))
            compressed_positive_bitmap[i] = np.any(self.positive_bitmap[start:end])
        self.positive_bitmap = np.concatenate([compressed_positive_bitmap, np.zeros(self.size_per_side - len(compressed_positive_bitmap), dtype=bool)])

        n_groups_neg = int(np.ceil(len(self.negative_bitmap) / self.base))
        compressed_negative_bitmap = np.zeros(n_groups_neg, dtype=bool)
        for i in range(n_groups_neg):
            start = i * self.base
            end = min((i + 1) * self.base, len(self.negative_bitmap))
            compressed_negative_bitmap[i] = np.any(self.negative_bitmap[start:end])
        self.negative_bitmap = np.concatenate([compressed_negative_bitmap, np.zeros(self.size_per_side - len(compressed_negative_bitmap), dtype=bool)])

    def update(self, value: int):
        if np.abs(value) >= self.limit:
            self._increase_exponent()
            self.update(value)
            return
        if value >= 0:
            idx = math.floor(value / self.bucket_size)
            self.positive_bitmap[idx] = True
        else:
            idx = -math.floor(value / self.bucket_size)-1
            self.negative_bitmap[idx] = True

    def query(self, value: int) -> bool:
        if np.abs(value) >= self.limit:
            return False
        if value >= 0:
            idx = math.floor(value / self.bucket_size)
            return self.positive_bitmap[idx]
        else:
            idx = -math.floor(value / self.bucket_size)-1
            return self.negative_bitmap[idx]
        
    def prune_b_adic_array(self, b_adic_array: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(b_adic_array), dtype=bool)

        for i, (level, index) in enumerate(b_adic_array):
            if level == self.exponent:
                if index >= 0:
                    if 0 <= index < len(self.positive_bitmap):
                        mask[i] = self.positive_bitmap[index]
                else:
                    idx = -index - 1
                    if 0 <= idx < len(self.negative_bitmap):
                        mask[i] = self.negative_bitmap[idx]

            elif level > self.exponent:
                level_diff = level - self.exponent
                scale = self.base ** level_diff
                if index >= 0:
                    start = index * scale
                    end = start + scale
                    mask[i] = np.any(self.positive_bitmap[start:end])
                else:
                    start = -index * scale - 1
                    end = start + scale
                    mask[i] = np.any(self.negative_bitmap[start:end])

            else:  # level < self.exponent
                level_diff = self.exponent - level
                scale = self.base ** level_diff
                idx = index // scale
                if index >= 0:
                    if 0 <= idx < len(self.positive_bitmap):
                        mask[i] = self.positive_bitmap[idx]
                else:
                    idx = -idx - 1
                    if 0 <= idx < len(self.negative_bitmap):
                        mask[i] = self.negative_bitmap[idx]

        return b_adic_array[mask]
        
    def prune_b_adic_indices(self, level, b_adic_indices: np.ndarray) -> np.ndarray:
        positive = None
        # Check if all values in b_adic_indices are >= 0
        if np.all(b_adic_indices >= 0):
            positive = True
        elif np.all(b_adic_indices < 0):
            positive = False
        else:
            raise ValueError("B-adic indices must be all positive or all negative.")
        
        mask = np.zeros(len(b_adic_indices), dtype=bool)
        if level == self.exponent:
            if positive:
                mask = self.positive_bitmap[b_adic_indices]
            else:
                mask = self.negative_bitmap[-b_adic_indices - 1]
        elif level > self.exponent:
            # If the level of the B-adic range is greater than the exponent, we need to check if any of the buckets in the range are set.
            level_diff = level - self.exponent
            scale = self.base ** level_diff
            if positive:
                start_idx = b_adic_indices * scale
                offsets = np.arange(scale)  # shape: (scale,)
                ranges = start_idx[:, None] + offsets  # shape: (num_fields, scale)
                values = self.positive_bitmap[ranges]
                mask = np.any(values, axis=1)
            else:
                start_idx = (-b_adic_indices) * scale -1
                offsets = np.arange(scale)  # shape: (scale,)
                ranges = start_idx[:, None] + offsets  # shape: (num_fields, scale)
                values = self.negative_bitmap[ranges]
                mask = np.any(values, axis=1)
        else:
            # If the level of the B-adic range is less than the exponent, we need to check if the range is covered by the bitmap.
            level_diff = self.exponent - level
            scale = self.base ** level_diff
            if positive:
                idx = np.floor(b_adic_indices / scale).astype(int)
                mask = self.positive_bitmap[idx]
            else:
                idx = -np.floor(b_adic_indices / scale).astype(int) - 1
                mask = self.negative_bitmap[idx]

        return b_adic_indices[mask]
            
            

        
    def query_b_adic_range(self, b_adic_range: BAdicRange) -> bool:
        assert b_adic_range.base == self.base, "Base of the B-adic range must match the bitmap base."
        if b_adic_range.level == self.exponent:
            if b_adic_range.index < 0:
                idx = -b_adic_range.index - 1
                return self.negative_bitmap[idx]
            else:
                idx = b_adic_range.index
                return self.positive_bitmap[idx]
        elif b_adic_range.level > self.exponent:
            # If the level of the B-adic range is greater than the exponent, we need to check if any of the buckets in the range are set.
            level_diff = b_adic_range.level - self.exponent
            if b_adic_range.index >= 0:
                # Positive range
                start_idx = b_adic_range.index * (self.base ** level_diff)
                end_idx = start_idx + (self.base ** level_diff)
                return np.any(self.positive_bitmap[start_idx:end_idx])
            else:
                # Negative range
                start_idx = -b_adic_range.index * (self.base ** level_diff) - 1
                end_idx = start_idx + (self.base ** level_diff)
                return np.any(self.negative_bitmap[start_idx:end_idx])
        else:
            # If the level of the B-adic range is less than the exponent, we need to check if the range is covered by the bitmap.
            level_diff = self.exponent - b_adic_range.level
            if b_adic_range.index >= 0:
                idx = math.floor(b_adic_range.index / (self.base ** (level_diff)))
                return self.positive_bitmap[idx]
            else:
                idx = -math.floor(b_adic_range.index / (self.base ** (level_diff))) - 1
                return self.negative_bitmap[idx]
            
    def merge(self, other: NumericalBitmap) -> NumericalBitmap:
        assert self.base == other.base, "Base of the bitmaps must match."
        assert self.size_per_side == other.size_per_side, "Size of the bitmaps must match."
        other_copy = copy.deepcopy(other)
        self_copy = copy.deepcopy(self)
        while self_copy.exponent < other_copy.exponent:
            self_copy._increase_exponent()
        while other_copy.exponent < self_copy.exponent:
            other_copy._increase_exponent()
        assert self_copy.exponent == other_copy.exponent, "Exponents of the bitmaps must match after increasing."
        self_copy.positive_bitmap = np.logical_or(self_copy.positive_bitmap, other_copy.positive_bitmap)
        self_copy.negative_bitmap = np.logical_or(self_copy.negative_bitmap, other_copy.negative_bitmap)

        return self_copy
    
    def get_size(self, unit: str = "MB") -> int:
        n_bytes = np.ceil(self.size_per_side*2 / 8.0)
        if unit == "MB":
            return n_bytes / (1024 * 1024)
        elif unit == "KB":
            return n_bytes / 1024
        elif unit == "B":
            return n_bytes
        else:            
            raise ValueError("Unit must be 'MB', 'KB', or 'B'.")
        
    def to_json(self) -> dict:
        return {
            "base": self.base,
            "exponent": self.exponent,
            "bucket_size": self.bucket_size,
            "size_per_side": self.size_per_side,
            "limit": self.limit,
            "positive_bitmap": self.positive_bitmap.tolist(),
            "negative_bitmap": self.negative_bitmap.tolist()
        }
    
    def __eq__(self, other: NumericalBitmap) -> bool:
        return (self.base == other.base and
                self.exponent == other.exponent and
                self.bucket_size == other.bucket_size and
                self.size_per_side == other.size_per_side and
                np.array_equal(self.positive_bitmap, other.positive_bitmap) and
                np.array_equal(self.negative_bitmap, other.negative_bitmap))
    
    @staticmethod
    def from_json(json_dict: dict) -> NumericalBitmap:
        """
        Build a NumericalBitmap from a JSON dictionary.
        """
        bitmap = NumericalBitmap(base=json_dict["base"], size_per_side=json_dict["size_per_side"])
        bitmap.exponent = json_dict["exponent"]
        bitmap.bucket_size = json_dict["bucket_size"]
        bitmap.limit = json_dict["limit"]
        bitmap.positive_bitmap = np.array(json_dict["positive_bitmap"], dtype=bool)
        bitmap.negative_bitmap = np.array(json_dict["negative_bitmap"], dtype=bool)
        return bitmap
    
            
class BaseSketchParameters:
    def build_sketch(self):
        pass

    def peek_size(self):
        pass

class CMParameters(BaseSketchParameters):
    def __init__(self, width: int = None, depth: int = None, error_eps: float = None, delta: float = None, seed: int = 7, epsilon: float = None):    
        if (width is not None and depth is not None and error_eps is None and delta is None) or \
            (width is None and depth is None and error_eps is not None and delta is not None):
            self.width = width
            self.depth = depth
            self.error_eps = error_eps
            self.delta = delta
            self.seed = seed
            self.epsilon = epsilon
        else:
            raise ValueError("Invalid parameters for Count-Min Sketch.")
    
    def reduce_size(self, factor: float) -> CMParameters:
        assert factor > 1, "Factor must be greater than 1 to reduce size."
        if (self.width is not None and self.depth is not None and self.error_eps is None and self.delta is None):
            new_width = int(np.ceil(self.width / factor))
            if new_width < 1000:
                return CMParameters(width=1000, depth=self.depth, seed=self.seed, epsilon=self.epsilon) # Avoid reducing below a reasonable size
            return CMParameters(width=new_width, depth=self.depth, seed=self.seed, epsilon=self.epsilon)
        elif (self.width is None and self.depth is None and self.error_eps is not None and self.delta is not None):
            new_error_eps = self.error_eps * factor
            if new_error_eps > np.e / 1000.0:
                return CMParameters(width=1000, depth=int(np.ceil(np.log(1 / self.delta))), seed=self.seed, epsilon=self.epsilon) # Avoid reducing below a reasonable size
            return CMParameters(error_eps=new_error_eps, delta=self.delta, seed=self.seed, epsilon=self.epsilon)

    def peek_size(self):
        if self.width is not None and self.depth is not None:
            return self.width, self.depth
        elif self.error_eps is not None and self.delta is not None:
            # Calculate width and depth based on error_eps and delta
            width = int(np.ceil(np.e / self.error_eps))
            depth = int(np.ceil(np.log(1 / self.delta)))
            return width, depth

    def build_sketch(self):
        return CountMinSketch(width=self.width, depth=self.depth, error_eps=self.error_eps, delta=self.delta, seed=self.seed, epsilon=self.epsilon)

class BFParameters(BaseSketchParameters):
    def __init__(self, size: int =None, hash_count: int =None, n_values: int =None, p: float=None, seed:int = 7, epsilon: float = None):
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
        self.seed = seed
        self.epsilon = epsilon

    def peek_size(self):
        if self.size is not None and self.hash_count is not None:
            return self.size, self.hash_count
        elif self.n_values is not None and self.p is not None:
            # Calculate size and hash_count based on n_values and p
            size = BloomFilter._optimal_size(self.n_values, self.p)
            hash_count = BloomFilter._optimal_hash_count(size, self.n_values)
            return size, hash_count

    def build_sketch(self):
        return BloomFilter(size=self.size, hash_count=self.hash_count, n_values=self.n_values, p=self.p, seed=self.seed, epsilon=self.epsilon)


class PachaSketch:
    """
    A PachaSketch is a multi-dimensional sketch that efficiently answers multidimensional count queries.
    """
    levels: int
    num_dimensions: int
    cat_col_map: List[int]
    num_col_map: List[int]
    bases: np.ndarray
    base_sketches: List[BaseSketch]
    ad_tree: ADTree
    numerical_bitmaps: List[NumericalBitmap] 
    cat_index: BloomFilter
    num_index: BloomFilter
    region_index: BloomFilter
    max_values: np.ndarray
    min_values: np.ndarray
    epsilon: float
    bitmap_lock: List[threading.Lock]
    cat_lock: threading.Lock
    num_lock: threading.Lock
    region_lock: threading.Lock
    sketch_locks: List[threading.Lock]


    def __init__(self, levels: int = None, num_dimensions: int= None, cat_col_map: List[int]= None, num_col_map: List[int]= None, 
                 bases: List[int]= None, base_sketch_parameters: List[BaseSketchParameters]= None,
                 ad_tree: ADTree= None, 
                 cat_index_parameters: BFParameters= None, num_index_parameters: BFParameters= None, region_index_parameters: BFParameters= None, 
                 epsilon: float = None, numerical_bitmaps_size: int = 100000, json_dict: dict = None):
        if json_dict is None:
            assert levels is not None and num_dimensions is not None, \
                "Levels and number of dimensions must be provided."
            self.levels = levels
            self.num_dimensions = num_dimensions

            assert len(cat_col_map) + len(num_col_map) == num_dimensions, \
                "The sum of categorical and numerical columns must equal the number of dimensions."
            assert set(cat_col_map).union(set(num_col_map)) == set(range(num_dimensions)), \
                "Column maps must cover all dimensions without overlap."
            self.cat_col_map = cat_col_map
            self.num_col_map = num_col_map

            assert len(bases) == len(num_col_map), \
            "The number of bases must match the number of numerical columns." 
            self.bases = np.asarray(bases, dtype=int)

            self.numerical_bitmaps = []
            for i in range(len(bases)):
                self.numerical_bitmaps.append(NumericalBitmap(base=bases[i], size_per_side=numerical_bitmaps_size))

            assert len(base_sketch_parameters) == levels, \
                "The number of base sketch parameters must match the number of levels."
            
            self.base_sketches: List[BaseSketch] = []
            for i in range(levels):
                self.base_sketches.append(
                    base_sketch_parameters[i].build_sketch()
                )
                if epsilon is not None and base_sketch_parameters[i].epsilon is None:
                    self.base_sketches[i].add_privacy_noise(epsilon)
            self.epsilon = epsilon

            assert ad_tree is not None, "ADTree must be provided."
            self.ad_tree = ad_tree
            self.cat_index = cat_index_parameters.build_sketch()
            self.num_index = num_index_parameters.build_sketch()
            self.region_index = region_index_parameters.build_sketch() 
            # if epsilon is not None:
            #     if cat_index_parameters.epsilon is None:
            #         self.cat_index.add_privacy_noise(epsilon)
            #     if num_index_parameters.epsilon is None:
            #         self.num_index.add_privacy_noise(epsilon)
            #     if region_index_parameters.epsilon is None:
            #         self.region_index.add_privacy_noise(epsilon)

            self.max_values = np.full(len(num_col_map), -np.inf)
            self.min_values = np.full(len(num_col_map), np.inf)
        else:
            self.levels = json_dict["levels"]
            self.num_dimensions = json_dict["num_dimensions"]
            self.cat_col_map = json_dict["cat_col_map"]
            self.num_col_map = json_dict["num_col_map"]
            self.bases = np.asarray(json_dict["bases"], dtype=int)
            self.ad_tree = ADTree(json_dict=json_dict["ad_tree"])
            self.cat_index = BloomFilter(json_dict=json_dict["cat_index"])
            self.num_index = BloomFilter(json_dict=json_dict["num_index"])
            self.region_index = BloomFilter(json_dict=json_dict["region_index"])

            self.numerical_bitmaps = []
            for bm_json in json_dict["numerical_bitmaps"]:
                self.numerical_bitmaps.append(NumericalBitmap.from_json(bm_json))

            self.base_sketches = []
            for sketch_json in json_dict["base_sketches"]:
                if sketch_json["type"] == "CountMinSketch":
                    self.base_sketches.append(CountMinSketch(json_dict=sketch_json))
                elif sketch_json["type"] == "BloomFilter":
                    self.base_sketches.append(BloomFilter(json_dict=sketch_json))
                else:
                    raise ValueError(f"Unknown sketch type: {sketch_json['type']}")
            self.max_values = json_dict["max_values"]
            for i in range(len(self.max_values)):
                if self.max_values[i] == None:
                    self.max_values[i] = -math.inf
            self.max_values = np.asarray(self.max_values, dtype=int)
            self.min_values = json_dict["min_values"]
            for i in range(len(self.min_values)):
                if self.min_values[i] == None:
                    self.min_values[i] = math.inf
            self.min_values = np.asarray(self.min_values, dtype=int)
            self.epsilon = json_dict["epsilon"] 

        # self.bitmap_lock = [threading.Lock() for _ in range(len(self.numerical_bitmaps))]
        # self.cat_lock = threading.Lock()
        # self.num_lock = threading.Lock()
        # self.region_lock = threading.Lock()
        # self.sketch_locks = [threading.Lock() for _ in range(self.levels)]

    def get_numerical_mappings(self, element: np.ndarray) -> np.ndarray:
        # element = np.asarray(element)
        all_levels = np.arange(self.levels)
        matrix = self.bases[:, None] ** all_levels[None, :]
        cubes = np.floor(element[:, None] / matrix[0, :]).astype(int)
        return np.column_stack([all_levels, cubes.T])            
            

    def update(self, element: tuple):
        # assert len(element) == self.num_dimensions, \
        #     "Element must have the same number of dimensions as the sketch."
        cat_values = tuple(element[i] for i in self.cat_col_map)
        num_values = np.asarray([element[i] for i in self.num_col_map], dtype=int)
        cat_mappings = self.ad_tree.get_mapping(cat_values)

        max_level_idx = np.floor(num_values / self.bases ** self.levels).astype(int)
        max_level_min = max_level_idx * self.bases ** self.levels
        max_level_max = (max_level_idx+1) * self.bases ** self.levels - 1

        self.max_values = np.max([max_level_max, self.max_values], axis=0)
        self.min_values = np.min([max_level_min, self.min_values], axis=0)

        # for i, value in enumerate(num_values):
        #     self.max_values[i] = max(self.max_values[i], value)
        #     self.min_values[i] = min(self.min_values[i], value)
        
        for i, val in enumerate(num_values):
            if not self.numerical_bitmaps[i].query(val):
            # with self.bitmap_lock[i]:
                self.numerical_bitmaps[i].update(val)

        num_mappings = self.get_numerical_mappings(num_values)
        mapped_regions = region_cross_product(cat_mappings, num_mappings)

    # with self.cat_lock:
        self.cat_index.update_batch(cat_mappings)
    # with self.num_lock:
        self.num_index.update_batch(num_mappings)
    # with self.region_lock:
        self.region_index.update_batch(mapped_regions)

        level_col = mapped_regions[:, len(self.cat_col_map)]

        # Sort by level
        sorted_idx = np.argsort(level_col)
        sorted_regions = mapped_regions[sorted_idx]
        sorted_levels = level_col[sorted_idx]

        # Find where the value changes
        _, group_boundaries = np.unique(sorted_levels, return_index=True)

        # Split into groups
        groups = np.split(sorted_regions, group_boundaries[1:])

        unique_levels = sorted_levels[group_boundaries].astype(int)

        for level, regions in zip(unique_levels, groups):
        # with self.sketch_locks[level]:
            self.base_sketches[level].update_batch(regions)

        return self
    
    def minimal_spatial_b_adic_cover(self, num_predicates: List[Tuple[int, int]]) -> np.ndarray:
        minimal_b_adic_covers = []
        for i in range(len(num_predicates)):
            cover_ranges = minimal_b_adic_cover_array(self.bases[i], num_predicates[i][0], num_predicates[i][1])
            unpruned_ranges = self.numerical_bitmaps[i].prune_b_adic_array(cover_ranges)
            minimal_b_adic_covers.append(unpruned_ranges)
        
        if any(len(covers) == 0 for covers in minimal_b_adic_covers):
            return np.asarray([])
        
        cached_pruned_ranges = {}
        def downgrade_combination(combination: np.ndarray):
            combination = np.asarray(combination)
            min_level = combination[:, 0].min()
            min_level = min(min_level, self.levels - 1)
            downgraded = []
            for i, (level, idx) in enumerate(combination):
                key = (i, level, idx, min_level)
                if key in cached_pruned_ranges:
                    downgraded.append(cached_pruned_ranges[key])
                    continue
                b_adic_indices = downgrade_b_adic_range_indices(base=self.bases[i], level=level, idx=idx, new_level=min_level)
                unpruned_indices = self.numerical_bitmaps[i].prune_b_adic_indices(min_level, b_adic_indices)
                cached_pruned_ranges[key] = unpruned_indices                
                if len(unpruned_indices) >= 1:
                    downgraded.append(unpruned_indices)
                else:
                    return [], []
            return min_level, downgraded


        levels = []
        indices = []
        for combination in product(*minimal_b_adic_covers):
            level, downgraded = downgrade_combination(combination)
            if len(downgraded) == 0:
                continue
            else:
                combination_indices = cartesian_product(downgraded)
                levels.append(np.full(len(combination_indices), level))
                indices.append(combination_indices)
        levels = np.concatenate(levels, axis=0)
        indices = np.concatenate(indices, axis=0)
        return np.hstack([levels[:, None], indices])

    
    def get_subqueries(self, query: List[Any], detailed = False, debug=False) -> List[Tuple[Tuple, BAdicCube]]:
        assert len(query) == self.num_dimensions, \
            "Query must have the same number of dimensions as the sketch."
        cat_predicates = [query[i] for i in self.cat_col_map]
        num_predicates = [query[i] for i in self.num_col_map]
        
        for i, query_d in enumerate(cat_predicates):
            if isinstance(query_d, list):
                cat_predicates[i] = set(query_d)
            elif isinstance(query_d, set):
                continue
            elif isinstance(query_d, str) and query_d == "*":
                cat_predicates[i] = {"*"}
            else:
                idx = query.index(query_d)
                raise TypeError(f"Query predicate at index {idx} expected to be a set or '*'.")
        
        for i, query_d in enumerate(num_predicates):
            if (isinstance(query_d, tuple) or isinstance(query_d, list)) and len(query_d) == 2:
                if not isinstance(query_d[0], int) or not isinstance(query_d[1], int):
                    raise TypeError("Bounds must be integers.")
                if query_d[0] > query_d[1]:
                    raise ValueError("Lower bound cannot be greater than upper bound.")
                continue
            elif isinstance(query_d, str) and query_d == "*":
                num_predicates[i] = (self.min_values[i], self.max_values[i])
            else:
                idx = query.index(query_d)
                raise TypeError(f"Query predicate at index {idx} expected to be a tuple of (lower_bound, upper_bound) or '*'.")
  
        relevant_nodes = self.ad_tree.get_relevant_nodes(cat_predicates, for_query=True)
        b_adic_cubes = self.minimal_spatial_b_adic_cover(num_predicates)

        cat_regions = relevant_nodes[self.cat_index.query_batch(relevant_nodes)]
        num_regions = b_adic_cubes[self.num_index.query_batch(b_adic_cubes)]

        candidate_regions = region_cross_product(cat_regions, num_regions)

        query_regions = candidate_regions[self.region_index.query_batch(candidate_regions)]

        if debug or detailed:
            level_idx = len(cat_predicates)
            levels, counts = np.unique(query_regions[:, level_idx], return_counts=True)
            queries_per_level = [0] * self.levels
            for level, count in zip(levels, counts):
                queries_per_level[int(level)] = count
        
        if debug:
            print(f"Categorical regions: {len(relevant_nodes)}")
            print(f"Indexed categorical regions: {len(cat_regions)}")
            print(f"Numerical regions: {len(b_adic_cubes)}")
            print(f"Indexed numerical regions: {len(num_regions)}")
            print(f"Candidate regions: {len(candidate_regions)}")
            print(f"Query regions: {len(query_regions)}")
            for i, count in enumerate(queries_per_level):
                if count > 0:
                    print(f"Level {i} queries: {count}")
        
        if detailed:
            return query_regions, {
                "relevant_nodes": len(relevant_nodes),
                "cat_regions": len(cat_regions),
                "b_adic_cubes": len(b_adic_cubes),
                "num_regions": len(num_regions),
                "candidate_regions": len(candidate_regions),
                "query_regions": len(query_regions),
                "queries_per_level": queries_per_level
            }
        return query_regions, None

    def query(self, query: List[Any], detailed = False, debug=False) -> int:
        query_regions, details = self.get_subqueries(query, detailed=detailed, debug=debug)

        level_col = query_regions[:, len(self.cat_col_map)]

        # Sort by level
        sorted_idx = np.argsort(level_col)
        sorted_queries = query_regions[sorted_idx]
        sorted_levels = level_col[sorted_idx]

        # Find where the value changes
        _, group_boundaries = np.unique(sorted_levels, return_index=True)

        # Split into groups
        groups = np.split(sorted_queries, group_boundaries[1:])

        unique_levels = sorted_levels[group_boundaries].astype(int)

        estimate = 0
        for level, queries in zip(unique_levels, groups):
            estimate += np.sum(self.base_sketches[level].query_batch(queries))
        
        if debug:
            print(f"Estimate: {estimate}")

        if detailed:
            return estimate, details
        return estimate
    
    def merge(self, other: PachaSketch) -> PachaSketch:
        if not isinstance(other, PachaSketch):
            raise TypeError("Can only merge with another PachaSketch.")
        if self.levels != other.levels or self.num_dimensions != other.num_dimensions:
            raise ValueError("PachaSketches must have the same levels and number of dimensions to merge.")
        if self.cat_col_map != other.cat_col_map or self.num_col_map != other.num_col_map:
            raise ValueError("PachaSketches must have the same categorical and numerical column maps to merge.")
        if self.bases != other.bases:
            raise ValueError("PachaSketches must have the same bases to merge.")
        if self.ad_tree != other.ad_tree:
            raise ValueError("PachaSketches must have the same ADTree to merge.")
        merged_sketch = copy.deepcopy(self)
        merged_sketch.cat_index = self.cat_index.merge(other.cat_index)
        merged_sketch.num_index = self.num_index.merge(other.num_index)
        merged_sketch.region_index = self.region_index.merge(other.region_index)
        merged_sketch.max_values = [max(self.max_values[i], other.max_values[i]) for i in range(len(self.max_values))]
        merged_sketch.min_values = [min(self.min_values[i], other.min_values[i]) for i in range(len(self.min_values))]
        
        for i in range(len(self.numerical_bitmaps)):
            merged_sketch.numerical_bitmaps[i] = self.numerical_bitmaps[i].merge(other.numerical_bitmaps[i])

        for i in range(self.levels):
            merged_sketch.base_sketches[i] = self.base_sketches[i].merge(other.base_sketches[i])
        
        if merged_sketch.epsilon is not None and other.epsilon is not None:
            merged_sketch.epsilon += other.epsilon
        elif other.epsilon is not None:
            merged_sketch.epsilon = other.epsilon
        return merged_sketch
    
    def update_data_frame(self, df: pd.DataFrame, workers=8):
        # with ThreadPoolExecutor(max_workers=workers) as executor:
        #     futures = [
        #         executor.submit(self.update, tuple(row))
        #         for _, row in df.iterrows()
        #     ]
        #     for _ in tqdm(as_completed(futures), total=len(futures), desc="Updating"):
        #         pass

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Updating"):
            self.update(row)
        return self
    
    def to_json(self) -> dict:   
        return {
            "levels": self.levels,
            "num_dimensions": self.num_dimensions,
            "cat_col_map": self.cat_col_map,
            "num_col_map": self.num_col_map,
            "bases": self.bases.tolist(),
            "ad_tree": self.ad_tree.to_json(),
            "numerical_bitmaps": [bitmap.to_json() for bitmap in self.numerical_bitmaps],
            "cat_index": self.cat_index.to_json(),
            "num_index": self.num_index.to_json(),
            "region_index": self.region_index.to_json(),
            "base_sketches": [sketch.to_json() for sketch in self.base_sketches],
            "max_values": self.max_values,
            "min_values": self.min_values,
            "epsilon": self.epsilon
        }
    
    def save_to_file(self, file_path: str):
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "wb") as f:
                f.write(orjson.dumps(self.to_json()))
        else:
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(self.to_json()))

    def get_size(self, unit: str = "MB", consider_ad_tree: bool = False) -> int:
        size = self.cat_index.get_size(unit) + self.num_index.get_size(unit) + self.region_index.get_size(unit)
        for sketch in self.base_sketches:
            size += sketch.get_size(unit)

        for bitmap in self.numerical_bitmaps:
            size += bitmap.get_size(unit)

        if consider_ad_tree:
            size += self.ad_tree.get_size(unit)
        
        return size

    
    def __eq__(self, other: object) -> bool:
        # if not isinstance(other, PachaSketch):
        #     return False
        return (
            self.levels == other.levels and
            self.num_dimensions == other.num_dimensions and
            self.cat_col_map == other.cat_col_map and
            self.num_col_map == other.num_col_map and
            self.bases == other.bases and
            self.ad_tree == other.ad_tree and
            self.cat_index == other.cat_index and
            self.num_index == other.num_index and
            self.region_index == other.region_index and
            self.base_sketches == other.base_sketches and
            self.numerical_bitmaps == other.numerical_bitmaps and
            self.max_values == other.max_values and
            self.min_values == other.min_values
        )
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # del state['cat_lock']
        # del state['num_lock']
        # del state['sketch_locks']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.cat_lock = threading.Lock()
        # self.num_lock = threading.Lock()
        # self.sketch_locks = [threading.Lock() for _ in range(self.levels)]

    @staticmethod
    def build_with_uniform_size(
        levels: int, num_dimensions: int, cat_col_map: List[int], num_col_map: List[int], 
        bases: List[int], ad_tree: ADTree, cm_params: CMParameters, 
        cat_index_parameters: BFParameters, num_index_parameters: BFParameters,
        region_index_parameters: BFParameters,
        bf_params: BFParameters = None, n_sparse_levels: int = 0, epsilon: float = None) -> PachaSketch:
        """
        Build a PachaSketch with uniform size for base sketches.
        """

        if n_sparse_levels > levels:
            raise ValueError("Number of sparse levels cannot exceed total levels.")
        if n_sparse_levels > 0 and bf_params is None:
            raise ValueError("Bloom Filter parameters must be provided for sparse levels.")
        
        base_sketch_parameters = [bf_params] * n_sparse_levels + [cm_params] * (levels-n_sparse_levels) 
            
        return PachaSketch(
            levels=levels,
            num_dimensions=num_dimensions,
            cat_col_map=cat_col_map,
            num_col_map=num_col_map,
            bases=bases,
            base_sketch_parameters=base_sketch_parameters,
            ad_tree=ad_tree,
            cat_index_parameters=cat_index_parameters,
            num_index_parameters=num_index_parameters,
            region_index_parameters=region_index_parameters,
            epsilon=epsilon
        )
    
    @staticmethod
    def build_with_uniform_size_ldp(
        levels: int, num_dimensions: int, cat_col_map: List[int], num_col_map: List[int], 
        bases: List[int], ad_tree: ADTree, cm_params: CMParameters, 
        cat_index_parameters: BFParameters, num_index_parameters: BFParameters,
        region_index_parameters: BFParameters, epsilon: float, n_silos: int = 1,
        bf_params: BFParameters = None, n_sparse_levels: int = 0) -> PachaSketch:
        """
        Build a PachaSketch with uniform size for base sketches and add n_silos times the DP Noise to the base sketches.
        """
        p_sketch = PachaSketch.build_with_uniform_size(
            levels=levels,
            num_dimensions=num_dimensions,
            cat_col_map=cat_col_map,
            num_col_map=num_col_map,
            bases=bases,
            ad_tree=ad_tree,
            cm_params=cm_params,
            cat_index_parameters=cat_index_parameters,
            num_index_parameters=num_index_parameters,
            region_index_parameters=region_index_parameters,
            bf_params=bf_params,
            n_sparse_levels=n_sparse_levels
        )

        assert epsilon > 0, "Epsilon must be greater than 0."
        assert n_silos > 0, "Number of silos must be greater than 0."

        for i in range(p_sketch.levels):
            p_sketch.base_sketches[i].add_privacy_noise_ldp(epsilon=epsilon, n_silos=n_silos)

        p_sketch.epsilon = epsilon
        
        return p_sketch
    
    @staticmethod
    def build_with_decreasing_size(
        levels: int, num_dimensions: int, cat_col_map: List[int], num_col_map: List[int], 
        bases: List[int], ad_tree: ADTree, cm_params: CMParameters, 
        cat_index_parameters: BFParameters, num_index_parameters: BFParameters,
        region_index_parameters: BFParameters, beta: float = 0.5, epsilon: float = None) -> PachaSketch:
        """
        Build a PachaSketch with uniform size for base sketches.
        beta [0.0, 1.0] is a hyperparameter that controls the size reduction factor for each level.
        """
        base_sketch_parameters = []
        base_sketch_parameters.append(cm_params)

        bases_array = np.array(bases)
        for i in range(1,levels):
            factor = ((bases_array ** i).prod())**beta
            level_sketch_parameters = cm_params.reduce_size(factor)
            base_sketch_parameters.append(level_sketch_parameters)
                    
        return PachaSketch(
            levels=levels,
            num_dimensions=num_dimensions,
            cat_col_map=cat_col_map,
            num_col_map=num_col_map,
            bases=bases,
            base_sketch_parameters=base_sketch_parameters,
            ad_tree=ad_tree,
            cat_index_parameters=cat_index_parameters,
            num_index_parameters=num_index_parameters,
            region_index_parameters=region_index_parameters
        )
    
    @staticmethod
    def from_json(file_path: str) -> PachaSketch:
        """
        Build a PachaSketch from a JSON file.
        """
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "rb") as f:
                data_bytes = f.read()
                json_dict = orjson.loads(data_bytes)
        else:
            with open(file_path, 'rb') as f:
                json_dict = orjson.loads(f.read())

        return PachaSketch(json_dict=json_dict)


