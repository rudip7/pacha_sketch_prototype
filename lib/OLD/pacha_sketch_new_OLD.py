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

from pympler import asizeof

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product
from ..sketches import BaseSketch, CountMinSketch, BloomFilter
from ..encoders import BAdicRange, BAdicCube, NumericRange, minimal_b_adic_cover, sort_b_adic_ranges, minimal_spatial_b_adic_cover, get_hilbert_ranges

from typing import List, Tuple, Dict, Any, Set


__all__ = ["PachaSketch", "ADTree", "CMParameters", "BFParameters"]

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

    def get_mapping(self, element: tuple) -> list[tuple]:
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
        return mappings
    
    def get_relevant_nodes(self, predicates: List[Set[Any]], for_query=False) -> list[tuple]:
        if len(predicates) != self.num_dimensions:
            raise ValueError("Predicates length does not match the number of dimensions.")
        
        if predicates == list({"*"} for _ in range(self.num_dimensions)):
            return [tuple("*" for _ in range(self.num_dimensions))]
        
        last_predicate = self.num_dimensions - 1
        for predicate in reversed(predicates):
            if predicate != {"*"}:
                break
            last_predicate -= 1
        
        for i in range(self.num_dimensions):
            if predicates[i] == {"*"}:
                if i < last_predicate:
                    predicates[i] = self.possible_values[i]
                continue
            elif not predicates[i].issubset(self.possible_values[i]):
                raise ValueError(f"Predicate {predicates[i]} at index {i} is not in the possible values.")
        
        relevant_nodes = list(product(*predicates))
        if not for_query:
            relevant_nodes = [tuple("*" for _ in range(self.num_dimensions))] + relevant_nodes
        return relevant_nodes   
    
    def to_json(self) -> str:
        return {
            "num_dimensions": self.num_dimensions,
            "possible_values": [list(values) for values in self.possible_values],
            "names": self.names
        }
    
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
    bases: List[int]
    base_sketches: List[BaseSketch]
    ad_tree: ADTree
    cat_index: BloomFilter
    num_index: BloomFilter
    max_values: List[float]
    min_values: List[float]

    def __init__(self, levels: int = None, num_dimensions: int= None, cat_col_map: List[int]= None, num_col_map: List[int]= None, 
                 bases: List[int]= None, base_sketch_parameters: List[BaseSketchParameters]= None,
                 ad_tree: ADTree= None, 
                 cat_index_parameters: BFParameters= None, num_index_parameters: BFParameters= None, 
                 epsilon: float = None, json_dict: dict = None):
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
            self.bases = bases

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
            if epsilon is not None:
                if cat_index_parameters.epsilon is None:
                    self.cat_index.add_privacy_noise(epsilon)
                if num_index_parameters.epsilon is None:
                    self.num_index.add_privacy_noise(epsilon)

            self.max_values = [-math.inf] * len(num_col_map)
            self.min_values = [math.inf] * len(num_col_map)
        else:
            self.levels = json_dict["levels"]
            self.num_dimensions = json_dict["num_dimensions"]
            self.cat_col_map = json_dict["cat_col_map"]
            self.num_col_map = json_dict["num_col_map"]
            self.bases = json_dict["bases"]
            self.ad_tree = ADTree(json_dict=json_dict["ad_tree"])
            self.cat_index = BloomFilter(json_dict=json_dict["cat_index"])
            self.num_index = BloomFilter(json_dict=json_dict["num_index"])
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
            self.min_values = json_dict["min_values"]
            for i in range(len(self.min_values)):
                if self.min_values[i] == None:
                    self.min_values[i] = math.inf
            self.epsilon = json_dict["epsilon"] 

        self.cat_lock = threading.Lock()
        self.num_lock = threading.Lock()
        self.sketch_locks = [threading.Lock() for _ in range(self.levels)]

    def get_numerical_mappings(self, element: tuple) -> list[BAdicCube]:
        if len(element) != len(self.num_col_map):
            raise ValueError("Element length does not match the number of numerical dimensions.")
        for i, value in enumerate(element):
            self.max_values[i] = max(self.max_values[i], value)
            self.min_values[i] = min(self.min_values[i], value)
        
        num_mappings = []
        for level in range(self.levels):
            b_adic_ranges = []
            for i, base in enumerate(self.bases):
                b_adic_ranges.append(BAdicRange(base, level, element[i] // base**level))
            num_mappings.append(BAdicCube(b_adic_ranges))
        return num_mappings            
            

    def update(self, element: tuple):
        assert len(element) == self.num_dimensions, \
            "Element must have the same number of dimensions as the sketch."
        cat_values = tuple(element[i] for i in self.cat_col_map)
        num_values = tuple(element[i] for i in self.num_col_map)
        cat_mappings = self.ad_tree.get_mapping(cat_values)
        num_mappings = self.get_numerical_mappings(num_values)

        mapped_regions = list(product(*[cat_mappings, num_mappings]))
        
        with self.cat_lock:
            for region in mapped_regions:
                self.cat_index.update(region[0])
        with self.num_lock:
            for region in mapped_regions:
                self.num_index.update(region[1])

        for region in mapped_regions:
            with self.sketch_locks[region[1].level]:
                self.base_sketches[region[1].level].update(region)

        return self

    def query(self, query: List[Any], detailed = False, debug=False) -> int:
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
        b_adic_cubes = minimal_spatial_b_adic_cover(num_predicates, self.bases)

        cat_regions = []
        for node in relevant_nodes:
            if self.cat_index.query(node):
                cat_regions.append(node)
        
        num_regions = []
        for cube in b_adic_cubes:
            if self.num_index.query(cube):
                num_regions.append(cube)

        query_regions = list(product(cat_regions, num_regions))

        if debug:
            print(f"Categorical regions: {len(relevant_nodes)}")
            print(f"Indexed categorical regions: {len(cat_regions)}")
            print(f"Numerical regions: {len(b_adic_cubes)}")
            print(f"Indexed numerical regions: {len(num_regions)}")
            print(f"Query regions: {len(query_regions)}")

            queries_per_level = [0] * self.levels
            for region in query_regions:
                queries_per_level[region[1].level] += 1
            for i, count in enumerate(queries_per_level):
                if count > 0:
                    print(f"Level {i} queries: {count}")

        estimate = 0
        for region in query_regions:
            num_region = region[1]
            estimate += self.base_sketches[num_region.level].query(region)
        
        if detailed:
            return estimate, {
                "cat_regions": len(relevant_nodes),
                "num_regions": len(num_regions),
                "query_regions": len(query_regions),
                "queries_per_level": queries_per_level
            }
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
        merged_sketch.max_values = [max(self.max_values[i], other.max_values[i]) for i in range(len(self.max_values))]
        merged_sketch.min_values = [min(self.min_values[i], other.min_values[i]) for i in range(len(self.min_values))]
        
        for i in range(self.levels):
            merged_sketch.base_sketches[i] = self.base_sketches[i].merge(other.base_sketches[i])
        
        if merged_sketch.epsilon is not None and other.epsilon is not None:
            merged_sketch.epsilon += other.epsilon
        elif other.epsilon is not None:
            merged_sketch.epsilon = other.epsilon
        return merged_sketch
    
    def update_data_frame(self, df: pd.DataFrame, workers=8):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.update, tuple(row))
                for _, row in df.iterrows()
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Updating"):
                pass
        return self
    
    def to_json(self) -> dict:   
        return {
            "levels": self.levels,
            "num_dimensions": self.num_dimensions,
            "cat_col_map": self.cat_col_map,
            "num_col_map": self.num_col_map,
            "bases": self.bases,
            "ad_tree": self.ad_tree.to_json(),
            "cat_index": self.cat_index.to_json(),
            "num_index": self.num_index.to_json(),
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

    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PachaSketch):
            return False
        return (
            self.levels == other.levels and
            self.num_dimensions == other.num_dimensions and
            self.cat_col_map == other.cat_col_map and
            self.num_col_map == other.num_col_map and
            self.bases == other.bases and
            self.ad_tree == other.ad_tree and
            self.cat_index == other.cat_index and
            self.num_index == other.num_index and
            self.base_sketches == other.base_sketches and
            self.max_values == other.max_values and
            self.min_values == other.min_values
        )
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cat_lock']
        del state['num_lock']
        del state['sketch_locks']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cat_lock = threading.Lock()
        self.num_lock = threading.Lock()
        self.sketch_locks = [threading.Lock() for _ in range(self.levels)]

    @staticmethod
    def build_with_uniform_size(
        levels: int, num_dimensions: int, cat_col_map: List[int], num_col_map: List[int], 
        bases: List[int], ad_tree: ADTree, cm_params: CMParameters, 
        cat_index_parameters: BFParameters, num_index_parameters: BFParameters,
        bf_params: BFParameters = None, n_sparse_levels: int = 0) -> PachaSketch:
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
            num_index_parameters=num_index_parameters
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


class PachaSketchMatrix:
    """
    A PachaSketch is a multi-dimensional sketch that efficiently answers multidimensional count queries.
    """
    levels: int
    num_dimensions: int
    cat_col_map: List[int]
    num_col_map: List[int]
    bases: List[int]
    base_sketches: List[List[BaseSketch]]
    ad_tree: ADTree
    cat_index: BloomFilter
    num_index: BloomFilter
    max_values: List[float]
    min_values: List[float]
    epsilon: float
    cat_lock: threading.Lock
    num_lock: threading.Lock
    sketch_locks: List[List[threading.Lock]]

    def __init__(self, levels: int = None, num_dimensions: int= None, cat_col_map: List[int]= None, num_col_map: List[int]= None, 
                 bases: List[int]= None, base_sketch_parameters: List[BaseSketchParameters]= None,
                 ad_tree: ADTree= None, 
                 cat_index_parameters: BFParameters= None, num_index_parameters: BFParameters= None, 
                 epsilon: float = None, json_dict: dict = None):
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
            self.bases = bases

            assert len(base_sketch_parameters) == levels, \
                "The number of base sketch parameters must match the number of levels."
            
            self.base_sketches: List[List[BaseSketch]] = []
            for i in range(levels):
                level_sketches = []
                for j in range(len(cat_col_map)+1):
                    level_sketches.append(
                        base_sketch_parameters[i].build_sketch()
                    )
                    # if epsilon is not None and base_sketch_parameters[i].epsilon is None:
                    #     self.base_sketches[i][j].add_privacy_noise(epsilon)
                self.base_sketches.append(level_sketches)
            self.epsilon = epsilon

            assert ad_tree is not None, "ADTree must be provided."
            self.ad_tree = ad_tree
            self.cat_index = cat_index_parameters.build_sketch()
            self.num_index = num_index_parameters.build_sketch()
            if epsilon is not None:
                if cat_index_parameters.epsilon is None:
                    self.cat_index.add_privacy_noise(epsilon)
                if num_index_parameters.epsilon is None:
                    self.num_index.add_privacy_noise(epsilon)

            self.max_values = [-math.inf] * len(num_col_map)
            self.min_values = [math.inf] * len(num_col_map)
        else:
            self.levels = json_dict["levels"]
            self.num_dimensions = json_dict["num_dimensions"]
            self.cat_col_map = json_dict["cat_col_map"]
            self.num_col_map = json_dict["num_col_map"]
            self.bases = json_dict["bases"]
            self.ad_tree = ADTree(json_dict=json_dict["ad_tree"])
            self.cat_index = BloomFilter(json_dict=json_dict["cat_index"])
            self.num_index = BloomFilter(json_dict=json_dict["num_index"])
            self.base_sketches = []
            for sketch_level in json_dict["base_sketches"]:
                level_sketches = []
                for sketch_json in sketch_level:
                    if sketch_json["type"] == "CountMinSketch":
                        level_sketches.append(CountMinSketch(json_dict=sketch_json))
                    elif sketch_json["type"] == "BloomFilter":
                        level_sketches.append(BloomFilter(json_dict=sketch_json))
                    else:
                        raise ValueError(f"Unknown sketch type: {sketch_json['type']}")
                self.base_sketches.append(level_sketches)
            self.max_values = json_dict["max_values"]
            for i in range(len(self.max_values)):
                if self.max_values[i] == None:
                    self.max_values[i] = -math.inf
            self.min_values = json_dict["min_values"]
            for i in range(len(self.min_values)):
                if self.min_values[i] == None:
                    self.min_values[i] = math.inf
            self.epsilon = json_dict["epsilon"] 

        self.cat_lock = threading.Lock()
        self.num_lock = threading.Lock()
        self.sketch_locks = []
        for _ in range(self.levels):
            level_locks = []
            for _ in range(len(cat_col_map) + 1):
                level_locks.append(threading.Lock())
            self.sketch_locks.append(level_locks)

    def get_numerical_mappings(self, element: tuple) -> list[BAdicCube]:
        if len(element) != len(self.num_col_map):
            raise ValueError("Element length does not match the number of numerical dimensions.")
        for i, value in enumerate(element):
            self.max_values[i] = max(self.max_values[i], value)
            self.min_values[i] = min(self.min_values[i], value)
        
        num_mappings = []
        for level in range(self.levels):
            b_adic_ranges = []
            for i, base in enumerate(self.bases):
                b_adic_ranges.append(BAdicRange(base, level, element[i] // base**level))
            num_mappings.append(BAdicCube(b_adic_ranges))
        return num_mappings            
            

    def update(self, element: tuple):
        assert len(element) == self.num_dimensions, \
            "Element must have the same number of dimensions as the sketch."
        cat_values = tuple(element[i] for i in self.cat_col_map)
        num_values = tuple(element[i] for i in self.num_col_map)
        cat_mappings = self.ad_tree.get_mapping(cat_values)
        num_mappings = self.get_numerical_mappings(num_values)

        mapped_regions = list(product(*[cat_mappings, num_mappings]))
        
        with self.cat_lock:
            for region in mapped_regions:
                self.cat_index.update(region[0])
        with self.num_lock:
            for region in mapped_regions:
                self.num_index.update(region[1])

        for region in mapped_regions:
            cat_level = self.ad_tree.get_level(region[0])  
            with self.sketch_locks[region[1].level][cat_level]:
                self.base_sketches[region[1].level][cat_level].update(region)

        return self

    def query(self, query: List[Any], detailed = False, debug=False) -> int:
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
        b_adic_cubes = minimal_spatial_b_adic_cover(num_predicates, self.bases)

        cat_regions:List[Tuple] = []
        for node in relevant_nodes:
            if self.cat_index.query(node):
                cat_regions.append(node)
        
        num_regions:List[BAdicCube] = []
        for cube in b_adic_cubes:
            if self.num_index.query(cube):
                num_regions.append(cube)

        query_regions = list(product(cat_regions, num_regions))

        if debug:
            print(f"Categorical regions: {len(relevant_nodes)}")
            print(f"Indexed categorical regions: {len(cat_regions)}")
            print(f"Numerical regions: {len(b_adic_cubes)}")
            print(f"Indexed numerical regions: {len(num_regions)}")
            print(f"Query regions: {len(query_regions)}")

            numerical_per_level = [0] * self.levels
            for region in num_regions:
                numerical_per_level[region.level] += 1
            for i, count in enumerate(numerical_per_level):
                if count > 0:
                    print(f"Level {i} queries: {count}")

            categorical_per_level = [0] * (self.ad_tree.num_dimensions + 1)
            for region in cat_regions:
                categorical_per_level[self.ad_tree.get_level(region)] += 1
            for i, count in enumerate(categorical_per_level):
                if count > 0:
                    print(f"Level {i} queries: {count}")

        estimate = 0
        for region in query_regions:
            cat_region = region[0]
            num_region = region[1]
            estimate += self.base_sketches[num_region.level][self.ad_tree.get_level(cat_region)].query(region)
        
        if detailed:
            return estimate, {
                "cat_regions": len(relevant_nodes),
                "num_regions": len(num_regions),
                "query_regions": len(query_regions),
                "numerical_per_level": numerical_per_level,
                "categorical_per_level": categorical_per_level
            }
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
        merged_sketch.max_values = [max(self.max_values[i], other.max_values[i]) for i in range(len(self.max_values))]
        merged_sketch.min_values = [min(self.min_values[i], other.min_values[i]) for i in range(len(self.min_values))]
        
        for i in range(self.levels):
            for j in range(len(self.cat_col_map) + 1):
                merged_sketch.base_sketches[i][j] = self.base_sketches[i][j].merge(other.base_sketches[i][j])
        
        if merged_sketch.epsilon is not None and other.epsilon is not None:
            merged_sketch.epsilon += other.epsilon
        elif other.epsilon is not None:
            merged_sketch.epsilon = other.epsilon
        return merged_sketch
    
    def update_data_frame(self, df: pd.DataFrame, workers=8):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.update, tuple(row))
                for _, row in df.iterrows()
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Updating"):
                pass
        return self
    
    def to_json(self) -> dict:   
        base_sketches_json = []
        for level_sketches in self.base_sketches:
            level_sketches_json = []
            for sketch in level_sketches:
                level_sketches_json.append(sketch.to_json())
            base_sketches_json.append(level_sketches_json)
        return {
            "levels": self.levels,
            "num_dimensions": self.num_dimensions,
            "cat_col_map": self.cat_col_map,
            "num_col_map": self.num_col_map,
            "bases": self.bases,
            "ad_tree": self.ad_tree.to_json(),
            "cat_index": self.cat_index.to_json(),
            "num_index": self.num_index.to_json(),
            "base_sketches": base_sketches_json,
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

    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PachaSketch):
            return False
        return (
            self.levels == other.levels and
            self.num_dimensions == other.num_dimensions and
            self.cat_col_map == other.cat_col_map and
            self.num_col_map == other.num_col_map and
            self.bases == other.bases and
            self.ad_tree == other.ad_tree and
            self.cat_index == other.cat_index and
            self.num_index == other.num_index and
            self.base_sketches == other.base_sketches and
            self.max_values == other.max_values and
            self.min_values == other.min_values
        )
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cat_lock']
        del state['num_lock']
        del state['sketch_locks']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cat_lock = threading.Lock()
        self.num_lock = threading.Lock()
        self.sketch_locks = [threading.Lock() for _ in range(self.levels)]

    @staticmethod
    def build_with_uniform_size(
        levels: int, num_dimensions: int, cat_col_map: List[int], num_col_map: List[int], 
        bases: List[int], ad_tree: ADTree, cm_params: CMParameters, 
        cat_index_parameters: BFParameters, num_index_parameters: BFParameters,
        bf_params: BFParameters = None, n_sparse_levels: int = 0) -> PachaSketch:
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
            num_index_parameters=num_index_parameters
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








