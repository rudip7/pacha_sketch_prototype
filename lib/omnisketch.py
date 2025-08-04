from __future__ import annotations

import math

import numpy as np
import xxhash
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
from ctypes import Union, c_int32
from itertools import product

from pympler import asizeof

from .sketches import HashFunctionFamily
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Set

import random
import math
import copy


class DieHash:
    def __init__(self, max_level, seed=7):
        self.m = max_level
        rng = random.Random(seed)
        self.a = rng.randint(-2**31, 2**31 - 1)
        self.b = rng.randint(-2**31, 2**31 - 1)
        # self.test_die_hash_distribution()  # Uncomment to test distribution

    def hash(self, id_: int) -> int:
        modulus = 2 ** (self.m - 1)
        x = (self.a * id_ + self.b) % modulus
        if x < 0:
            raise RuntimeError("x is negative")

        if x == 0:
            level = self.m - 1
        else:
            level = self.m - int(math.ceil(math.log(x, 2))) - 1

        if level > self.m:
            raise RuntimeError("Level too high")
        if level < 0:
            raise RuntimeError(
                f"Level negative ({level}) for id {id_}, x {x}, m {self.m}, a {self.a}, b {self.b}"
            )
        return level

    def test_die_hash_distribution(self):
        counts = [0] * self.m
        for i in range(1_000_000_000):
            level = self.hash(i)
            if level >= self.m or level < 0:
                raise RuntimeError(f"Invalid level {level} for i = {i}")
            counts[level] += 1
        for i, count in enumerate(counts):
            print(f"Level {i}: {count}")

    def clone(self):
        return copy.deepcopy(self)



import random
import heapq
from sortedcontainers import SortedSet

class Kmin:
    sketch: SortedSet 

    def __init__(self, delta=0.05, max_sample_size=10_000, b=30, repetition=0):
        self.k = max_sample_size
        self.delta = delta
        self.cur_sample_size = 0
        self.n = 0
        self.simax = float('-inf')
        self.seed = None
        self.rng = random.Random()
        self.sketch = SortedSet([])  
        self.cur_tree_root = float('inf')
        self.max_hash = min(2 ** b, 2**31 - 1)
        self.repetition = repetition

    def __copy__(self) -> Kmin:
        copy_kmin = Kmin(self.delta)
        copy_kmin.k = self.k
        copy_kmin.cur_sample_size = self.cur_sample_size
        copy_kmin.sketch = copy.deepcopy(self.sketch)
        return copy_kmin

    def hash(self, x: int) -> int:
        self.rng.seed(x + self.repetition)
        return self.rng.randint(0, self.max_hash)

    def add(self, hx: int):
        self.n += 1
        if self.cur_sample_size < self.k:
            self.sketch.add(hx)  # use negative to simulate max-heap
            self.cur_sample_size += 1
            self.cur_tree_root = self.sketch[-1]  # max element in the sample
        else:
            if hx < self.cur_tree_root:
                self.sketch.pop()
                self.sketch.add(hx)
                self.cur_tree_root = self.sketch[-1]

    def reset(self):
        self.sketch = []
        self.cur_sample_size = 0
        self.cur_tree_root = float('inf')

    # This method merges two Kmin instances without respepcting the max size limit. 
    def merge_samples(self, other: Kmin):
        if not isinstance(other, Kmin):
            raise TypeError("Can only merge with another Kmin instance.")
        copy_kmin = self.__copy__()


        copy_kmin.n += other.n
        copy_kmin.cur_sample_size += other.cur_sample_size
        copy_kmin.sketch = copy_kmin.sketch.union(other.sketch)
        copy_kmin.cur_tree_root = copy_kmin.sketch[-1] 
        
        return copy_kmin

    def get_sample(self) -> list[int]:
        return self.sketch

    def __str__(self):
        return str(self.sketch)
    
    def __repr__(self):
        if self.cur_sample_size == 0:
            return "Empty Kmin(k={self.k}, delta={self.delta})"
        return f"Kmin(k={self.k}, delta={self.delta}, cur_sample_size={self.cur_sample_size}, " \
               f"cur_tree_root={self.cur_tree_root}, n={self.n})"


class CountMin:
    cm: list[list[Kmin]]  
    def __init__(self, die_hash_functions, attr, width, depth, delta_ds=0.05, 
                 repetition=0):
        self.die_hash_functions = die_hash_functions  # list of DieHash
        self.attr = attr
        self.rng = random.Random()
        self.width = width
        self.depth = depth
        self.delta_ds = delta_ds
        self.repetition = repetition
        self.cm = [[None for _ in range(self.width)] for _ in range(self.depth)]
        self.init_sketch()

    def init_sketch(self):
        for j in range(self.depth):
            for i in range(self.width):
                self.cm[j][i] = Kmin(self.delta_ds)

    def hash(self, attr_value: int, depth: int, width: int) -> list[int]:
        self.rng.seed(attr_value + self.repetition)
        return [self.rng.randint(0, width - 1) for _ in range(depth)]

    def add(self, id_: int, attr_value: int, hx: int):
        hashes = self.hash(attr_value, self.depth, self.width)
        for j in range(self.depth):
            w = hashes[j]
            self.cm[j][w].add(hx)

    def query(self, attr_value: int) -> List[Kmin]:
        hashes = self.hash(attr_value, self.depth, self.width)
        result = [None] * self.depth
        for j in range(self.depth):
            w = hashes[j]
            result[j] = self.cm[j][w]
        return result

    def reset(self):
        for j in range(self.depth):
            for i in range(self.width):
                self.cm[j][i].reset()

from typing import List


class CountMinDyad:
    def __init__(self, attr: int, interval_size: int, width: int, depth: int, delta_ds=0.05,
                 dyadic_range_bits=33):
        self.attr = attr
        self.interval_size = interval_size
        self.width = width
        self.depth = depth
        self.delta_ds = delta_ds
        self.dyadic_range_bits = dyadic_range_bits

        self.rng = random.Random()
        self.n = 0
        self.cm: List[List[Kmin]] = [
            [None for _ in range(self.width)] for _ in range(self.depth)
        ]
        self.init_sketch()

    def init_sketch(self):
        for j in range(self.depth):
            for i in range(self.width):
                self.cm[j][i] = Kmin(self.delta_ds)

    def hash(self, attr_value: int, depth: int, width: int) -> List[int]:
        if not isinstance(attr_value, int):
            try:
                attr_value = int(attr_value)
            except (ValueError, TypeError):
                attr_value = hash(attr_value)
        self.rng.seed(attr_value)
        return [self.rng.randint(0, width - 1) for _ in range(depth)]

    def get_range_signature(self, start: int, stop: int) -> int:
        sig = start << self.dyadic_range_bits
        sig |= stop
        return sig

    def add(self, lower: int, higher: int, hx: int):
        sig = self.get_range_signature(lower, higher)
        hashes = self.hash(sig, self.depth, self.width)
        for j in range(self.depth):
            w = hashes[j]
            self.cm[j][w].add(hx)
        self.n += 1

    def range_query(self, lower: int, higher: int) -> List[Kmin]:
        result = [None] * self.depth
        sig = self.get_range_signature(lower, higher)
        hashes = self.hash(sig, self.depth, self.width)
        for j in range(self.depth):
            w = hashes[j]
            result[j] = self.cm[j][w]
            # seen_n[j] += self.cm[j][w].n
        return result

    def reset(self):
        for j in range(self.depth):
            for i in range(self.width):
                self.cm[j][i].reset()


def sorted_intersection(sets: list[SortedSet]) -> list:
    if not sets:
        return []

    # Sort sets by size to start from the smallest
    sets.sort(key=len)

    # Convert each SortedSet to an iterator and initialize current values
    iterators = [iter(s) for s in sets]
    current_values = []

    # Prime the iterators
    for it in iterators:
        try:
            current_values.append(next(it))
        except StopIteration:
            return []  # One set is empty, intersection is empty

    # result = []
    intersection_count = 0

    while True:
        # If all current_values are equal, add to result and advance all
        if all(val == current_values[0] for val in current_values):
            # result.append(current_values[0])
            intersection_count += 1
            try:
                for i in range(len(iterators)):
                    current_values[i] = next(iterators[i])
            except StopIteration:
                break  # One iterator exhausted — intersection complete
        else:
            # Find max among current values
            max_val = max(current_values)
            for i in range(len(iterators)):
                while current_values[i] < max_val:
                    try:
                        current_values[i] = next(iterators[i])
                    except StopIteration:
                        return intersection_count  # One iterator exhausted

    return intersection_count











class Sample:
    def add(self, id_or_hx):
        pass

    def reset(self):
        pass

class DistinctSample(Sample):
    def __init__(self, die_hash_or_other, max_sample_size=10_000, max_level=25):
        self.sample_level = 0
        self.cur_sample_size = 0
        self.intersect_size = 0
        self.t = 1
        self.max_sample_size = max_sample_size
        self.max_level = max_level
        self.sample = dict()  # {level: set of ids}

        if isinstance(die_hash_or_other, DistinctSample):
            other = die_hash_or_other
            self.sample_level = other.sample_level
            self.cur_sample_size = other.cur_sample_size
            self.t = other.t
            self.die_hash = other.die_hash.clone()
            self.sample = {level: set(ids) for level, ids in other.sample.items()}
        else:
            self.die_hash = die_hash_or_other

    def add(self, id_, hx=None):
        level = self.die_hash.hash(id_)
        if level >= self.sample_level:
            if level not in self.sample:
                self.sample[level] = set()
            self.sample[level].add(id_)
            self.cur_sample_size += 1

        if self.cur_sample_size > self.max_sample_size:
            self.remove()

        assert self.cur_sample_size <= self.max_sample_size

    def remove(self):
        if self.sample_level in self.sample:
            level_size = len(self.sample[self.sample_level])
            self.cur_sample_size -= level_size
            self.sample[self.sample_level].clear()
            del self.sample[self.sample_level]
        else:
            raise RuntimeError(f"Sample does not contain level {self.sample_level}")

        self.sample_level += 1
        while self.sample_level <= self.max_level and self.sample_level not in self.sample:
            self.sample_level += 1

    def intersect(self, other: DistinctSample) -> DistinctSample:
        self.sample_level = max(self.sample_level, other.sample_level)
        new_size = 0
        for level in sorted(self.sample.keys(), reverse=True):
            if level >= self.sample_level and level in other.sample:
                self.sample[level].intersection_update(other.sample[level])
                new_size += len(self.sample[level])
                if new_size > self.max_sample_size:
                    raise RuntimeError("Intersection larger than one sample.")
            else:
                self.sample.pop(level, None)
        self.cur_sample_size = new_size
        return self

    def set_intersect_size(self, intersect_size: int):
        self.intersect_size = intersect_size

    def reset(self):
        self.sample_level = 0
        self.cur_sample_size = 0
        self.t = 1
        self.sample.clear()

    def __str__(self):
        return str(self.sample)
    
    def __repr__(self):
        if self.cur_sample_size == 0:
            return f"Empty DS(sample_level={self.sample_level}, max_sample_size={self.max_sample_size})"
        return f"DS(sample_level={self.sample_level}, cur_sample_size={self.cur_sample_size}, " \
               f"intersect_size={self.intersect_size}, t={self.t}, max_sample_size={self.max_sample_size})"

