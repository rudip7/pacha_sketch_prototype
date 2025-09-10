from __future__ import annotations

import math

import numpy as np

import copy
import random

import pandas as pd

from typing import List, Any
from tqdm import tqdm

from sortedcontainers import SortedSet

__all__ = ["Kmin", "CountMin", "CountMinDyad", "OmniSketch", "sorted_intersection", "compute_max_B"]

class Kmin:
    sketch: SortedSet 

    def __init__(self, delta=0.05, max_sample_size=10_000):
        self.k = max_sample_size
        self.delta = delta
        self.cur_sample_size = 0
        self.n = 0
        self.simax = float('-inf')
        self.seed = None
        self.rng = random.Random()
        self.sketch: SortedSet = SortedSet([])  
        self.cur_tree_root = float('inf')
        self.b = int(math.ceil(math.log(4*max_sample_size**2.5 / delta)))
        self.max_hash = min(2 ** self.b, 2**31 - 1)

    def __copy__(self) -> Kmin:
        copy_kmin = Kmin(self.delta)
        copy_kmin.k = self.k
        copy_kmin.cur_sample_size = self.cur_sample_size
        copy_kmin.sketch = copy.deepcopy(self.sketch)
        return copy_kmin

    def hash(self, x: int) -> int:
        self.rng.seed(x)
        return self.rng.randint(0, self.max_hash)

    def add(self, hx: int):
        self.n += 1
        if self.cur_sample_size < self.k:
            self.sketch.add(hx)  # use negative to simulate max-heap
            self.cur_sample_size = len(self.sketch)
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
    def __init__(self, attr, width, depth, delta_ds=0.05, max_sample_size=10_000):
        self.attr = attr
        self.rng = random.Random()
        self.width = width
        self.depth = depth
        self.delta_ds = delta_ds
        self.max_sample_size = max_sample_size
        self.cm = [[None for _ in range(self.width)] for _ in range(self.depth)]
        self.init_sketch()

    def init_sketch(self):
        for j in range(self.depth):
            for i in range(self.width):
                self.cm[j][i] = Kmin(delta=self.delta_ds, max_sample_size=self.max_sample_size)

    def hash(self, attr_value: int, depth: int, width: int) -> list[int]:
        self.rng.seed(attr_value)
        return [self.rng.randint(0, width - 1) for _ in range(depth)]

    def add(self, id_: int, attr_value: int, hx: int):
        if not isinstance(attr_value, int):
            attr_value = hash(attr_value)
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


class CountMinDyad:
    def __init__(self, attr: int, interval_size: int, width: int, depth: int, delta_ds=0.05,
                 max_sample_size=10_000, dyadic_range_bits=16):
        self.attr = attr
        self.interval_size = interval_size
        self.width = width
        self.depth = depth
        self.delta_ds = delta_ds
        self.max_sample_size = max_sample_size
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
                self.cm[j][i] = Kmin(delta=self.delta_ds, max_sample_size=self.max_sample_size)

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


def compute_max_B(mem_budget, w, d, n_cat, n_num, dyadic_levels, delta):
    """
    Find the maximum integer B such that:
    mem_budget >= w * d * |A| * (32 + B * ceil(log(4 * B**2.5 / delta)) + 3*32 + 1)
    """
    factor = w * d * (n_cat + n_num * dyadic_levels)
    left, right = 1, 10**8
    max_B = 0
    while left <= right:
        B = (left + right) // 2
        ceil_log = math.ceil(math.log(4 * (B ** 2.5) / delta))
        rhs = factor * (32 + B * ceil_log + 3 * 32 + 1)
        if rhs <= mem_budget:
            max_B = B
            left = B + 1
        else:
            right = B - 1
    # print(f"Maximum B found: {max_B}, bits: {ceil_log}, rhs: {rhs}, Difference: {mem_budget - rhs}")
    return max_B

class OmniSketch:

    cmsketches: List[CountMin]
    cmsketches_range: List[List[CountMinDyad]]

    def __init__(self, cat_col_map, num_col_map,
                 delta=0.1, eps = 0.1, max_sample_size=10_000, 
                 dyadic_range_bits=16):
        
        self.depth = math.ceil(math.log(2/delta))
        self.width = 1+math.ceil(math.e * ((eps+1)/eps)**(1/self.depth))
        self.max_sample_size = max_sample_size
        self.num_attributes = len(cat_col_map) + len(num_col_map)
        self.dyadic_range_bits = dyadic_range_bits

        self.cat_col_map = cat_col_map
        self.num_col_map = num_col_map
        
        self.delta = delta
        self.eps = eps

        self.cmsketches = [None] * len(cat_col_map)
        for i in range(len(cat_col_map)):
            self.cmsketches[i] = CountMin(attr=i, width=self.width, depth=self.depth, delta_ds=delta/2, 
                                          max_sample_size=max_sample_size)

        self.cmsketches_range = [[None for _ in range(self.dyadic_range_bits + 1)] for _ in range(len(num_col_map))]
        for i in range(len(num_col_map)):
            for j in range(self.dyadic_range_bits, -1, -1):
                self.cmsketches_range[i][self.dyadic_range_bits - j] = CountMinDyad(attr=i, interval_size=j, 
                                                                                  width=self.width, depth=self.depth,
                                                                                  delta_ds=delta/2,
                                                                                  max_sample_size=max_sample_size, 
                                                                                  dyadic_range_bits=dyadic_range_bits)
        # Check parameters
        self.kmin_tmp = Kmin(delta=delta/2, max_sample_size=max_sample_size)
        
    def update(self, id_, element: tuple):
        hx = self.kmin_tmp.hash(id_)
        element : np.ndarray = np.asarray(element)
        cat_values = element[self.cat_col_map]
        num_values = element[self.num_col_map].astype(int)

        for i in range(len(self.cat_col_map)):  
            self.cmsketches[i].add(id_, cat_values[i], hx)

        for i in range(len(self.num_col_map)):
            ranges = self.wrapper_init_log_ranges(num_values[i])
            for j in range(len(ranges[2])):
                self.cmsketches_range[i][j].add(ranges[1][j], ranges[2][j], hx)
    
    def update_data_frame(self, df: pd.DataFrame) -> OmniSketch:
        for id_, row in tqdm(df.iterrows(), total=len(df), desc="Updating"):
            self.update(id_ + 1, row)
        return self

    def wrapper_init_log_ranges(self, l):
        if l < 0:
            print("Error: l < 0")
            raise ValueError("Log range cannot be negative")
        ranges = self.get_log_ranges(l + 1)
        for i in range(len(ranges[2])):
            ranges[1][i] -= 1
            ranges[2][i] -= 1
        return ranges

    def get_log_ranges(self, input_key):
        coeff = [0] * self.dyadic_range_bits
        lower_bound = [0] * self.dyadic_range_bits
        upper_bound = [0] * self.dyadic_range_bits

        half_point = int(math.pow(2, self.dyadic_range_bits - 1))
        if input_key < half_point:
            coeff[0] = 0
            lower_bound[0] = 1
            upper_bound[0] = half_point
        else:
            coeff[0] = 1
            lower_bound[0] = half_point
            upper_bound[0] = half_point * 2

        pow_ = half_point
        for i in range(1, self.dyadic_range_bits - 1):
            prev_coeff = coeff[i - 1]
            new_coeff_lower = prev_coeff * 2
            pow_ //= 2
            if (new_coeff_lower + 1) * pow_ < input_key:
                new_coeff_lower += 1
                coeff[i] = new_coeff_lower
            else:
                coeff[i] = new_coeff_lower
            lower_bound[i] = coeff[i] * pow_ + 1
            upper_bound[i] = (coeff[i] + 1) * pow_
        lower_bound[self.dyadic_range_bits - 1] = input_key
        upper_bound[self.dyadic_range_bits - 1] = input_key
        coeff[self.dyadic_range_bits - 1] = input_key

        return [coeff, lower_bound, upper_bound]
    

    def query(self, query: List[Any], details=False) -> int | tuple[int, bool]:
        cat_predicates = [query[i] for i in self.cat_col_map]
        num_predicates = [query[i] for i in self.num_col_map]

        n_predicates = 0
        s_cap = 0
        n_max = 0
        ns = np.zeros((self.num_attributes, self.depth), dtype=int)
        samples : List[Kmin] = []

        b_virtual = self.max_sample_size
        
        # Categorical predicates
        for i in range(len(cat_predicates)):
            if cat_predicates[i] != '*':
                n_predicates += 1
                attr = self.cat_col_map[i]
                dim_samples = self.cmsketches[i].query(cat_predicates[i])
                for d in range(self.depth):
                    samples.append(dim_samples[d])
                    ns[attr, d] += dim_samples[d].n
        
        # Numerical predicates
        for i in range(len(num_predicates)):
            if num_predicates[i] == '*':
                continue
            n_predicates += 1
            attr = self.num_col_map[i]
            lower = num_predicates[i][0]
            upper = num_predicates[i][1]

            ranges_list = self.wrap_log_ranges(lower, upper)
            dim_samples : List[Kmin] = [None] * self.depth
            # b_virtual does not make sense here, because gets updated in the loop
            # b_virtual = len(ranges_list) * self.max_sample_size
            for j in range(len(ranges_list)):
                cm: CountMinDyad = self.cmsketches_range[attr][self.get_index_of_range(ranges_list[j])]
                range_samples = cm.range_query(ranges_list[j][0], ranges_list[j][1])        
                for d in range(self.depth):
                    if dim_samples[d] is None:
                        dim_samples[d] = range_samples[d]
                    else:
                        dim_samples[d] = dim_samples[d].merge_samples(range_samples[d])
                    ns[attr, d] += range_samples[d].n
            for d in range(self.depth):
                samples.append(dim_samples[d])

        n_max = np.max(ns)
        s_cap = sorted_intersection(samples)

        constraint = 3 * math.log((4 * num_predicates * self.depth * math.sqrt(b_virtual)) / self.delta) / (self.eps * self.eps)
        estimate = math.ceil(s_cap * n_max / b_virtual)

        if s_cap < constraint:
            estimate = int(math.ceil(2 * n_max * math.log((4 * num_predicates * self.depth * math.sqrt(b_virtual)) / self.delta) / (b_virtual * self.eps * self.eps)))
            if details:
                return estimate, False
            return estimate
        else:
            if details:
                return estimate, True
            return estimate
        

    def point_query(self, query: List[Any]):
        cat_predicates = [query[i] for i in self.cat_col_map]
        # num_predicates = [query[i] for i in self.num_col_map]

        s_cap = 0
        n_max = 0
        samples = [] #[None] * len(cat_predicates)
        
        n_predicates = 0
        for i in range(len(cat_predicates)):
            if cat_predicates[i] != '*':
                n_predicates += 1
                temp = self.cmsketches[i].query(cat_predicates[i])
                samples.append(temp)
        n_max = self.get_nmax(samples)
        s_cap = sorted_intersection(samples)

        # Based on the formulas in the paper n_max is missing in the numerator and max_sample_size 
        # is missing in the denominator. This formula only works if n_max == max_sample_size
        constraint = 3 * math.log((4 * n_predicates * self.depth * math.sqrt(self.max_sample_size)) / self.delta) / (self.eps * self.eps)
        case2_estimate = math.ceil(s_cap * n_max / self.max_sample_size)

        intersect_size = s_cap
        if s_cap < constraint:
            return int(math.ceil(2 * n_max * math.log((4 * n_predicates * self.depth * math.sqrt(self.max_sample_size)) / self.delta) / (self.max_sample_size * self.eps * self.eps)))
        else:
            # q.thrm33_case2 = True
            return int(math.ceil(s_cap * n_max / self.max_sample_size))

    def get_nmax(self, samples: List[Kmin]):
        n_max = 0
        for sample in samples:
            if sample.n > n_max:
                n_max = sample.n
        return n_max

    def range_query(self, query, q):
        num_predicates = [query[i] for i in self.num_col_map]
        s_cap = 0
        n_max = 0
        b_virtual = 0
        samples = []
        ns = np.zeros((self.num_attributes, self.depth), dtype=int)

        b_virtual = 0
        num_predicates = 0
        for i in range(len(num_predicates)):
            if num_predicates[i] == '*':
                continue
            num_predicates += 1
            attr = self.num_col_map[i]
            lower = num_predicates[i][0]
            upper = num_predicates[i][1]

            ranges_list = self.wrap_log_ranges(lower, upper)
            samples : List[Kmin] = [None] * self.depth
            # b_virtual does not make sense here, because gets updated in the loop
            b_virtual = len(ranges_list) * self.max_sample_size
            for j in range(len(ranges_list)):
                cm: CountMinDyad = self.cmsketches_range[attr][self.get_index_of_range(ranges_list[j])]
                range_samples = cm.range_query(ranges_list[j][0], ranges_list[j][1])        
                for d in range(self.depth):
                    if samples[d] is None:
                        samples[d] = range_samples[d]
                    else:
                        samples[d] = samples[d].merge_samples(range_samples[d])
                    ns[attr, d] += range_samples[d].n

        n_max = np.max(ns)
        s_cap = sorted_intersection(samples)
        constraint = 3 * math.log((4 * num_predicates * self.depth * math.sqrt(b_virtual)) / self.delta) / (self.eps * self.eps)
        case2_estimate = math.ceil(s_cap * n_max / b_virtual)

        if s_cap < constraint:
            return int(math.ceil(2 * n_max * math.log((4 * num_predicates * self.depth * math.sqrt(b_virtual)) / self.delta) / (b_virtual * self.eps * self.eps)))
        else:
            return int(math.ceil(s_cap * n_max / b_virtual))

    def wrap_log_ranges(self, low, up):
        if low < 0 or up < 0:
            print(f"Error because low or up < 0: low: {low} up: {up}")
            exit(1)
        temp = self.get_log_ranges_arr_list(low + 1, up + 1, self.dyadic_range_bits)
        for i in temp:
            i[0] -= 1
            i[1] -= 1
        return temp
    
    def peek_parameters(self):
        print(f"OmniSketch parameters: \n"
                f"delta={self.delta}, eps={self.eps}\n"
                f"depth={self.depth}, width={self.width}\n"
                f"max_sample_size={self.max_sample_size}, bits={self.kmin_tmp.b}")
        
        print("----------------------------------------------------")

        factor = self.width * self.depth * (len(self.cat_col_map) + len(self.num_col_map) * self.dyadic_range_bits)
        ceil_log = math.ceil(math.log(4 * (self.max_sample_size ** 2.5) / self.delta))
        total_size = factor * (32 + self.max_sample_size * ceil_log + 3 * 32 + 1)
        total_size /= 8 * 1024 * 1024  # Convert bits to MB
        print(f"Total Size: {total_size} MB")
        

    # They call index, what we call level
    @staticmethod
    def get_index_of_range(longs):
        return int(math.log(longs[1] - longs[0] + 1) / math.log(2))

    # Minimal dyadic range cover.
    @staticmethod
    def get_log_ranges_arr_list(start_inclusive, stop_inclusive, dyadic_range_bits=16):
        start_inclusive -= 1
        stop_inclusive -= 1
        init_diff = stop_inclusive - start_inclusive + 1
        result = []
        total_sum = 0
        pow_ = 1
        for j in range(dyadic_range_bits):
            if start_inclusive + pow_ - 1 > stop_inclusive:
                break
            elif start_inclusive % (pow_ * 2) == 0 and start_inclusive + pow_ - 1 <= stop_inclusive:
                pass
            else:
                result.append([1 + start_inclusive, 1 + start_inclusive + pow_ - 1])
                total_sum += pow_
                start_inclusive += pow_
            pow_ *= 2

        pow_ = int(math.pow(2, dyadic_range_bits))
        for j in range(dyadic_range_bits, -1, -1):
            if start_inclusive % pow_ == 0 and start_inclusive + pow_ - 1 <= stop_inclusive:
                result.append([1 + start_inclusive, 1 + start_inclusive + pow_ - 1])
                total_sum += pow_
                start_inclusive += pow_
            pow_ //= 2

        if total_sum != init_diff:
            print("Error - no full coverage")
        return result

    def reset(self):
        for i in range(self.num_attributes):
            if self.has_predicate[i]:
                if self.range_queries:
                    for j in range(self.dyadic_range_bits + 1):
                        self.cmsketches_range[i][j].reset()
                else:
                    self.cmsketches[i].reset()

    @staticmethod
    def build_with_memory_budget(mem_budget: float, cat_col_map: List[int], num_col_map: List[int], 
                                 delta:float, eps:float, dyadic_range_bits=16) -> OmniSketch:
        
        mem_budget = mem_budget * 1024 * 1024 * 8  # Convert MB to bits
        d = math.ceil(math.log(2/delta))
        w = 1+math.ceil(math.e * ((eps+1)/eps)**(1/d)) 

        n_cat = len(cat_col_map)
        n_num = len(num_col_map)

        max_sample_size = compute_max_B(mem_budget, w, d, n_cat, n_num, dyadic_range_bits, delta)

        return OmniSketch(cat_col_map=cat_col_map, num_col_map=num_col_map,
                            delta=delta, eps=eps, max_sample_size=max_sample_size, 
                            dyadic_range_bits=dyadic_range_bits)

