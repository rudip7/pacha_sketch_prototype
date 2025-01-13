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
import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product


__all__ = ["H3HashFunctions", "CountMinSketch", "BAdicRange", "NumericRange"]



import hashlib
from ctypes import c_int32

def _int32(val: Any) -> int:
    """Hash an arbitrary value into a 32-bit integer."""
    if isinstance(val, int) and -2147483648 <= val <= 2147483647:
        return val

    if isinstance(val, tuple):
        # Convert tuple to string and hash it
        val = str(val)
    
    if isinstance(val, str):
        # Generate a consistent hash value for the string
        hash_value = int(hashlib.sha256(val.encode()).hexdigest(), 16) % (2**31 - 1) - 2**31
    else:
        hash_value = abs(hash(val))

    if -2147483648 <= hash_value <= 2147483647:
        return hash_value
    return c_int32(hash_value).value



class H3HashFunctions:
    """A collection of hash functions from the H3 hash family."""

    n_functions: int
    seed: int
    limit: int
    q_matrices: NDArray[np.int32]
    _rng: np.random.Generator

    
    def __init__(self, n_functions=0, limit=0, seed=42, bits=32, json_dic=None):
        if json_dic == None:
            self.n_functions = n_functions
            self.seed = seed
            self.limit = limit
            self.bits = int(bits) 
            if self.bits == 64:
                self.q_matrices = np.random.randint(int(-2**63+1),int(2**63-1),size=(n_functions,64), dtype='int64')
            elif self.bits < 64 and self.bits > 32:
                self.q_matrices = np.random.randint(int(-2**(self.bits-1)+1),int(2**(self.bits-1)-1),size=(n_functions,self.bits), dtype='int64')
            elif self.bits == 32:
                ii32 = np.iinfo(np.int32)
                self.q_matrices = np.random.randint(ii32.min, ii32.max, size=(n_functions,self.bits), dtype=np.int32) 
            elif self.bits < 32:  
                self.q_matrices = np.random.randint(int(-2**(self.bits-1)+1),int(2**(self.bits-1)-1),size=(n_functions,self.bits), dtype=np.int32) 
            else:
                print(self.bits)
        else:
            self.n_functions = json_dic["n_functions"]
            self.limit = json_dic["limit"]
            self.bits = json_dic["bits"]
            self.q_matrices = np.asarray(json_dic["q_matrices"])
            
    def hash_value(self, inp):
        if inp == 0:
            inp = 2**self.bits-1
        result = np.zeros(self.n_functions, dtype=int)
        inp_copy = inp
        for i in np.arange(self.n_functions, dtype=int):
            current = 0
            for k in np.arange(self.bits, dtype=int):
                if inp_copy == 0:
                    break
                current = current ^ ((1 & inp_copy) * self.q_matrices[i][k])
#                 current = int32(current)
#                 print("function: {}, bit: {}, current: {}, last: {}, change: {}".format(i, k, current, (1 & inp_copy),((1 & inp_copy) * self.q_matrices[i][k])))
                inp_copy >>= 1 
#             print("")
            result[i] = int(current)
            if (current < 0):
                current = -1 * current
            if (self.limit > 0):
                result[i] = int(current % self.limit)
            else:
                result[i] = int(current)
            
            inp_copy = inp
        return result

#     def toJSON(self):
#         return json.dumps(self.__dict__, cls=ComplexEncoder)
        
    def to_string(self):
        print("H3HashFunctions\nn_functions = "+str(self.n_functions)+"\nq_matrices = "+str(self.q_matrices))


class CountMinSketch:
    """A basic implementation of a Count-Min sketch."""

    depth: int
    width: int
    hash_functions: H3HashFunctions
    counters: NDArray[np.int32]
    
    def __init__(self, width=-1, depth=-1, eps=0.0, delta=0.0, min_val=-2**31, max_val=2**31-1, seed=42, ntype='int32', json_dic=None):
        if json_dic == None:
            if(ntype == 'int32'):
                self.bits = 32  
            elif (ntype == 'int64'):
                self.bits = 64
            self.elementsProcessed = 0

            if(width > 0 and depth > 0 and eps==0.0 and delta==0.0):
                self.width = width
                self.depth = depth
                self.exactCounters = False
            elif(width <= 0 and depth <= 0 and eps>0.0 and delta>0.0):
                self.min_val = min_val
                self.max_val = max_val
                possibleValues = max_val - min_val
                self.width = int(np.ceil(np.log(possibleValues)/eps))
                self.depth = int(np.ceil(np.log(1.0/delta)))
                if(self.width*self.depth > possibleValues):
    #                 print("For the required epsilon = "+str(eps)+", and delta = "+str(delta)+", and "+str(max_val)+" diferent elements is better to use exact counters instead of a Count-Min Sketch")
                    self.counters = np.zeros(possibleValues, dtype=int)
                    self.hash_functions = None
                    self.exactCounters = True
                else:
                    self.exactCounters = False
    #             print(str(self.width)+"   :;   "+str(self.depth))
            else:
                raise Exception("Define either a valid width and depth or a valid epsilon and delta.")
            if(self.exactCounters == False):
                self.seed = seed
                self.hash_functions = H3HashFunctions(self.depth,self.width,self.seed,self.bits)
                self.counters = np.zeros((self.depth, self.width), dtype=int)
        else:
            self.bits = json_dic["bits"] 
            self.elementsProcessed = json_dic["elementsProcessed"]
            self.width = json_dic["width"]
            self.depth = json_dic["depth"]
            self.exactCounters = json_dic["exactCounters"]
            self.min_val = json_dic["min_val"]
            self.max_val = json_dic["max_val"]
            self.counters = np.asarray(json_dic["counters"])
            self.hash_functions = H3HashFunctions(json_dic=json_dic["hash_functions"])            
        
    def update(self, element):
        if(self.exactCounters):
            if(element >= self.min_val and element <= self.max_val ):
                idx = self.getIndex(element)
                self.counters[idx] += 1
            else:
                return
        else:
            indices = self.hash_functions.hash_value(element)
            temp = 0
            for idx in indices:
                self.counters[temp][idx]+=1
                temp+=1
        self.elementsProcessed+=1
        
    def getIndex(self, element):
        return element-self.min_val
        
    def query(self, element):
        if(self.exactCounters):
            if(element >= self.min_val and element <= self.max_val ):
                idx = self.getIndex(element)
                return self.counters[idx]
            else:
                return 0
        else:
            result = 2**self.bits-1
            indices = self.hash_functions.hash_value(element)
            temp = 0
            for idx in indices:
                if (self.counters[temp][idx] < result):
                    result = self.counters[temp][idx]
                temp+=1
            return result
        
    def query_interval(self, low, high):
        result = 0
        for val in np.arange(low, high+1):
            result += self.query(val)
        return result
        
    def query_buckets(self, buckets):
        result = np.zeros(len(buckets))
        i = 0
        for b in buckets:
            result[i] = self.query_interval(b.low, b.high)
            i+=1
        return result
        
    def to_string(self, showMatrix=True):
        if(self.exactCounters):
            if showMatrix:
                return("Exact counter\nprocessed elements = "+str(self.elementsProcessed)+"\n"+str(self.counters))
            else:
                return("Exact counter\nprocessed elements = "+str(self.elementsProcessed))
        else:
            if showMatrix:
                return("Count-Min Sketch\ndepth = "+str(self.depth)+" width = "+str(self.width)+" ; processed elements = "+str(self.elementsProcessed)+"\n"+str(self.counters))
            else:
                return("Count-Min Sketch\ndepth = "+str(self.depth)+" width = "+str(self.width)+" ; processed elements = "+str(self.elementsProcessed))

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

def minimal_spatial_b_adic_cover(bounds: list, bases: list):
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