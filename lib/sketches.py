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


__all__ = ["H3HashFunctions", "HashFunctionFamily", "BaseSketch", "CountMinSketch", "BloomFilter"]



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

    
    def __init__(self, n_functions=0, limit=0, seed=42, bits=32, json_dict: dict = None):
        if json_dict == None:
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
            self.n_functions = json_dict["n_functions"]
            self.limit = json_dict["limit"]
            self.bits = json_dict["bits"]
            self.q_matrices = np.asarray(json_dict["q_matrices"])
            
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

class HashFunctionFamily:
    def __init__(self, num_hashes: int = None, max_range: int = None, seed=7, json_dict: dict = None):
        """
        Initialize a family of hash functions.
        :param num_hashes: Number of hash functions (depth of CMS).
        :param max_range: Maximum range (width of CMS rows).
        :param json_dict: Optional dictionary to initialize from JSON.
        """
        if json_dict is None:
            random.seed(seed)
            self.num_hashes = num_hashes
            self.max_range = max_range
            self.coefficients = [(random.randint(1, max_range), random.randint(0, max_range)) for _ in range(num_hashes)]
            self.prime = self._next_prime(max_range)
        else:
            self.num_hashes = json_dict["num_hashes"]
            self.max_range = json_dict["max_range"]
            self.coefficients = []
            for coeff in json_dict["coefficients"]:
                self.coefficients.append(tuple(coeff))
            self.prime = json_dict["prime"]
    
    def _next_prime(self, n):
        """Find the next prime number >= n."""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def hash(self, x, index):
        """
        Compute the hash value for an element x using the index-th hash function.
        :param x: Element to hash.
        :param index: Index of the hash function.
        :return: Hash value in the range [0, max_range).
        """
        a, b = self.coefficients[index]
        return (a * hash(x) + b) % self.prime % self.max_range
    
    def hash_value(self, x):
        """
        Compute the hash values for an element x using all hash functions.
        :param x: Element to hash.
        :return: List of hash values.
        """
        return [self.hash(x, i) for i in range(self.num_hashes)]
    
    def to_json(self) -> dict:
        """
        Convert the hash function family to a JSON-serializable dictionary.
        :return: Dictionary representation of the hash function family.
        """
        return {
            "num_hashes": self.num_hashes,
            "max_range": self.max_range,
            "coefficients": self.coefficients,
            "prime": self.prime
        }
    
    def __eq__(self, other):
        if not isinstance(other, HashFunctionFamily):
            return False
        return (
            self.num_hashes == other.num_hashes and
            self.max_range == other.max_range and
            self.coefficients == other.coefficients and
            self.prime == other.prime
        )



class BaseSketch:
    def update(self, element):
        pass

    def query(self, element):
        pass

    def merge(self, other: BaseSketch) -> BaseSketch:
        pass

    def add_privacy_noise(self, epsilon: float):
        pass
    
    def to_json(self) -> dict:
        pass


class CountMinSketchHadamard(BaseSketch):
    """A Count-Min sketch implementation using Hadamard transform."""

    depth: int
    width: int
    hash_functions: HashFunctionFamily
    counters: NDArray[np.int32]
    
    def __init__(self, width: int = None, depth: int = None, error_eps: float = None, delta: float = None, epsilon: float = None, seed: int = 7, json_dict: dict = None):
        if json_dict is None:
            self.processed_elements = 0

            if (width is not None and depth is not None and error_eps is None and delta is None):
                self.width = width
                self.depth = depth
                self.exactCounters = False
            elif(width is None and depth is None and error_eps is not None and delta is not None):
                self.width = int(np.ceil(math.e/error_eps))
                self.depth = int(np.ceil(np.log(1.0/delta)))
            else:
                raise Exception("Define either a valid width and depth or a valid epsilon and delta.")
 
            self.hash_functions = HashFunctionFamily(self.depth, self.width, seed=seed)
            self.epsilon = epsilon
            if epsilon is None:
                self.counters = np.zeros((self.depth, self.width), dtype=int)
            else:
                assert epsilon > 0, "Differential privacy parameter must be greater than 0."
                noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=(self.depth, self.width))
                self.counters = np.round(noise).astype(int)
        else:
            self.processed_elements = json_dict["processed_elements"]
            self.width = json_dict["width"]
            self.depth = json_dict["depth"]
            self.counters = np.asarray(json_dict["counters"])
            self.hash_functions = HashFunctionFamily(json_dict=json_dict["hash_functions"])
            self.epsilon = json_dict["epsilon"]          
        
    


class CountMinSketch(BaseSketch):
    """A basic implementation of a Count-Min sketch."""

    depth: int
    width: int
    hash_functions: HashFunctionFamily
    counters: NDArray[np.int32]
    
    def __init__(self, width: int = None, depth: int = None, error_eps: float = None, delta: float = None, epsilon: float = None, seed: int = 7, json_dict: dict = None):
        if json_dict is None:

            self.processed_elements = 0

            # if(width > 0 and depth > 0 and error_eps==0.0 and delta==0.0):
            if (width is not None and depth is not None and error_eps is None and delta is None):
                self.width = width
                self.depth = depth
                self.exactCounters = False
            elif(width is None and depth is None and error_eps is not None and delta is not None):
                # self.width = int(np.ceil(np.log(possibleValues)/error_eps))
                self.width = int(np.ceil(math.e/error_eps))
                self.depth = int(np.ceil(np.log(1.0/delta)))
            else:
                raise Exception("Define either a valid width and depth or a valid epsilon and delta.")
 
            # self.hash_functions = H3HashFunctions(self.depth,self.width,self.seed,self.bits)
            self.hash_functions = HashFunctionFamily(self.depth, self.width, seed=seed)
            self.epsilon = epsilon
            if epsilon is None:
                self.counters = np.zeros((self.depth, self.width), dtype=int)
            else:
                assert epsilon > 0, "Differential privacy parameter must be greater than 0."
                noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=(self.depth, self.width))
                self.counters = np.round(noise).astype(int)
        else:
            self.processed_elements = json_dict["processed_elements"]
            self.width = json_dict["width"]
            self.depth = json_dict["depth"]
            self.counters = np.asarray(json_dict["counters"])
            self.hash_functions = HashFunctionFamily(json_dict=json_dict["hash_functions"])
            self.epsilon = json_dict["epsilon"]          
        
    def update(self, element):
        indices = self.hash_functions.hash_value(element)
        temp = 0
        for idx in indices:
            self.counters[temp][idx]+=1
            temp+=1
        self.processed_elements+=1
        
    def query(self, element):
        result = math.inf
        indices = self.hash_functions.hash_value(element)
        temp = 0
        for idx in indices:
            if (self.counters[temp][idx] < result):
                result = self.counters[temp][idx]
            temp+=1
        return result
    
    def merge(self, other: CountMinSketch) -> CountMinSketch:
        """
        Merge another CountMinSketch into this one.
        :param other: Another CountMinSketch instance.
        """
        assert isinstance(other, CountMinSketch), "Can only merge with another CountMinSketch."
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge CountMinSketches with different dimensions.")
        if self.hash_functions != other.hash_functions:
            raise ValueError("Cannot merge CountMinSketches with different hash functions.")
        
        merged_sketch = copy.deepcopy(self)

        merged_sketch.counters += other.counters
        merged_sketch.processed_elements += other.processed_elements
        if merged_sketch.epsilon is not None and other.epsilon is not None:
            merged_sketch.epsilon += other.epsilon
        elif other.epsilon is not None:
            merged_sketch.epsilon = other.epsilon
        return merged_sketch
    
    def add_privacy_noise(self, epsilon: float):
        """
        Add Laplace noise to the counters for differential privacy.
        :param epsilon: Scale of the Laplace noise.
        """
        assert epsilon > 0, "Differential privacy parameter must be greater than 0."
        if self.epsilon is not None and self.epsilon > 0:
            self.epsilon += epsilon
        else:
            self.epsilon = epsilon

        noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=self.counters.shape)
        self.counters += np.round(noise).astype(int)
        
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
    
    def to_json(self) -> dict:
        return {
            "type": "CountMinSketch",
            "processed_elements": self.processed_elements,
            "width": self.width,
            "depth": self.depth,
            "counters": self.counters.tolist(),
            "hash_functions": self.hash_functions.to_json(),
            "epsilon": self.epsilon
        }
            
        
    def to_string(self, showMatrix=True):
        if(self.exactCounters):
            if showMatrix:
                return("Exact counter\nprocessed elements = "+str(self.processed_elements)+"\n"+str(self.counters))
            else:
                return("Exact counter\nprocessed elements = "+str(self.processed_elements))
        else:
            if showMatrix:
                return("Count-Min Sketch\ndepth = "+str(self.depth)+" width = "+str(self.width)+" ; processed elements = "+str(self.processed_elements)+"\n"+str(self.counters))
            else:
                return("Count-Min Sketch\ndepth = "+str(self.depth)+" width = "+str(self.width)+" ; processed elements = "+str(self.processed_elements))

    def __eq__(self, other):
        if not isinstance(other, CountMinSketch):
            return False
        return (
            self.width == other.width and
            self.depth == other.depth and
            np.array_equal(self.counters, other.counters) and
            self.hash_functions == other.hash_functions and
            self.processed_elements == other.processed_elements
        )


class BloomFilter(BaseSketch):
    def __init__(self, size: int =None, hash_count: int =None, n_values: int =None, p: float=None, epsilon: float = None, seed:int = 7, json_dict: dict = None):
        """
        Initialize the Bloom Filter.
        :param n_values: Estimated number of elements to store
        :param p: Desired false positive probability
        """
        if json_dict is None:
            if size is not None and hash_count is not None:
                self.size = size
                self.hash_count = hash_count
                self.hash_functions = HashFunctionFamily(hash_count, size, seed=seed)
            elif n_values is not None and p is not None:
                self.size = self._optimal_size(n_values, p)
                self.hash_count = self._optimal_hash_count(self.size, n_values)
                self.hash_functions = HashFunctionFamily(self.hash_count, self.size, seed=seed)
            else:
                raise ValueError("Invalid arguments. Provide either size and hash_count or n_values and p.")
            self.epsilon = epsilon
            if epsilon is None:
                self.bit_array = np.zeros(self.size, dtype=bool)
            else:
                assert epsilon > 0, "Differential privacy parameter must be greater than 0."
                flip_prob = 1 / (np.exp(epsilon) + 1)
                self.bit_array = np.random.rand(self.size) < flip_prob
            self.processed_elements = 0
        else:
            self.processed_elements = json_dict["processed_elements"]
            self.size = json_dict["size"]
            self.hash_count = json_dict["hash_count"]
            self.bit_array = np.asarray(json_dict["bit_array"], dtype=bool)
            self.hash_functions = HashFunctionFamily(json_dict=json_dict["hash_functions"])
            self.epsilon = json_dict["epsilon"]

    def _optimal_size(self, n_values, p):
        """
        Calculate the size of the bit array for given n_values and p.
        """
        return int(math.ceil(-(n_values * math.log(p)) / (math.log(2) ** 2)))

    def _optimal_hash_count(self, size, n_values):
        """
        Calculate the optimal number of hash functions (k).
        """
        return int(math.ceil((size / n_values) * math.log(2)))

    def update(self, element):
        """
        Add an item to the Bloom Filter.
        """
        indices = self.hash_functions.hash_value(element)
        for idx in indices:
            self.bit_array[idx] = True
        self.processed_elements += 1

    def query(self, element) -> bool:
        """
        Check if an item is in the Bloom Filter.
        Returns True if the item might be in the set, False if it's definitely not.
        """
        return all(self.bit_array[hash_val] for hash_val in self.hash_functions.hash_value(element))
    
    def merge(self, other: BloomFilter) -> BloomFilter:
        """
        Merge another Bloom Filter into this one.
        :param other: Another BloomFilter instance.
        """
        assert isinstance(other, BloomFilter), "Can only merge with another BloomFilter."
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Cannot merge BloomFilters with different sizes or hash counts.")
        if self.hash_functions != other.hash_functions:
            raise ValueError("Cannot merge BloomFilters with different hash functions.")
        
        merged_sketch = copy.deepcopy(self)

        merged_sketch.bit_array = np.logical_or(merged_sketch.bit_array, other.bit_array)
        merged_sketch.processed_elements += other.processed_elements
        if merged_sketch.epsilon is not None and other.epsilon is not None:
            merged_sketch.epsilon += other.epsilon
        elif other.epsilon is not None:
            merged_sketch.epsilon = other.epsilon
        return merged_sketch
    
    def add_privacy_noise(self, epsilon: float):
        """
        Add noise to the Bloom Filter for differential privacy.
        :param epsilon: Scale of the noise.
        """
        assert epsilon > 0, "Differential privacy parameter must be greater than 0."
        if self.epsilon is not None and self.epsilon > 0:
            self.epsilon += epsilon
        else:
            self.epsilon = epsilon
        flip_indices = np.random.rand(self.size) < (1 / (np.exp(epsilon) + 1))
        self.bit_array = np.logical_xor(self.bit_array, flip_indices)
    
    def to_json(self) -> dict:
        """
        Convert the Bloom Filter to a JSON-serializable dictionary.
        """
        return {
            "type": "BloomFilter",
            "processed_elements": self.processed_elements,
            "size": self.size,
            "hash_count": self.hash_count,
            "bit_array": self.bit_array.tolist(),
            "hash_functions": self.hash_functions.to_json(),
            "epsilon": self.epsilon
        }
    
    def __eq__(self, other):
        if not isinstance(other, BloomFilter):
            return False
        return (
            self.size == other.size and
            self.hash_count == other.hash_count and
            np.array_equal(self.bit_array, other.bit_array) and
            self.hash_functions == other.hash_functions
        )


