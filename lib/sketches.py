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

class HashFunctionFamily:
    def __init__(self, num_hashes, max_range):
        """
        Initialize a family of hash functions.
        :param num_hashes: Number of hash functions (depth of CMS).
        :param max_range: Maximum range (width of CMS rows).
        """
        self.num_hashes = num_hashes
        self.max_range = max_range
        self.coefficients = [(random.randint(1, max_range), random.randint(0, max_range)) for _ in range(num_hashes)]
        self.prime = self._next_prime(max_range)
    
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


class BaseSketch:
    def update(self, element):
        pass

    def query(self, element):
        pass


class CountMinSketch(BaseSketch):
    """A basic implementation of a Count-Min sketch."""

    depth: int
    width: int
    # hash_functions: H3HashFunctions
    hash_functions: HashFunctionFamily
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
                # self.width = int(np.ceil(np.log(possibleValues)/eps))
                self.width = int(np.ceil(math.e/eps))
                self.depth = int(np.ceil(np.log(1.0/delta)))
                if(self.width*self.depth > possibleValues):
    #                 print("For the required epsilon = "+str(eps)+", and delta = "+str(delta)+", and "+str(max_val)+" diferent elements is better to use exact counters instead of a Count-Min Sketch")
                    self.counters = np.zeros(possibleValues, dtype=int)
                    # self.hash_functions = None
                    self.exactCounters = True
                else:
                    self.exactCounters = False
    #             print(str(self.width)+"   :;   "+str(self.depth))
            else:
                raise Exception("Define either a valid width and depth or a valid epsilon and delta.")
            if(self.exactCounters == False):
                self.seed = seed
                # self.hash_functions = H3HashFunctions(self.depth,self.width,self.seed,self.bits)
                self.hash_functions = HashFunctionFamily(self.depth, self.width)
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
            # self.hash_functions = H3HashFunctions(json_dic=json_dic["hash_functions"])            
        
    def update(self, element):
        if(self.exactCounters):
            if(element >= self.min_val and element <= self.max_val ):
                idx = self.getIndex(element)
                self.counters[idx] += 1
            else:
                return
        else:
            # indices = self.hash_functions.hash_value(element)
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

class BloomFilter(BaseSketch):
    def __init__(self, size=None, hash_count=None, n_values=None, p=None):
        """
        Initialize the Bloom Filter.
        :param n_values: Estimated number of elements to store
        :param p: Desired false positive probability
        """
        if size is not None and hash_count is not None:
            self.size = size
            self.hash_count = hash_count
            self.hash_functions = HashFunctionFamily(hash_count, size)
            self.bit_array = np.zeros(size, dtype=bool)
        elif n_values is not None and p is not None:
            self.size = self._optimal_size(n_values, p)
            self.hash_count = self._optimal_hash_count(self.size, n_values)
            self.hash_functions = HashFunctionFamily(self.hash_count, self.size)
            self.bit_array = np.zeros(self.size, dtype=bool)
        else:
            raise ValueError("Invalid arguments. Provide either size and hash_count or n_values and p.")
        self.elementsProcessed = 0

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
        self.elementsProcessed += 1

    def query(self, element):
        """
        Check if an item is in the Bloom Filter.
        Returns True if the item might be in the set, False if it's definitely not.
        """
        return all(self.bit_array[hash_val] for hash_val in self.hash_functions.hash_value(element))


