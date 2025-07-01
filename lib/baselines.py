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
from tqdm import tqdm

import orjson
import gzip

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product

from typing import List, Tuple, Dict, Any

__all__ = ["Baseline", "CentralDPServer", "LDPClient", "LDPServer", "LDPEncoderGRR", \
            "query_df", "infer_domains_and_ranges", "translate_query_region"]

def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([convert_np_types(v) for v in list(obj)])
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
    
def translate_query_region(region):
    cat_predicates = []
    for i, cat in enumerate(region[0]):
        if cat == "*":
            cat_predicates.append(cat)
        else:
            cat_predicates.append([cat])
    num_predicates = []
    for i, num in enumerate(region[1].b_adic_ranges):
        num_predicates.append((num.low, num.high-1))
    return cat_predicates + num_predicates

def query_df(df: pd.DataFrame, query: List[Any]) -> int:
    mask = pd.Series([True] * len(df))
    for col, predicate in zip(df.columns, query):
        if predicate == '*':
            continue
        elif isinstance(predicate, list) and len(predicate) == 2 and all(isinstance(x, (int, float)) for x in predicate):
            lower, upper = predicate
            mask &= ((df[col] >= lower) & (df[col] <= upper))
        elif isinstance(predicate, list) or isinstance(predicate, set):
            mask &= df[col].isin(predicate)
        else:
            raise ValueError(f"Unsupported query predicate: {predicate}")
    return int(mask.sum())

def infer_domains_and_ranges(df: pd.DataFrame):
    """
    Infers categorical domains and numerical ranges from a DataFrame.

    Args:
        df: The input DataFrame.
        categorical_threshold: Max number of unique values to treat a column as categorical.

    Returns:
        (categorical_domains, numerical_ranges): Tuple of dictionaries.
    """
    categorical_domains = {}
    numerical_ranges = {}

    for col in df.columns:
        if df[col].dtype == object:
            categorical_domains[col] = df[col].dropna().unique().tolist()
        elif np.issubdtype(df[col].dtype, np.number):
            numerical_ranges[col] = (df[col].min(), df[col].max())

    return categorical_domains, numerical_ranges


class Baseline:
    def query(self, element: list[Any]) -> float:
        pass

class CentralDPServer:
    def __init__(self, df: pd.DataFrame, epsilon: float):
        self.df = df
        self.remaining_budget = epsilon  # Track remaining budget
        self.columns = df.columns.tolist()

    def _apply_query(self, query):
        """Apply multidimensional query to the dataframe."""
        mask = pd.Series([True] * len(self.df))
        for col, q in zip(self.columns, query):
            if q == '*':
                continue
            elif isinstance(q, list) or isinstance(q, set):  # exact match list (e.g., ['UK'])
                mask &= self.df[col].isin(q)
            elif isinstance(q, tuple):  # range query
                lower, upper = q
                mask &= (self.df[col] >= lower) & (self.df[col] <= upper)
            else:
                raise ValueError(f"Unsupported query condition: {q}")
        return mask

    def query(self, query: list[Any], per_query_epsilon: float = None):
        """Executes a multidimensional count query with Laplace noise and tracks budget."""
        if per_query_epsilon is None:
            per_query_epsilon = self.remaining_budget
        if self.remaining_budget < per_query_epsilon:
            raise ValueError("Privacy budget exceeded.")

        self.remaining_budget -= per_query_epsilon
        true_count = self._apply_query(query).sum()
        scale = 1.0 / per_query_epsilon
        noisy_count = true_count + np.random.laplace(loc=0.0, scale=scale)
        return max(0, int(round(noisy_count)))

    def get_remaining_budget(self):
        return self.remaining_budget

class LDPClient:
    def __init__(self, df: pd.DataFrame, epsilon: float):
        self.df = df
        self.remaining_budget = epsilon
        self.columns = df.columns.tolist()

    def _apply_query(self, query):
        mask = pd.Series([True] * len(self.df))
        for col, q in zip(self.columns, query):
            if q == '*':
                continue
            elif isinstance(q, list):
                mask &= self.df[col].isin(q)
            elif isinstance(q, tuple):
                lower, upper = q
                mask &= (self.df[col] >= lower) & (self.df[col] <= upper)
            else:
                raise ValueError(f"Unsupported query condition: {q}")
        return mask

    def query(self, query: list[Any], per_query_epsilon: float = None):
        if per_query_epsilon is None:
            per_query_epsilon = self.remaining_budget
        if self.remaining_budget < per_query_epsilon:
            raise ValueError("Privacy budget exceeded.")
        
        self.remaining_budget -= per_query_epsilon
        true_count = self._apply_query(query).sum()
        scale = 1.0 / per_query_epsilon
        noise = np.random.laplace(loc=0.0, scale=scale)
        return true_count + noise

    def get_remaining_budget(self):
        return self.remaining_budget


class LDPServer:
    def __init__(self, df: pd.DataFrame, epsilon: float, number_of_partitions: int):
        self.clients: List[LDPClient] = []
        self.epsilon = epsilon
        self.per_client_epsilon = epsilon  # total budget per client

        partition_size = len(df) // number_of_partitions
        for i in range(number_of_partitions):
            start = i * partition_size
            end = None if i == number_of_partitions - 1 else (i + 1) * partition_size
            partition_df = df.iloc[start:end].reset_index(drop=True)
            self.clients.append(LDPClient(partition_df, self.per_client_epsilon))

    def query(self, query, per_query_epsilon):
        noisy_sum = 0
        for client in self.clients:
            noisy_sum += client.query(query, per_query_epsilon)
        return max(0, int(round(noisy_sum)))

    def get_remaining_budgets(self):
        return [client.get_remaining_budget() for client in self.clients]


class LDPEncoderGRR:
    def __init__(self, df: pd.DataFrame = None, epsilon: float = None, 
                 categorical_domains: Dict[str, List[Any]] = None, numerical_ranges: Dict[str, Tuple[float, float]] = None,
                 json_dict: Dict[str, Any] = None):
        if json_dict is None:
            assert df is not None, "DataFrame must be provided if json_dict is not given."
            assert epsilon is not None, "Epsilon must be provided if json_dict is not given."
            if categorical_domains is None or numerical_ranges is None:
                categorical_domains, numerical_ranges = infer_domains_and_ranges(df)

            self.columns = list(categorical_domains.keys()) + list(numerical_ranges.keys())
            self.epsilon = epsilon
            self.epsilon_per_attribute = epsilon / len(self.columns)
            self.cat_domains = categorical_domains
            self.num_ranges = numerical_ranges
            self.private_df = self.privatize(df)
        else:
            self.columns = json_dict["columns"]
            self.epsilon = json_dict["epsilon"]
            self.epsilon_per_attribute = json_dict["epsilon_per_attribute"]
            self.cat_domains = json_dict["cat_domains"]
            self.num_ranges = json_dict["num_ranges"]
            self.private_df = pd.read_parquet(json_dict["data_path"])

    def _perturb_categorical(self, value, domain):
        k = len(domain)
        p = np.exp(self.epsilon_per_attribute) / (np.exp(self.epsilon_per_attribute) + k - 1)
        if np.random.rand() < p:
            return value
        else:
            return np.random.choice([v for v in domain if v != value])

    def _perturb_numerical(self, value, lower, upper):
        scale = (upper - lower) / self.epsilon_per_attribute
        noisy = value + np.random.laplace(0, scale)
        return np.clip(noisy, lower, upper)

    def privatize(self, df: pd.DataFrame) -> pd.DataFrame:
        noisy_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Privatizing rows"):
            noisy_row = {}
            for col in df.columns:
                if col in self.cat_domains:
                    noisy_row[col] = self._perturb_categorical(row[col], self.cat_domains[col])
                elif col in self.num_ranges:
                    min_val, max_val = self.num_ranges[col]
                    noisy_row[col] = self._perturb_numerical(row[col], min_val, max_val)
                else:
                    noisy_row[col] = row[col]  # no transformation if not declared
            noisy_rows.append(noisy_row)
        return pd.DataFrame(noisy_rows)

    def query(self, query: List[Any]) -> int:
        return query_df(self.private_df, query)
    
    def save_to_file(self, file_path: str, data_path: str):
        """
        Save the private DataFrame and metadata to a file.
        """
        metadata = {
            "columns": self.columns,
            "epsilon": self.epsilon,
            "epsilon_per_attribute": self.epsilon_per_attribute,
            "cat_domains": self.cat_domains,
            "num_ranges": self.num_ranges,
            "data_path": data_path
        }
        metadata = convert_np_types(metadata)
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "wb") as f:
                f.write(orjson.dumps(metadata))
        else:
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(metadata))

        self.private_df.to_parquet(data_path, index=False)

    @staticmethod
    def load_from_file(file_path: str) -> LDPEncoderGRR:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, "rb") as f:
                data_bytes = f.read()
                json_dict = orjson.loads(data_bytes)
        else:
            with open(file_path, 'rb') as f:
                json_dict = orjson.loads(f.read())

        return LDPEncoderGRR(json_dict=json_dict)
        
