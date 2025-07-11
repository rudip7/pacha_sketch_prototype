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
from .pacha_sketch import PachaSketch
from .encoders import BAdicCube, BAdicRange
import os

__all__ = ["Baseline", "CentralDPServer", "LDPClient", "LDPServer", "LDPEncoderGRR", \
            "query_df", "infer_domains_and_ranges", "translate_query_region"]


def check_accruracy(sketch: PachaSketch, df: pd.DataFrame, query_regions: List[List[Any]], level:int = None):
    if level is None:
        query_regions_level = query_regions
    else:
        query_regions_level = [region for region in query_regions if region[1].level == level]
    estimates = []
    print("Estimating with sketch...")
    for region in tqdm(query_regions_level, desc="Sketch Query"):
        num_region = region[1]
        estimates.append(sketch.base_sketches[num_region.level].query(region))
    estimates = np.asarray(estimates)

    true_counts = []
    print("Computing true counts...")
    for region in tqdm(query_regions_level, desc="True Count"):
        true_counts.append(query_df(df, translate_query_region(region)))
    true_counts = np.asarray(true_counts)

    abs_errors = np.abs(estimates - true_counts)
    rel_errors = abs_errors / len(df)

    print(f"Level {level}")
    print(f"Mean Abs Error: {np.mean(abs_errors)}, Mean Rel Error: {np.mean(rel_errors)}")
    print(f"Max Abs Error: {np.max(abs_errors)}, Max Rel Error: {np.max(rel_errors)}")
    print(f"Min Abs Error: {np.min(abs_errors)}, Min Rel Error: {np.min(rel_errors)}")
    print(f"Correct estimates: {np.sum(abs_errors == 0)}")

    return estimates, true_counts, abs_errors, rel_errors

def evaluate_queries_baselines(df: pd.DataFrame, queries: list, baseline: Baseline, path_to_file: str = None):
    print("Computing true counts...")
    true_counts = []
    for query in tqdm(queries, desc="True Count"):
        true_counts.append(query_df(df, query))
    true_counts = np.asarray(true_counts)
    
    print("Computing estimates...")
    estimates = []
    for query in tqdm(queries, desc="Estimates"):
        est = baseline.query(query)
        estimates.append(est)
    
    measurements = {
        "true_counts": true_counts,
        "estimates": estimates
    }

    results_df = pd.DataFrame(measurements)

    results_df["absolute_error"] = np.abs(results_df["true_counts"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len(df)
    results_df["relative_error"] = results_df["absolute_error"] / results_df["true_counts"]

    if path_to_file is not None:
        results_df.to_csv(path_to_file, index=False)

    return results_df

def query_regions_on_sketch(p_sketch: PachaSketch, unique_levels: np.ndarray, groups: List[NDArray]) -> int:
        estimate = 0
        for level, queries in zip(unique_levels, groups):
            estimate += np.sum(p_sketch.base_sketches[level].query_batch(queries))
        return estimate

def evaluate_equivalent_pacha_sketches(df: pd.DataFrame, queries: list, p_sketches: List[PachaSketch], dir_path: str, file_names: List[str]):
    if len(p_sketches) != len(file_names):
        raise ValueError("Number of PachaSketches must match number of file names.")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    print("Computing true counts...")
    true_counts = np.array([query_df(df, query) for query in tqdm(queries, desc="True Count")], dtype=np.int32)

    print("Computing estimates...")
    n_queries = len(queries)
    n_sketches = len(p_sketches)
    n_levels = p_sketches[0].levels
    # Preallocate
    estimates = np.empty((n_sketches, n_queries), dtype=np.float64)
    relevant_nodes = np.empty((n_sketches, n_queries), dtype=np.int32)
    cat_regions = np.empty((n_sketches, n_queries), dtype=np.int32)
    b_adic_cubes = np.empty((n_sketches, n_queries), dtype=np.int32)
    num_regions = np.empty((n_sketches, n_queries), dtype=np.int32)
    candidate_regions = np.empty((n_sketches, n_queries), dtype=np.int32)
    query_reg = np.empty((n_sketches, n_queries), dtype=np.int32)
    level_queries = np.zeros((n_sketches, n_queries, n_levels), dtype=np.int32)
    
    for i, query in enumerate(tqdm(queries, desc="Estimates")):
        p_sketch = p_sketches[0]
        query_regions, details = p_sketches[0].get_subqueries(query, detailed=True)
        level_col = query_regions[:, len(p_sketch.cat_col_map)]

        # Sort by level
        sorted_idx = np.argsort(level_col)
        sorted_queries = query_regions[sorted_idx]
        sorted_levels = level_col[sorted_idx]

        # Find where the value changes
        _, group_boundaries = np.unique(sorted_levels, return_index=True)

        # Split into groups
        groups = np.split(sorted_queries, group_boundaries[1:])

        unique_levels = sorted_levels[group_boundaries].astype(int)
        
        for j, p_sketch in enumerate(p_sketches):
            est = query_regions_on_sketch(p_sketch, unique_levels, groups)
            estimates[j, i] = est
            relevant_nodes[j, i] = details["relevant_nodes"]
            cat_regions[j, i] = details["cat_regions"]
            b_adic_cubes[j, i] = details["b_adic_cubes"]
            num_regions[j, i] = details["num_regions"]
            candidate_regions[j, i] = details["candidate_regions"]
            query_reg[j, i] = details["query_regions"]
            level_queries[j, i, :len(details["queries_per_level"])] = details["queries_per_level"]
        
    for j, p_sketch in enumerate(p_sketches):
        measurements = {
            "true_counts": true_counts,
            "estimates": estimates[j],
            "relevant_nodes": relevant_nodes[j],
            "cat_regions": cat_regions[j],
            "b_adic_cubes": b_adic_cubes[j],
            "num_regions": num_regions[j],
            "candidate_regions": candidate_regions[j],
            "query_regions": query_reg[j]
        }

        for lvl in range(n_levels):
            measurements[f"level_{lvl}_queries"] = level_queries[j, :, lvl]
        
        results_df = pd.DataFrame(measurements)
        # Vectorized error calculations
        results_df["absolute_error"] = np.abs(results_df["true_counts"] - results_df["estimates"])
        results_df["normalized_error"] = results_df["absolute_error"] / len(df)
        results_df["relative_error"] = np.divide(
            results_df["absolute_error"],
            results_df["true_counts"],
            out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
            where=results_df["true_counts"] != 0
        )
        results_df["total_sketch_queries"] = results_df[
            ["relevant_nodes", "b_adic_cubes", "candidate_regions", "query_regions"]
        ].sum(axis=1)

        
        file_name = file_names[j]
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        results_df.to_csv(f"{dir_path}/{file_name}", index=False)
    
    


def evaluate_queries(df: pd.DataFrame, queries: list, p_sketch: PachaSketch, path_to_file: str = None):
    print("Computing true counts...")
    true_counts = np.array([query_df(df, query) for query in tqdm(queries, desc="True Count")], dtype=np.int32)

    print("Computing estimates...")
    n_queries = len(queries)
    n_levels = p_sketch.levels

    # Preallocate
    estimates = np.empty(n_queries, dtype=np.float64)
    relevant_nodes = np.empty(n_queries, dtype=np.int32)
    cat_regions = np.empty(n_queries, dtype=np.int32)
    b_adic_cubes = np.empty(n_queries, dtype=np.int32)
    num_regions = np.empty(n_queries, dtype=np.int32)
    candidate_regions = np.empty(n_queries, dtype=np.int32)
    query_regions = np.empty(n_queries, dtype=np.int32)
    level_queries = np.zeros((n_queries, n_levels), dtype=np.int32)
    runtimes = np.zeros(n_queries, dtype=np.float64)

    for i, query in enumerate(tqdm(queries, desc="Estimates")):
        start_time = time.time()
        est, details = p_sketch.query(query, detailed=True)
        runtimes[i] = time.time() - start_time
        estimates[i] = est
        relevant_nodes[i] = details["relevant_nodes"]
        cat_regions[i] = details["cat_regions"]
        b_adic_cubes[i] = details["b_adic_cubes"]
        num_regions[i] = details["num_regions"]
        candidate_regions[i] = details["candidate_regions"]
        query_regions[i] = details["query_regions"]
        level_queries[i, :len(details["queries_per_level"])] = details["queries_per_level"]

    # Construct dataframe in one go
    measurements = {
        "runtimes": runtimes,
        "true_counts": true_counts,
        "estimates": estimates,
        "relevant_nodes": relevant_nodes,
        "cat_regions": cat_regions,
        "b_adic_cubes": b_adic_cubes,
        "num_regions": num_regions,
        "candidate_regions": candidate_regions,
        "query_regions": query_regions
    }

    for lvl in range(n_levels):
        measurements[f"level_{lvl}_queries"] = level_queries[:, lvl]

    results_df = pd.DataFrame(measurements)

    # Vectorized error calculations
    results_df["absolute_error"] = np.abs(results_df["true_counts"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len(df)
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_counts"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_counts"] != 0
    )
    results_df["total_sketch_queries"] = results_df[
        ["relevant_nodes", "b_adic_cubes", "candidate_regions", "query_regions"]
    ].sum(axis=1)

    if path_to_file is not None:
        results_df.to_csv(path_to_file, index=False)

    return results_df

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

def translate_cube_to_query(cat_predicates: list, b_adic_cube: BAdicCube):
    query = cat_predicates.copy()
    for b_range in b_adic_cube.b_adic_ranges:
        query.append((b_range.low, b_range.high-1))
    return query

def filter_df(df: pd.DataFrame, query: List[Any]) -> pd.DataFrame:
    """
    Filters a DataFrame based on a multidimensional query.

    Args:
        df: The input DataFrame.
        query: A list of predicates for each column. Use '*' for wildcard (no filter).

    Returns:
        A filtered DataFrame.
    """
    mask = pd.Series([True] * len(df))
    for col, predicate in zip(df.columns, query):
        if predicate == '*':
            continue
        elif (isinstance(predicate, tuple) or isinstance(predicate, list)) and len(predicate) == 2 and all(isinstance(x, (int, float)) for x in predicate):
            lower, upper = predicate
            mask &= ((df[col] >= lower) & (df[col] <= upper))
        elif isinstance(predicate, list) or isinstance(predicate, set):
            mask &= df[col].isin(predicate)
        else:
            raise ValueError(f"Unsupported query predicate: {predicate}")
    return df[mask]

def query_df(df: pd.DataFrame, query: List[Any]) -> int:
    mask = pd.Series([True] * len(df))
    for col, predicate in zip(df.columns, query):
        if predicate == '*':
            continue
        elif (isinstance(predicate, tuple) or isinstance(predicate, list)) and len(predicate) == 2 and all(isinstance(x, (int, float)) for x in predicate):
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

class CentralDPServer(Baseline):
    def __init__(self, df: pd.DataFrame, epsilon: float, per_query_epsilon: float = None):
        self.df = df
        self.epsilon = epsilon
        self.remaining_budget = epsilon  # Track remaining budget
        self.per_query_epsilon = per_query_epsilon
        self.columns = df.columns.tolist()

    def reset_budget(self, epsilon: float = None, per_query_epsilon: float = None):
        """Reset the privacy budget to a new value."""
        if epsilon is not None:
            self.epsilon = epsilon
        if per_query_epsilon is not None:
            self.per_query_epsilon = per_query_epsilon
        self.remaining_budget = self.epsilon

    
    def query(self, query: list[Any]):
        """Executes a multidimensional count query with Laplace noise and tracks budget."""
        return self.query_with_budget(query=query, per_query_epsilon=self.per_query_epsilon)


    def query_with_budget(self, query: list[Any], per_query_epsilon: float = None):
        """Executes a multidimensional count query with Laplace noise and tracks budget."""
        if per_query_epsilon is None:
            if self.per_query_epsilon is not None:
                per_query_epsilon = self.per_query_epsilon
            else:
                per_query_epsilon = self.remaining_budget
        if self.remaining_budget - per_query_epsilon < 0:
            raise ValueError("Privacy budget exceeded.")

        self.remaining_budget -= per_query_epsilon
        true_count = query_df(self.df, query)
        scale = 1.0 / per_query_epsilon
        noisy_count = true_count + np.random.laplace(loc=0.0, scale=scale)
        return max(0, int(round(noisy_count)))

    def get_remaining_budget(self):
        return self.remaining_budget

class LDPClient:
    def __init__(self, df: pd.DataFrame, epsilon: float, per_query_epsilon: float = None):
        self.df = df
        self.epsilon = epsilon
        self.remaining_budget = epsilon
        self.per_query_epsilon = per_query_epsilon
        self.columns = df.columns.tolist()


    def reset_budget(self, epsilon: float = None, per_query_epsilon: float = None):
        """Reset the privacy budget to a new value."""
        if epsilon is not None:
            self.epsilon = epsilon
        if per_query_epsilon is not None:
            self.per_query_epsilon = per_query_epsilon
        self.remaining_budget = self.epsilon

    def query(self, query: list[Any], per_query_epsilon: float = None):
        if per_query_epsilon is None:
            if self.per_query_epsilon is not None:
                per_query_epsilon = self.per_query_epsilon
            else:
                per_query_epsilon = self.remaining_budget
        # if self.remaining_budget < per_query_epsilon:
        #     raise ValueError("Privacy budget exceeded.")
        
        self.remaining_budget -= per_query_epsilon
        true_count = query_df(self.df, query)
        scale = 1.0 / per_query_epsilon
        noise = np.random.laplace(loc=0.0, scale=scale)
        return true_count + noise

    def get_remaining_budget(self):
        return self.remaining_budget


class LDPServer(Baseline):
    def __init__(self, df: pd.DataFrame, epsilon: float, number_of_partitions: int, per_query_epsilon: float = None):
        self.clients: List[LDPClient] = []
        self.epsilon = epsilon
        self.per_client_epsilon = epsilon  # total budget per client
        self.per_query_epsilon = per_query_epsilon 

        partition_size = len(df) // number_of_partitions
        for i in range(number_of_partitions):
            start = i * partition_size
            end = None if i == number_of_partitions - 1 else (i + 1) * partition_size
            partition_df = df.iloc[start:end].reset_index(drop=True)
            self.clients.append(LDPClient(partition_df, self.per_client_epsilon))
    
    def reset_budget(self, epsilon: float = None, per_query_epsilon: float = None):
        """Reset the privacy budget for all clients."""
        if epsilon is not None:
            self.epsilon = epsilon
            self.per_client_epsilon = epsilon
        if per_query_epsilon is not None:
            self.per_query_epsilon = per_query_epsilon
        
        for client in self.clients:
            client.reset_budget(epsilon=self.per_client_epsilon, per_query_epsilon=self.per_query_epsilon)

    def query(self, query: List[Any]):
        return self.query_with_budget(query=query, per_query_epsilon=self.per_query_epsilon)

    def query_with_budget(self, query: List[Any], per_query_epsilon: float = None):
        noisy_sum = 0
        for client in self.clients:
            noisy_sum += client.query(query, per_query_epsilon)
        return max(0, int(round(noisy_sum)))

    def get_remaining_budgets(self):
        return [client.get_remaining_budget() for client in self.clients]


class LDPEncoderGRR(Baseline):
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
        
