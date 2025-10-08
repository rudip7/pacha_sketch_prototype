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

import gzip
from scipy.stats import entropy
import orjson
from tqdm import tqdm


import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product

from typing import List, Tuple, Dict, Any



__all__ = ["add_true_results"]

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


def add_true_results(path_to_estimates: str, path_to_true_counts: str, len_df: int) -> pd.DataFrame:
    true_df = pd.read_csv(path_to_true_counts)
    results_df = pd.read_csv(path_to_estimates)

    if "true_counts" in results_df.columns:
        # print("The results dataframe already contains true counts.")
        return results_df

    true_df = true_df[["baseline_runtimes", "true_counts"]]

    for col in true_df.columns:
        results_df[col] = true_df[col]

    results_df["absolute_error"] = np.abs(results_df["true_counts"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len_df
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_counts"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_counts"] != 0
    )
    # results_df["total_sketch_queries"] = results_df[
    #     ["relevant_nodes", "b_adic_cubes", "candidate_regions", "query_regions"]
    # ].sum(axis=1)

    results_df.to_csv(path_to_estimates, index=False)
    return results_df

def compute_true_counts(path_to_estimates: str, path_to_data:str, path_to_queries:str, limit=-1):
    results_df = pd.read_csv(path_to_estimates)
    if "true_counts" in results_df.columns:
        # print("The results dataframe already contains true counts.")
        return results_df
    
    data_df = pd.read_csv(path_to_data)
    if limit > 0:
        data_df = data_df.head(limit)
    len_df = len(data_df)
    
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_counts = np.empty(n_queries, dtype=np.int32)
    i = 0
    for query in tqdm(queries, desc="True Count"):
        true_counts[i] = query_df(data_df, query)
        i += 1

    results_df["true_counts"] = true_counts

    results_df["absolute_error"] = np.abs(results_df["true_counts"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len_df
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_counts"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_counts"] != 0
    )

    results_df.to_csv(path_to_estimates, index=False)
    return results_df