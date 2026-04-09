from __future__ import annotations

import math
from numbers import Number

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

def get_mask_from_query(df: pd.DataFrame, query: List[Any]) -> pd.Series:
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
    return mask

def query_df(df: pd.DataFrame, query: List[Any]) -> int:
    mask = get_mask_from_query(df, query)
    return int(mask.sum())

def query_df_sum(df: pd.DataFrame, query: List[Any], agg_col="n_extendedprice") -> Number:
    mask = get_mask_from_query(df, query)
    if agg_col == "q6":
        return (df.loc[mask, "n_extendedprice"] * df.loc[mask, "c_discount"]).astype(int).sum()
    return df[mask][agg_col].sum()

def query_df_avg(df: pd.DataFrame, query: List[Any], agg_col="n_extendedprice") -> Number:
    mask = get_mask_from_query(df, query)
    filtered_df = df[mask]
    if len(filtered_df) == 0:
        return 0
        
    if agg_col == "q6":
        return (df.loc[mask, "n_extendedprice"] * df.loc[mask, "c_discount"]).astype(int).mean()
    return df[mask][agg_col].mean()

def query_df_max(df: pd.DataFrame, query: List[Any], agg_col="n_extendedprice") -> Number:
    mask = get_mask_from_query(df, query)
    if agg_col == "q6":
        return (df.loc[mask, "n_extendedprice"] * df.loc[mask, "c_discount"]).astype(int).max()
    return df[mask][agg_col].max()

def query_df_min(df: pd.DataFrame, query: List[Any], agg_col="n_extendedprice") -> Number:
    mask = get_mask_from_query(df, query)
    if agg_col == "q6":
        return (df.loc[mask, "n_extendedprice"] * df.loc[mask, "c_discount"]).astype(int).min()
    return df[mask][agg_col].min()


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

def compute_true_sums(path_to_estimates: str, path_to_data:str, path_to_queries:str, limit=-1, agg_col="n_extendedprice"):
    results_df = pd.read_csv(path_to_estimates)
    if "true_counts" in results_df.columns:
        # print("The results dataframe already contains true counts.")
        return results_df
    
    data_df = pd.read_csv(path_to_data)
    if limit > 0:
        data_df = data_df.head(limit)
    len_df = len(data_df)

    if agg_col == "q6":
        total_mass = (data_df["n_extendedprice"] * data_df["c_discount"]).astype(int).sum()
    else:
        total_mass = (data_df[agg_col]).astype(int).sum()
    # total_mass
    
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_sums = np.empty(n_queries, dtype=np.int64)
    i = 0
    for query in tqdm(queries, desc="True Sum"):
        true_sums[i] = query_df_sum(data_df, query, agg_col=agg_col)
        i += 1

    results_df["true_sums"] = true_sums

    results_df["absolute_error"] = np.abs(results_df["true_sums"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / total_mass
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_sums"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_sums"] != 0
    )

    results_df.to_csv(path_to_estimates, index=False)
    return results_df

def compute_true_avgs(path_to_estimates: str, path_to_data:str, path_to_queries:str, limit=-1, agg_col="n_extendedprice"):
    results_df = pd.read_csv(path_to_estimates)
    # if "true_counts" in results_df.columns:
    #     # print("The results dataframe already contains true counts.")
    #     return results_df
    
    data_df = pd.read_csv(path_to_data)
    if limit > 0:
        data_df = data_df.head(limit)
    len_df = len(data_df)
    
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_avgs = np.empty(n_queries, dtype=np.float64)
    i = 0
    for query in tqdm(queries, desc="True Average"):
        true_avgs[i] = query_df_avg(data_df, query, agg_col=agg_col)
        i += 1

    results_df["true_avgs"] = true_avgs

    results_df["absolute_error"] = np.abs(results_df["true_avgs"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len_df
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_avgs"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_avgs"] != 0
    )

    results_df.to_csv(path_to_estimates, index=False)
    return results_df

def compute_true_maxs(path_to_estimates: str, path_to_data:str, path_to_queries:str, limit=-1, agg_col="n_extendedprice"):
    results_df = pd.read_csv(path_to_estimates)
    # if "true_maxs" in results_df.columns:
    #     # print("The results dataframe already contains true counts.")
    #     return results_df
    
    data_df = pd.read_csv(path_to_data)
    if limit > 0:
        data_df = data_df.head(limit)
    len_df = len(data_df)
    
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_maxs = np.empty(n_queries, dtype=np.int32)
    i = 0
    for query in tqdm(queries, desc="True Max"):
        true_res = query_df_max(data_df, query, agg_col=agg_col)
        if pd.isna(true_res):
            true_res = results_df["estimates"][i]
        true_maxs[i] = true_res
        i += 1

    results_df["true_maxs"] = true_maxs

    results_df["absolute_error"] = np.abs(results_df["true_maxs"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len_df
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_maxs"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_maxs"] != 0
    )

    results_df.to_csv(path_to_estimates, index=False)
    return results_df

def compute_true_mins(path_to_estimates: str, path_to_data:str, path_to_queries:str, limit=-1, agg_col="n_extendedprice"):
    results_df = pd.read_csv(path_to_estimates)
    # if "true_mins" in results_df.columns:
    #     # print("The results dataframe already contains true counts.")
    #     return results_df
    
    data_df = pd.read_csv(path_to_data)
    if limit > 0:
        data_df = data_df.head(limit)
    len_df = len(data_df)
    
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_mins = np.empty(n_queries, dtype=np.int32)
    i = 0
    for query in tqdm(queries, desc="True Min"):
        true_res = query_df_min(data_df, query, agg_col=agg_col)
        if pd.isna(true_res):
            true_res = results_df["estimates"][i]
        true_mins[i] = true_res
        i += 1

    results_df["true_mins"] = true_mins

    results_df["absolute_error"] = np.abs(results_df["true_mins"] - results_df["estimates"])
    results_df["normalized_error"] = results_df["absolute_error"] / len_df
    results_df["relative_error"] = np.divide(
        results_df["absolute_error"],
        results_df["true_mins"],
        out=np.zeros_like(results_df["absolute_error"], dtype=np.float64),
        where=results_df["true_mins"] != 0
    )

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


def get_true_counts(data_df:pd.DataFrame, path_to_queries:str):
    with open(path_to_queries, 'rb') as f:
        queries_json = orjson.loads(f.read())
    queries = queries_json['queries']
    n_queries = len(queries)
    true_counts = np.empty(n_queries, dtype=np.int32)
    i = 0
    for query in queries:
        true_counts[i] = query_df(data_df, query)
        i += 1
    return true_counts

def add_true_counts_and_error(path_to_estimates: str, len_df:int, true_counts:NDArray):
    results_df = pd.read_csv(path_to_estimates)
    # if "true_counts" in results_df.columns:
    #     # print("The results dataframe already contains true counts.")
    #     return results_df

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