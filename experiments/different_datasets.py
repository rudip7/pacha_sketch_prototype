import sys
# append the path of the parent directory
sys.path.append("..")

import math
import os
import time


import numpy as np
np.set_printoptions(legacy='1.25')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product
import copy


from tqdm import tqdm

from scipy.stats import pearsonr
from importlib import reload

import orjson
import gzip

from scipy.stats import entropy
from pympler import asizeof


from lib import sketches, visualization_utils, encoders, ploting, pacha_sketch
reload(ploting)
reload(sketches)
reload(visualization_utils)
reload(encoders)

reload(pacha_sketch)

from lib.sketches import BloomFilter, CountMinSketch, H3HashFunctions, HashFunctionFamily,\
      CountMinSketchHadamard, CountMinSketchLocalHashing, deterministic_hash, simple_deterministic_hash
from lib.visualization_utils import visualize_badic_cover, plot_b_adic_cubes
from lib.encoders import minimal_b_adic_cover, minimal_spatial_b_adic_cover, BAdicCube, BAdicRange, \
      minimal_b_adic_cover_array, downgrade_b_adic_range_indices
from lib.pacha_sketch import PachaSketch, ADTree, BFParameters, CMParameters, \
    cartesian_product, get_n_updates, get_n_updates_customized, MaterializedCombinations

from lib.ploting import set_style, plot_ylabel, plot_legend

from lib import baselines
reload(baselines)

from lib.baselines import CentralDPServer, LDPServer, LDPEncoderGRR, filter_df, query_df, \
      infer_domains_and_ranges, translate_query_region, evaluate_queries, check_accruracy, \
      evaluate_queries_baselines, evaluate_equivalent_pacha_sketches


## Section 1. Accuracy Experiments Different Datasets

def build_pacha_sketch_for_retail(len_df, delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=5) -> PachaSketch:
    """
    Builds a PachaSketch for the retail dataset with specified parameters.
    """
    # Retail
    cat_col_map_retail = [0, 1, 2]
    num_col_map_retail = [3, 4, 5]
    bases_retail = [4, 2, 2]

    n_cat_retail = len(cat_col_map_retail)
    n_num_retail = len(num_col_map_retail)

    cat_updates_retail, num_updates_retail, region_updates_retail = get_n_updates(n_cat_retail, n_num_retail, levels)

    retail_ad_tree = ADTree.from_json("../sketches/ad_trees/online_retail.json")
    error_eps = rel_eps / region_updates_retail
    retail_p_sketch = PachaSketch.build_with_uniform_size(
        levels=levels,
        num_dimensions=n_cat_retail + n_num_retail,
        cat_col_map=cat_col_map_retail,
        num_col_map=num_col_map_retail,
        bases=bases_retail,
        ad_tree=retail_ad_tree,
        cm_params=CMParameters(delta=delta, error_eps=error_eps),
        cat_index_parameters=BFParameters(n_values=len_df * cat_updates_retail, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len_df * num_updates_retail, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len_df * region_updates_retail, p=bloom_p))

    retail_p_sketch.get_size()
    print(f"Theoretical size: {retail_p_sketch.get_size()}")
    total_size = asizeof.asizeof(retail_p_sketch)
    print(f"Total size of tpch_p_sketch: {total_size / 1024/1024} MB")

    return retail_p_sketch

def build_pacha_sketch_for_bank(len_df, delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=5) -> PachaSketch:
    """
    Builds a PachaSketch for the bank marketing dataset with specified parameters.
    """
    # Bank Marketing
    cat_col_map_bank = [0, 1, 2, 3, 4, 5]
    num_col_map_bank = [6, 7, 8, 9]
    bases_bank = [4, 5, 2, 2]

    n_cat_bank = len(cat_col_map_bank)
    n_num_bank = len(num_col_map_bank)


    num_cols = ['duration', 'balance', 'age', 'date']
    relevant_combinations = [
        ['duration'],
        ['balance'],
        ['age'],
        ['date'],
        
        ['duration', 'age'],
        ['duration', 'date'],
        ['duration', 'balance'],
        ['balance', 'age'],
        ['balance', 'date'],
        
        ['duration', 'balance', 'age'],
        
        ['duration', 'balance', 'age', 'date']
        ]

    mat_combinations = MaterializedCombinations(col_names=num_cols, relevant_combinations=relevant_combinations)

    ad_tree_levels = len(cat_col_map_bank)
    num_combinations = len(relevant_combinations)
    cat_updates_bank, num_updates_bank, region_updates_bank = get_n_updates_customized(ad_tree_levels, num_combinations, levels)

    
    bank_ad_tree = ADTree.from_json("../sketches/ad_trees/bank_marketing.json")
    error_eps = rel_eps / region_updates_bank
    bank_p_sketch = PachaSketch.build_with_uniform_size(
        levels=levels,
        num_dimensions=n_cat_bank + n_num_bank,
        cat_col_map=cat_col_map_bank,
        num_col_map=num_col_map_bank,
        bases=bases_bank,
        ad_tree=bank_ad_tree,
        cm_params=CMParameters(delta=delta, error_eps=error_eps),
        cat_index_parameters=BFParameters(n_values=len_df * cat_updates_bank, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len_df * num_updates_bank, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len_df * region_updates_bank, p=bloom_p))

    bank_p_sketch.materialized = mat_combinations

    bank_p_sketch.get_size()
    print(f"Theoretical size: {bank_p_sketch.get_size()}")
    total_size = asizeof.asizeof(bank_p_sketch)
    print(f"Total size of bank_p_sketch: {total_size / 1024/1024} MB")

    return bank_p_sketch

def build_pacha_sketch_for_census(len_df, delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=5) -> PachaSketch :
    """
    Builds a PachaSketch for the census dataset with specified parameters.
    """
    # Census
    cat_col_map_census = [0, 1, 2, 3, 4, 5, 6]
    num_col_map_census = [7, 8, 9]
    bases_census = [2, 10, 2]

    n_cat_census = len(cat_col_map_census)
    n_num_census = len(num_col_map_census)

    ad_tree_levels = len(cat_col_map_census)
    num_combinations = 2**len(num_col_map_census) - 1

    cat_updates_census, num_updates_census, region_updates_census = get_n_updates_customized(ad_tree_levels, num_combinations, levels)

    census_ad_tree = ADTree.from_json("../sketches/ad_trees/acs_folktables.json")
    error_eps = rel_eps / region_updates_census
    census_p_sketch = PachaSketch.build_with_uniform_size(
        levels=levels,
        num_dimensions=n_cat_census + n_num_census,
        cat_col_map=cat_col_map_census,
        num_col_map=num_col_map_census,
        bases=bases_census,
        ad_tree=census_ad_tree,
        cm_params=CMParameters(delta=delta, error_eps=error_eps),
        cat_index_parameters=BFParameters(n_values=len_df * cat_updates_census, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len_df * num_updates_census, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len_df * region_updates_census, p=bloom_p))

    census_p_sketch.get_size()
    print(f"Theoretical size: {census_p_sketch.get_size()}")
    total_size = asizeof.asizeof(census_p_sketch)
    print(f"Total size of census_p_sketch: {total_size / 1024/1024} MB")
    
    return census_p_sketch

def build_pacha_sketch_for_tpch(len_df, delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=5):
    cat_col_map_tpch = [0, 1, 2, 3, 4]
    n_cat_tpch = len(cat_col_map_tpch)
    num_col_map_tpch = [5, 6, 7, 8, 9]
    n_num_tpch = len(num_col_map_tpch)
    bases_tpch = [5,5,5,10,2]

    ad_tree_levels = len(cat_col_map_tpch)
    num_combinations = 13
    cat_updates_tpch, num_updates_tpch, region_updates_tpch = get_n_updates_customized(ad_tree_levels, num_combinations, levels)

    tpch_ad_tree = ADTree.from_json("../sketches/ad_trees/tpch_lineitem.json")
    error_eps = rel_eps / region_updates_tpch
    tpch_p_sketch = PachaSketch.build_with_uniform_size(
        levels=levels,
        num_dimensions=n_cat_tpch + n_num_tpch,
        cat_col_map=cat_col_map_tpch,
        num_col_map=num_col_map_tpch,
        bases=bases_tpch,
        ad_tree=tpch_ad_tree,
        cm_params=CMParameters(delta=delta, error_eps=error_eps),
        cat_index_parameters=BFParameters(n_values=len_df * cat_updates_tpch, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len_df * num_updates_tpch, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len_df * region_updates_tpch, p=bloom_p))

    col_names = ['n_shipdate', 'n_commitdate', 'n_receiptdate', 'n_extendedprice', 'n_quantity']
    relevant_combinations = [
        ['n_shipdate'],
        ['n_commitdate'],
        ['n_receiptdate'],
        ['n_extendedprice'],
        ['n_quantity'],
        
        ['n_shipdate', 'n_commitdate'],
        ['n_shipdate', 'n_receiptdate'],
        ['n_shipdate', 'n_quantity'],
        ['n_commitdate', 'n_receiptdate'],
        ['n_commitdate', 'n_extendedprice'],
        ['n_extendedprice', 'n_quantity'],
        
        ['n_commitdate', 'n_receiptdate', 'n_extendedprice'],
        
        ['n_shipdate', 'n_commitdate', 'n_receiptdate', 'n_extendedprice', 'n_quantity']
        ]
    mat_combinations = MaterializedCombinations(col_names=col_names, relevant_combinations=relevant_combinations)
    tpch_p_sketch.materialized = mat_combinations

    tpch_p_sketch.get_size()
    print(f"Theoretical size: {tpch_p_sketch.get_size()}")
    total_size = asizeof.asizeof(tpch_p_sketch)
    print(f"Total size of tpch_p_sketch: {total_size / 1024/1024} MB")
    
    return tpch_p_sketch


def evaluate_all_queries(df, p_sketch, n_cat, n_num, dataset_name):

    print(f"Evaluating queries for {dataset_name} dataset...")
    print("Evaluating random queries...")
    query_path = f"../queries/{dataset_name}/{dataset_name}_random.json"
    with open(query_path, 'rb') as f:
        queries = orjson.loads(f.read())
    results = evaluate_queries(df, queries['queries'], p_sketch, path_to_file=f"../results/{dataset_name}/{dataset_name}_random.csv")

    print("Evaluating selectivity queries...")
    selectivities = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    for sel in selectivities:
        print(f"Evaluating selectivity {sel}...")
        query_path = f"../queries/{dataset_name}/selectivities/{dataset_name}_sel_{sel}.json"
        with open(query_path, 'rb') as f:
            queries = orjson.loads(f.read())
        results = evaluate_queries(df, queries['queries'], p_sketch, path_to_file=f"../results/{dataset_name}/selectivities/{dataset_name}_sel_{sel}.csv")
    
    print("Evaluating categorical queries...")
    n_cats = np.arange(1, n_cat+1)
    for n in n_cats:
        print(f"Evaluating categorical queries with {n} predicates...")
        query_path = f"../queries/{dataset_name}/categorical/{dataset_name}_cat_{n}.json"
        with open(query_path, 'rb') as f:
            queries = orjson.loads(f.read())
        results = evaluate_queries(df, queries['queries'], p_sketch, path_to_file=f"../results/{dataset_name}/categorical/{dataset_name}_cat_{n}.csv")

    print("Evaluating numerical queries...")
    n_nums = np.arange(1, n_num+1)
    for n in n_nums:
        print(f"Evaluating numerical queries with {n} predicates...")
        query_path = f"../queries/{dataset_name}/numerical/{dataset_name}_num_{n}.json"
        with open(query_path, 'rb') as f:
            queries = orjson.loads(f.read())
        results = evaluate_queries(df, queries['queries'], p_sketch, path_to_file=f"../results/{dataset_name}/numerical/{dataset_name}_num_{n}.csv")
    
    print("Evaluating mixed queries...")
    n_diminant = max(n_cat, n_num)
    n_mix = np.arange(1, n_diminant+1)
    for n in n_mix:
        print(f"Evaluating mixed queries with {n} predicates...")
        query_path = f"../queries/{dataset_name}/mixed/{dataset_name}_mix_{n}.json"
        with open(query_path, 'rb') as f:
            queries = orjson.loads(f.read())
        results = evaluate_queries(df, queries['queries'], p_sketch, path_to_file=f"../results/{dataset_name}/mixed/{dataset_name}_mix_{n}.csv")

    print(f"\nDone evaluating queries for {dataset_name} dataset.")

    
def main():
    """
    Main function to build PachaSketches for different datasets.
    """
    # Parameters
    delta=0.01
    rel_eps=0.0005
    p = 0.01
    levels=5

    multiprocessing = False

    print(f"Building PachaSketches with p: {p}, rel_eps: {rel_eps}") 
    
    # Build PachaSketch for Bank Marketing dataset
    print("\nBank\n")
    bank_df = pd.read_parquet("../data/clean/bank_marketing.parquet")
    bank_p_sketch = build_pacha_sketch_for_bank(len(bank_df), delta, rel_eps, p, levels)
    # bank_df=bank_df.head(10)  

    if multiprocessing:
        bank_p_sketch = bank_p_sketch.update_data_frame_multiprocessing(bank_df, workers=4)
    else:   
        bank_p_sketch = bank_p_sketch.update_data_frame(bank_df)

    evaluate_all_queries(bank_df, bank_p_sketch, n_cat=6, n_num=4, dataset_name="bank")
    
    print("-------------------------------------------------------------")
    # Build PachaSketch for Retail dataset
    print("\nRetail\n")
    retail_df = pd.read_parquet("../data/clean/online_retail.parquet")
    retail_p_sketch = build_pacha_sketch_for_retail(len(retail_df), delta, rel_eps, p, levels)
    # retail_df = retail_df.head(10)

    if multiprocessing:
        retail_p_sketch = retail_p_sketch.update_data_frame_multiprocessing(retail_df, workers=4)
    else:
        retail_p_sketch.update_data_frame(retail_df)

    evaluate_all_queries(retail_df, retail_p_sketch, n_cat=3, n_num=3, dataset_name="retail")

    print("-------------------------------------------------------------")
    # Build PachaSketch for Census dataset
    print("\nCensus\n")
    census_df = pd.read_parquet("../data/clean/acs_folktables.parquet")
    census_p_sketch = build_pacha_sketch_for_census(len(census_df), delta, rel_eps, p, levels)
    # census_df=census_df.head(10)

    if multiprocessing:
        census_p_sketch = census_p_sketch.update_data_frame_multiprocessing(census_df, workers=4)
    else:
        census_p_sketch.update_data_frame(census_df)  # Uncomment if you want to use single-threaded update
    
    evaluate_all_queries(census_df, census_p_sketch, n_cat=7, n_num=3, dataset_name="census")
    
    print("-------------------------------------------------------------")
    # Build PachaSketch for TPC-H dataset
    print("\nTPC-H\n")
    sf = 0.1
    df_path = f"../data/tpch/lineitem_{sf}.parquet"
    lineitem_df = pd.read_parquet(df_path)
    tpch_p_sketch = build_pacha_sketch_for_tpch(len(lineitem_df), delta, rel_eps, p, levels)
    # lineitem_df=lineitem_df.head(10)

    if multiprocessing:
        tpch_p_sketch = tpch_p_sketch.update_data_frame_multiprocessing(lineitem_df, workers=4)
    else:
        tpch_p_sketch.update_data_frame(lineitem_df)

    evaluate_all_queries(lineitem_df, tpch_p_sketch, n_cat=5, n_num=5, dataset_name="tpch")

    print(f"PachaSketches built successfully.")

if __name__ == "__main__":
    main()
