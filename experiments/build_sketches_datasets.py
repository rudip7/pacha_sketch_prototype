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
    cartesian_product, get_n_updates

from lib.ploting import set_style, plot_ylabel, plot_legend

from lib import baselines
reload(baselines)

from lib.baselines import CentralDPServer, LDPServer, LDPEncoderGRR, filter_df, query_df, \
      infer_domains_and_ranges, translate_query_region, evaluate_queries, check_accruracy, \
      evaluate_queries_baselines, evaluate_equivalent_pacha_sketches


## Section 1. Accuracy Experiments Different Datasets

def build_pacha_sketch_for_retail(delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=6) -> PachaSketch:
    """
    Builds a PachaSketch for the retail dataset with specified parameters.
    """
    # Retail
    cat_col_map_retail = [0, 1, 2]
    num_col_map_retail = [3, 4, 5]
    bases_retail = [2, 2, 2]

    n_cat_retail = len(cat_col_map_retail)
    n_num_retail = len(num_col_map_retail)
    cat_updates_retail, num_updates_retail, region_updates_retail = get_n_updates(n_cat_retail, n_num_retail, levels)

    retail_df = pd.read_parquet("../data/clean/online_retail.parquet")
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
        cat_index_parameters=BFParameters(n_values=len(retail_df) * cat_updates_retail, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len(retail_df) * num_updates_retail, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len(retail_df) * region_updates_retail, p=bloom_p))

    query_path = "../queries/retail/online_retail_random.json"
    with open(query_path, 'rb') as f:
        retail_queries_rand = orjson.loads(f.read())

    retail_p_sketch = retail_p_sketch.update_data_frame_multiprocessing(retail_df, workers=10)
    # retail_p_sketch.update_data_frame(retail_df)
    evaluate_queries(retail_df, retail_queries_rand["queries"][:200], retail_p_sketch, path_to_file=f"../results/datasets/retail_rand_p_{bloom_p}_eps_{rel_eps}.csv")

    return retail_p_sketch

def build_pacha_sketch_for_bank(delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=6) -> PachaSketch:
    """
    Builds a PachaSketch for the bank marketing dataset with specified parameters.
    """
    # Bank Marketing
    cat_col_map_bank = [0, 1, 2, 3, 4, 5]
    num_col_map_bank = [6, 7, 8, 9]
    bases_bank = [4, 5, 2, 2]

    n_cat_bank = len(cat_col_map_bank)
    n_num_bank = len(num_col_map_bank)
    cat_updates_bank, num_updates_bank, region_updates_bank = get_n_updates(n_cat_bank, n_num_bank, levels)

    bank_df = pd.read_parquet("../data/clean/bank_marketing.parquet")
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
        cat_index_parameters=BFParameters(n_values=len(bank_df) * cat_updates_bank, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len(bank_df) * num_updates_bank, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len(bank_df) * region_updates_bank, p=bloom_p))

    bank_df = pd.read_parquet("../data/clean/bank_marketing.parquet")
    query_path = "../queries/bank/bank_random.json"
    with open(query_path, 'rb') as f:
        bank_queries_rand = orjson.loads(f.read())

    bank_p_sketch = bank_p_sketch.update_data_frame_multiprocessing(bank_df, workers=10)
    bank_results_rand = evaluate_queries(bank_df, bank_queries_rand["queries"][:200], bank_p_sketch, path_to_file=f"../results/datasets/bank_rand_p_{bloom_p}_eps_{rel_eps}.csv")
    # bank_p_sketch.update_data_frame(bank_df)
    return bank_p_sketch

def build_pacha_sketch_for_census(delta=0.01, rel_eps=0.0005, bloom_p=0.01, levels=6) -> PachaSketch :
    """
    Builds a PachaSketch for the census dataset with specified parameters.
    """
    # Census
    cat_col_map_census = [0, 1, 2, 3, 4, 5, 6]
    num_col_map_census = [7, 8, 9]
    bases_census = [2, 5, 2]

    n_cat_census = len(cat_col_map_census)
    n_num_census = len(num_col_map_census)
    cat_updates_census, num_updates_census, region_updates_census = get_n_updates(n_cat_census, n_num_census, levels)

    census_df = pd.read_parquet("../data/clean/acs_folktables.parquet")
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
        cat_index_parameters=BFParameters(n_values=len(census_df) * cat_updates_census, p=bloom_p),
        num_index_parameters=BFParameters(n_values=len(census_df) * num_updates_census, p=bloom_p),
        region_index_parameters=BFParameters(n_values=len(census_df) * region_updates_census, p=bloom_p))

    census_df = pd.read_parquet("../data/clean/acs_folktables.parquet")
    query_path = "../queries/census/census_random.json"
    with open(query_path, 'rb') as f:
        census_queries_rand = orjson.loads(f.read())
    
    census_p_sketch = census_p_sketch.update_data_frame_multiprocessing(census_df, workers=5)
    census_results_rand = evaluate_queries(census_df, census_queries_rand["queries"][:200], census_p_sketch, path_to_file=f"../results/datasets/census_rand_p_{bloom_p}_eps_{rel_eps}.csv")
    # census_p_sketch.update_data_frame(census_df)
    return census_p_sketch
    
def main():
    """
    Main function to build PachaSketches for different datasets.
    """
    # Parameters
    delta = 0.01
    levels = 6
    # rel_eps = 0.0005
    # bloom_p = 0.01

    rel_eps = 0.0005
    p = 0.01
    print(f"Building PachaSketches with p: {p}, rel_eps: {rel_eps}") 
    
    # Build PachaSketch for Bank Marketing dataset
    print("Bank")
    bank_p_sketch = build_pacha_sketch_for_bank(delta, rel_eps, p, levels)

    # Build PachaSketch for Retail dataset
    print("Retail")
    retail_p_sketch = build_pacha_sketch_for_retail(delta, rel_eps, p, levels)

    # Build PachaSketch for Census dataset
    print("Census")
    census_p_sketch = build_pacha_sketch_for_census(delta, rel_eps, p, levels)
    print(f"PachaSketches built successfully.")

    bank_p_sketch.save_to_file(f"../sketches/bank/bank_p{p}_eps{rel_eps}.json")
    retail_p_sketch.save_to_file(f"../sketches/retail/retail_p{p}_eps{rel_eps}.json")
    census_p_sketch.save_to_file(f"../sketches/census/census_p{p}_eps{rel_eps}.json")


if __name__ == "__main__":
    main()
