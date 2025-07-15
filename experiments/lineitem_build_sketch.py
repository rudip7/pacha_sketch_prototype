import sys
# append the path of the parent directory
sys.path.append("..")

import numpy as np
np.set_printoptions(legacy='1.25')

import pandas as pd
from pympler import asizeof

from importlib import reload
import orjson

from lib import pacha_sketch
reload(pacha_sketch)
from lib.pacha_sketch import PachaSketch, ADTree, BFParameters, CMParameters, \
     MaterializedCombinations, get_n_updates_customized


from lib import baselines
reload(baselines)
from lib.baselines import evaluate_queries

def build_pacha_sketch_for_tpch(lineitem_df: pd.DataFrame, rel_eps, bloom_p):
    levels = 5
    delta = 0.01

    cat_col_map_tpch = [0, 1, 2, 3, 4]
    n_cat_tpch = len(cat_col_map_tpch)
    num_col_map_tpch = [5, 6, 7, 8, 9]
    n_num_tpch = len(num_col_map_tpch)
    bases_tpch = [5,5,5,10,2]

    ad_tree_levels = len(cat_col_map_tpch)
    num_combinations = 13
    cat_updates_tpch, num_updates_tpch, region_updates_tpch = get_n_updates_customized(ad_tree_levels, num_combinations, levels)

    len_df = len(lineitem_df)
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
    
    # workers = (240 * 1024) // total_size
    # if total_size > 100_000:
    #     print(f"Total size of tpch_p_sketch is too large, proceeding with single-threaded update.")
    #     tpch_p_sketch.update_data_frame(lineitem_df)
    # else:
    # workers = 2
    # print(f"Proceeding with {workers} workers for updating tpch_p_sketch.")
    tpch_p_sketch.update_data_frame(lineitem_df)
    # tpch_p_sketch = tpch_p_sketch.update_data_frame_multiprocessing(lineitem_df, workers=workers)
    tpch_p_sketch.save_to_file(f"../sketches/tpch/tpch_0.1_eps_{rel_eps}_p_{bloom_p}.json")

    return tpch_p_sketch


    
def main():
    # scale_factors = [0.1]
    # scale_factors = [0.5]
    sf = 0.1
    probs = 2**np.arange(7)*0.005
    rel_epsilons = 2**np.arange(7) * 0.0025

    query_path = "../queries/tpch/tpch_random.json"
    with open(query_path, 'rb') as f:
        tpch_queries_rand = orjson.loads(f.read())

    query_path = "../queries/tpch/tpch_random_2.json"
    with open(query_path, 'rb') as f:
        tpch_queries_rand_2 = orjson.loads(f.read())
    
    df_path = f"../data/tpch/lineitem_{sf}.parquet"
    lineitem_df = pd.read_parquet(df_path)
    for p, rel_eps in zip(probs, rel_epsilons):
        print(f"Building PachaSketches with p: {p}, rel_eps: {rel_eps}")
        tpch_p_sketch = build_pacha_sketch_for_tpch(lineitem_df, rel_eps=rel_eps, bloom_p=p)
        print(f"Evaluating queries for {df_path}...")
        results = evaluate_queries(lineitem_df, tpch_queries_rand['queries'], tpch_p_sketch, path_to_file=f"../results/tpch/tpch_random_results_0.1_eps_{rel_eps}_p_{p}.csv")
        results_2 = evaluate_queries(lineitem_df, tpch_queries_rand_2['queries'], tpch_p_sketch, path_to_file=f"../results/tpch/tpch_random_results_2_0.1_eps_{rel_eps}_p_{p}.csv")
        print(f"Results for scale factor {sf} saved.")

if __name__ == "__main__":
    main()
