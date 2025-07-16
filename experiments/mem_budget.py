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

def build_pacha_sketch_for_tpch(lineitem_df: pd.DataFrame, mem_budget: float, n_cubes: int = 50_000) -> PachaSketch:
    levels = 5

    cat_col_map_tpch = [0, 1, 2, 3, 4]
    n_cat_tpch = len(cat_col_map_tpch)
    num_col_map_tpch = [5, 6, 7, 8, 9]
    n_num_tpch = len(num_col_map_tpch)
    bases_tpch = [5,5,5,10,2]

    ad_tree_levels = len(cat_col_map_tpch)
    num_combinations = 13
    cat_updates_tpch, num_updates_tpch, region_updates_tpch = get_n_updates_customized(ad_tree_levels, num_combinations, levels)
    
    rel_eps = 0.0005
    delta = 0.01
    error_eps = rel_eps / region_updates_tpch

    mem_cms = mem_budget * 0.75 / levels
    n_counters = mem_cms * 1024 * 1024 / 4
    depth = 3
    width = n_counters // depth
    error_eps = np.e/width
    while error_eps < rel_eps / region_updates_tpch and depth < 5:
        depth += 1
        width = n_counters // depth
        error_eps = np.e/width
    
    if error_eps < rel_eps / region_updates_tpch:
        error_eps = rel_eps / region_updates_tpch
        cms_params = CMParameters(delta=delta, error_eps=error_eps)
    else:
        cms_params = CMParameters(width=int(width), depth=int(depth))

    mem_cms = cms_params.peek_memory() * levels
   
    print(f"Memory for CMSs: {mem_cms} MB")
    mem_index = mem_budget - mem_cms
    p_num_index = num_updates_tpch / (num_updates_tpch + region_updates_tpch)
    p_region_index = 1.0 - p_num_index

    mem_num_index = mem_index * p_num_index
    print(f"Memory for num index: {mem_num_index} MB")
    mem_region_index = mem_index * p_region_index
    print(f"Memory for region index: {mem_region_index} MB")

    n_bits_num_index = int(np.ceil(mem_num_index * 8 * 1024 * 1024))
    n_bits_region_index = int(np.ceil(mem_region_index * 8 * 1024 * 1024))
    k = 3
    num_index_params = BFParameters(size=n_bits_num_index, hash_count=k)
    region_index_params = BFParameters(size=n_bits_region_index, hash_count=k)

    tpch_ad_tree = ADTree.from_json("../sketches/ad_trees/tpch_lineitem.json")
    
    tpch_p_sketch = PachaSketch.build_with_uniform_size(
        levels=levels,
        num_dimensions=n_cat_tpch + n_num_tpch,
        cat_col_map=cat_col_map_tpch,
        num_col_map=num_col_map_tpch,
        bases=bases_tpch,
        ad_tree=tpch_ad_tree,
        cm_params=cms_params,
        cat_index_parameters=num_index_params,
        num_index_parameters=num_index_params,
        region_index_parameters= region_index_params
        )
    
    tpch_p_sketch.max_n_cubes = n_cubes

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


    
def main(i: int):
    mem_budgets = 2**np.arange(6) * 64
    workers = np.asarray([10, 8, 6, 4, 2, 1])

    max_n_cubes = [  1_000,  25_000,  50_000,  50_000, 50_000, 50_000]

    if i < 0 or i >= len(mem_budgets):
        raise ValueError(f"Invalid index {i}. Must be between 0 and {len(mem_budgets) - 1}.")
    
    mem_budget = mem_budgets[i]
    n_workers = workers[i]
    n_cubes = max_n_cubes[i]
    
    query_path = "../queries/tpch/all_tpch_random.json"
    with open(query_path, 'rb') as f:
        tpch_queries_rand = orjson.loads(f.read())

    df_path = f"../data/tpch/lineitem_8.parquet"
    lineitem_df = pd.read_parquet(df_path).head(24_000_000)

    chunks = np.array_split(lineitem_df, 240)

    print(f"\nMemory budget: {mem_budget / 1024 / 1024} MB\n")
    tpch_p_sketch = build_pacha_sketch_for_tpch(lineitem_df, mem_budget, n_cubes)
    
    for j, chunk in enumerate(chunks):
        print(f"Processing chunk {j + 1}/{len(chunks)}...")
#        tpch_p_sketch = tpch_p_sketch.update_data_frame(chunk)
        tpch_p_sketch = tpch_p_sketch.update_data_frame_multiprocessing(chunk, workers=n_workers)
        if j == 0:
            temp_df = chunk
        else:
            temp_df = pd.concat([temp_df, chunk], ignore_index=True)
        results = evaluate_queries(temp_df, tpch_queries_rand['queries'], tpch_p_sketch, path_to_file=f"../results/tpch/fix_size/tpch_fix_{mem_budget}_MB_{j}.csv")
        med = results['normalized_error'].median()        
        if med < 0.0 or med > 1.0:
            break
    print(f"Finally done!")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mem_budget.py <index>")
        sys.exit(1)
    
    try:
        index = int(sys.argv[1])
    except ValueError:
        print("Index must be an integer.")
        sys.exit(1)

    main(index)
