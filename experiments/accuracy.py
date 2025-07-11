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
from lib.pacha_sketch import PachaSketch, ADTree, BFParameters, CMParameters, cartesian_product

from lib.ploting import set_style, plot_ylabel, plot_legend

from lib import baselines
reload(baselines)

from lib.baselines import CentralDPServer, LDPServer, LDPEncoderGRR, filter_df, query_df, \
      infer_domains_and_ranges, translate_query_region, evaluate_queries, check_accruracy, \
      evaluate_queries_baselines, evaluate_equivalent_pacha_sketches


delta = 0.01
abs_error_eps = 1.0
bloom_p = 0.01
## Retail
cat_updates = 4
num_updates = 6
retail_df = pd.read_parquet("../data/clean/online_retail_no_outliers.parquet")
query_path = "../queries/online_retail_2_cols.json"
with open(query_path, 'rb') as f:
    retail_queries_2 = orjson.loads(f.read())

query_path = "../queries/online_retail_4_cols.json"
with open(query_path, 'rb') as f:
    retail_queries_4 = orjson.loads(f.read())

retail_ad_tree = ADTree.from_json("../sketches/ad_trees/online_retail.json")

retail_p_sketch = PachaSketch.build_with_uniform_size(
    levels=6,
    num_dimensions=6,
    cat_col_map=[0,1,2],
    num_col_map=[3,4,5],
    bases=[2,2,2],
    ad_tree=retail_ad_tree,
    cm_params=CMParameters(delta=delta, error_eps=abs_error_eps / len(retail_df)),
    cat_index_parameters=BFParameters(n_values=len(retail_df)*cat_updates, p=bloom_p),
    num_index_parameters=BFParameters(n_values=len(retail_df)*num_updates, p=bloom_p),
    region_index_parameters= BFParameters(n_values=len(retail_df)*(cat_updates+num_updates), p=bloom_p))
retail_p_sketch.get_size()

retail_p_sketch.update_data_frame(retail_df)
# retail_results_2 = evaluate_queries(retail_df, retail_queries_2["queries"], retail_p_sketch, path_to_file="../results/accuracy/retail_2_p_sketch.csv")
# retail_results_4 = evaluate_queries(retail_df, retail_queries_4["queries"], retail_p_sketch, path_to_file="../results/accuracy/retail_4_p_sketch.csv")