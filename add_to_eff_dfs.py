'''
add some values to every dataframe in our effiiciency dataframe set from the various NNS
'''

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt



import precision_recall as pr


# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os

import sys

# df_names = ["original", "flipped", "both", "dup"]
df_names = [sys.argv[1]]

df_neg_base_path = "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/"
df_pos_base_path = "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/"

# df_paths = ["predsarrays/",
#                 "presdarrays_flipped/",
#                 "presdarrays_both/",
#                 "predsarrays_dup/"]
df_paths = [ "predsarrays_"+sys.argv[1]+"/"]

for i, name in enumerate(df_names):
    df = pd.read_csv(f"{df_neg_base_path+df_paths[i]}"+"df_concat")
    dup = rff.find_dups(df)
    matched = rff.match_implantations_eff(dup, "neg")
    matched.to_csv(f"{df_neg_base_path+df_paths[i]}"+"df_matched")
    
    df = pd.read_csv(f"{df_pos_base_path+df_paths[i]}"+"df_concat")
    dup = rff.find_dups(df)
    matched = rff.match_implantations_eff(dup, "pos")
    matched.to_csv(f"{df_pos_base_path+df_paths[i]}"+"df_matched")    