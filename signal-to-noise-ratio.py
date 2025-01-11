#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
hopefully we will find the signal to noise ratio here
"""


# In[1]:


import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.ndimage as snm


# In[2]:


import plot_time_series as pts
import precision_recall as pr
import importlib

# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os
import implantation_functions as imp
from scipy.signal import find_peaks


# In[3]:


plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


# In[15]:


def add_snr(df, direcname, half_aperture, tracking):
    """
    returns the dataframe with a new column of stdv ratios
    see get_sobel_ratio(stamp) and get_stdv_ratio
    """
    ratios_center = np.zeros(len(df))
    ratios_max = np.zeros(len(df))
    ratios_min = np.zeros(len(df))
    for number, index in enumerate(df.index):
        # print(index)
        tot, maxcols = rff.get_tot(df, index, direcname, half_aperture, tracking)
        stdv = rff.get_stdvs(tot, do_all=False, df=df, index=index)
        avg = np.average(tot)
        row = int(df.loc[index]["row_from_nn"]) + 10
        max_val = np.max(tot[row - 10 : row + 10])
        min_val = np.min(tot[row - 10 : row + 10])
        signal_center = tot[row] - avg
        signal_max = max_val - avg
        signal_min = min_val - avg
        ratios_center[number] = signal_center / stdv
        ratios_max[number] = signal_max / stdv
        ratios_min[number] = signal_min / stdv
    df["snr_center"] = ratios_center
    df["snr_max"] = ratios_max
    df["snr_min"] = ratios_min
    return df


# In[ ]:


# okay so now we should have a bunch of reduced dataframes

# Set the folder path where your DataFrames are stored
folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020"
direcname = "1020"
fpath = (
    "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"
    + direcname
    + "_minus_extras/"
)


# List all the files in the folder
files = [
    f for f in os.listdir(folder_path) if f.endswith("sobelratio")
]  # Adjust this to your file format


# Step 3: Read all DataFrames and add the filename as a new column
dataframes = []
for number, f in enumerate(files):
    dpath = os.path.join(folder_path, f + "_snr")

    # Skip if the reduced file already exists
    if os.path.isfile(dpath) and number > 6:
        print("i should skip this one: ", f)
        continue

    print("working on now: ", f)

    df = pd.read_csv(os.path.join(folder_path, f))
    nocrs = df[df["sobel_ratio"] > 5]
    df_new = add_snr(nocrs, direcname, half_aperture=2, tracking=True)
    df_new.to_csv(dpath)
    # dataframes.append(df)
