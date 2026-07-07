#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
I shall inspect the dataframes that have come - perhaps
specifically from 1020 - 
and make cuts see if anything is found
"""


# In[13]:


import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt


# In[4]:


import plot_time_series as pts
import precision_recall as pr
import importlib

# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os


# In[26]:


plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


# In[10]:


# Step 1: Set the folder path where your DataFrames are stored
folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020"

# Step 2: List all the files in the folder
files = [
    f for f in os.listdir(folder_path) if f.endswith("alllist")
]  # Adjust this to your file format

# Step 3: Read all DataFrames and add the filename as a new column
dataframes = []
for f in files:
    df = pd.read_csv(os.path.join(folder_path, f))
    df["quadrant"] = f[-8]  # Add the filename as a new column
    dataframes.append(df)

# Step 4: Concatenate all DataFrames into one
concatenated_df = pd.concat(dataframes, ignore_index=True)


wid = 10


# In[51]:


def inspect_df_with_quadrant(df, direcname):
    wid = 10
    figcols = 5
    figrows = 1 + len(df) // figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols * 1.8, figrows * 1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols * i + j
            if index < len(df):
                fname = df.iloc[index]["filename"] + "fits.fz"
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                quad = int(df.iloc[index]["quadrant"])
                fpath = (
                    "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"
                    + direcname
                    + "_minus_extras/"
                )
                path = fpath + fname
                hdul = fits.open(path)
                file1_array = hdul[quad].data
                norm = (file1_array - np.min(file1_array)) / (
                    np.max(file1_array) - np.min(file1_array)
                )
                # ax[i, j].imshow(norm[col+wid-10:col+wid+10, row-10+wid:row+10+wid])
                ax[i, j].imshow(
                    np.sqrt(
                        norm[
                            row - 10 + wid : row + 10 + wid,
                            col + wid - 10 : col + wid + 10,
                        ]
                    )
                )
                ax[i, j].set_axis_off()
                # ax[i, j].set_title(fname[26:-8])
                ax[i, j].set_title(str(index))
    plt.tight_layout()
    plt.show()


concat_nocr = concatenated_df[concatenated_df["type"] != "cr"]
print("original ", len(concatenated_df))
print("no cr ", len(concat_nocr))

concat_reduced_edges = rff.reduce_edges(concat_nocr, 15)


# In[46]:


concat_reduced_edges_size4 = rff.reduce_size(concat_reduced_edges, 4)


# In[109]:


def reduce_max(df, direcname, thr=300):
    drop_indices = []
    for index in range(len(df)):
        stamp, bg = rff.get_stamp_from_df(index, direcname, df, norm=False)
        littlerstamp = stamp[5:15, 5:15]
        # plt.imshow(littlerstamp)
        # plt.colorbar()
        if np.max(littlerstamp - bg) < thr:
            drop_indices.append(index)
    newdf = df.drop(drop_indices)
    return newdf


# In[110]:


newdf = reduce_max(concat_reduced_edges_size4, "1020")
print("final ", len(newdf))


# In[113]:

newdf.to_csv("df_reduce_max_1020")
