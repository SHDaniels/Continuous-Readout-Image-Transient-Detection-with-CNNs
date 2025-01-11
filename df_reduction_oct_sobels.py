#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
df reduction
"""


# In[13]:


import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import sobel


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


def get_sobel_ratio(stamp):
    """
    input: stamp: 2d brightnesses array with candidate transient
    larger = less likely to be a cosmic ray
    returns sobel_ratio = np.max(Lx)/np.max(Ly) where x is columns axis and y is rows axis
    """
    Lx = sobel(stamp, axis=1)  # column axis
    Ly = sobel(stamp, axis=0)  # rows axis
    sobel_ratio = np.max(Lx) / np.max(Ly)  # we want this to be as large as possible
    return sobel_ratio


def add_sobel_ratio(df, imagenobg):
    sobels = np.zeros(len(df))
    for index in df.index.to_numpy():
        row = int(df.loc[index]["row_from_nn"]) + 10
        col = int(df.loc[index]["col_from_nn"]) + 10
        stamp = imagenobg[row - 10 : row + 10, col - 10 : col + 10]
        sobel = get_sobel_ratio(stamp)
        sobels[index] = sobel

    #     if np.max(littlerstamp)<thr:
    #         drop_indices.append(index)
    # newdf = df.drop(drop_indices)
    df["sobel_ratio"] = sobels
    return df


print("defining paths", flush=True)

# Set the folder path where your DataFrames are stored
folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020"
direcname = "1020"
fpath = (
    "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"
    + direcname
    + "_minus_extras/"
)

print("listing files", flush=True)
# List all the files in the folder
files = [
    f for f in os.listdir(folder_path) if f.endswith("reduced")
]  # Adjust this to your file format
# print('Files found:', files)

# Read and process each DataFrame file
for f in files:
    dpath = os.path.join(folder_path, f + "_sobelratio")

    print("file i found: ", f)
    # Skip if the reduced file already exists
    if os.path.isfile(dpath):
        print("i should skip this one: ", f)
        continue

    print("working on now: ", f)
    df = pd.read_csv(os.path.join(folder_path, f))

    print("why is this happening: ", f, flush=True)

    # these will stay the same
    # fname = df.iloc[0]["filename"]+"fits.fz"
    fname = f[:47]
    quad = f[47]
    path = fpath + fname
    hdul = fits.open(path)

    file1_array = hdul[quad].data
    background = rff.get_image_background(file1_array)
    imagenobg = file1_array - background

    # Apply the reduce_max function for further reduction

    newdf = add_sobel_ratio(df, imagenobg)

    # Save the reduced DataFrame
    # break
    print("saving i guess??: ", f)
    newdf.to_csv(dpath)
