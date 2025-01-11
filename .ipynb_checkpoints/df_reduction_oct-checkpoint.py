#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
I shall inspect the dataframes that have come - perhaps
specifically from 1020 - 
and make cuts see if anything is found
'''


# In[13]:


import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt


# In[4]:


import plot_time_series as pts
import precision_recall as pr
import importlib
#importlib.reload(plot_time_series)
import gc #garbage collector
import glob #i don't know if there is a better way to do this but this was the 
#internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os


# In[26]:


plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


def reduce_max_3(df, imagenobg, thr=300):
    drop_indices = []
    for index in df.index.to_numpy():
        row = int(df.loc[index]["row_from_nn"])+10
        col = int(df.loc[index]["col_from_nn"])+10
        littlerstamp = imagenobg[row-4:row+4, col-4:col+4]
        

        if np.max(littlerstamp)<thr:
            drop_indices.append(index)
    newdf = df.drop(drop_indices)
    return newdf

print("defining paths", flush=True)

# Set the folder path where your DataFrames are stored
folder_path = '/lustre/lrspec/ZTF_data/science_extractions/1020'
direcname = '1020'
fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"

print('listing files', flush=True)
# List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('alllist')]  # Adjust this to your file format
# print('Files found:', files)

# Read and process each DataFrame file
for f in files:
    
    dpath = os.path.join(folder_path, f + '_reduced')
    
    print('file i found: ', f)
    # Skip if the reduced file already exists
    if os.path.isfile(dpath):
        print('i should skip this one: ', f)
        continue
        
    print('working on now: ', f)
    df = pd.read_csv(os.path.join(folder_path, f))
    
    # Reduce based on type != 'cr'
    df_nocr = df[df['type'] != 'cr'].copy()
    
    # Apply other reductions
    df_reduced_edges = rff.reduce_edges(df_nocr, 15)
    df_reduced_edges_size4 = rff.reduce_size(df_reduced_edges, 4)

    print('why is this happening: ', f, flush=True)
    # Add the quadrant as a new column
    quad= f[-8]
    df_reduced_edges_size4.loc[:, 'quadrant'] = np.full( len(df_reduced_edges_size4), quad)
    
    
    #these will stay the same
    # fname = df.iloc[0]["filename"]+"fits.fz"
    fname = f[:47]

    path = fpath+fname
    hdul = fits.open(path)
    file1_array = hdul[quad].data
    background = rff.get_image_background(file1_array)
    imagenobg = file1_array - background
    
    # Apply the reduce_max function for further reduction

    newdf = reduce_max_3(df_reduced_edges_size4, imagenobg, thr=300)

    # Save the reduced DataFrame
    # break
    print('saving i guess??: ', f)
    newdf.to_csv(dpath)



