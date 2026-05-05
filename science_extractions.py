#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""


old nb goal: the goal of this notebook is to run cnn and get efficiencies
and such from the folder of full_image_implants that i 
just made.

"""


# In[1]:


import numpy as np
from tensorflow import keras
from astropy.io import fits

import os

import get_postage_stamps as gpsf

import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the
import review_findings_functions as rff

import pandas as pd

import sys

name = sys.argv[1]

cnn_index = int(sys.argv[2])  # 0-4 from SLURM_ARRAY_TASK_ID

suffix = f"_{cnn_index}"

#they need to look like they do in the data direc /1020
lcfilenames = pd.read_csv('/home/3155/rwe/filenames_lcs.txt', sep=" ", header=None)
lcfilenames.columns = ["filename"]
filelist = []
for filename in lcfilenames['filename']:
    file = filename[11:50]+'.fits.fz'
    filelist.append(file)
filelist = np.unique(filelist)

# to_append = np.array(["_0", "_1","_2","_3","_4"])
# In[68]:

path_start = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_"
# paths = [path_start+name+a + "/" for a in to_append]
# # path = os.path.join(parent_d, d)
# for path in paths:
#     os.mkdir(path)


save_direc = path_start + name + suffix + "/"
# os.mkdir(, exist_ok=True)

if not os.path.exists(save_direc):
    os.mkdir(save_direc)

data_direc = "/lustre/lrspec/ZTF_data/2022/1020/"
    
path_save_long_term = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/" #cause i will delete the straight up output of this one


# for i, suffix in enumerate(to_append):
if 1==1: #yknow indents
    model = keras.models.load_model(f"/home/3155/CNN_{name}{suffix}")

    # save_direc = paths[i]
    
    wid = 10
    fpath_extractions_save = save_direc


    # for i, file in enumerate(os.listdir(data_direc)): 
    for file in filelist:
        image_filename = file
        # image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        if image_filename[-9:] =="x.fits.fz": #beginning crm
            continue
        # real_image_filename = filename_start + "fits.fz"
        image_filename_full = data_direc + image_filename
        if os.path.isfile( #if it already exists
            fpath_extractions_save
            + str(filename_start)
            + "_extraction_predsarray_"
            + str(4)
            + "_predsarray.npy"
        ):
            print("skipping ", filename_start)
            continue
        if os.path.isfile( #if it already exists
            path_save_long_term
            + str(filename_start)
            + "_extraction_predsarray_"
            + str(4)
            + "_predsarray.npy"
        ):
            print("skipping - exists in average", filename_start)
            continue        
        if os.path.isfile("/lustre/lrspec/ZTF_data/training_aug24/implant_train/"+image_filename): #checking it isn't in training data:
            print("skipping - in training data ", filename_start)
            continue
        

        print(image_filename_full)
        for iteration in range(1, 5):
            with fits.open(image_filename_full) as hdul:
                file1_array = hdul[iteration].data
                norm = (file1_array - np.min(file1_array)) / (
                    np.max(file1_array) - np.min(file1_array)
                )
                image = norm
                nrow, ncol = image.shape
                path = (
                    fpath_extractions_save
                    + str(filename_start)
                    + "_extraction_predsarray_"
                    + str(iteration)
                )
                rff.save_predsarrays(image, model, wid, path)
        del norm
        del file1_array
        del image
        # del hdul #may not need this when we have with as
        gc.collect()

