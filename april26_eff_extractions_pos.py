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

from tqdm.auto import tqdm
import os

import get_postage_stamps as gpsf

import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the
import review_findings_functions as rff

# internet's suggestion for searching with an incomplete path name


# In[12]:

import sys

name = sys.argv[1]



# In[ ]:


# some stuff that will stay the same
wid = 10
# fpath_image = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/images/"
# fpath_info = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/info/"
# fpath_extractions = "/lustre/lrspec/ZTF_data/2022_1014_copy/full_transient_extractions/"

# image_directory = os.fsencode(fpath_image)


# In[6]:


# model_best = keras.models.load_model("/home/3155/CNN_"+name)

to_append = np.array(["_0", "_1","_2","_3","_4"])
# In[68]:

path_start = "/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/predsarrays_"
paths = [path_start+a for a in to_append]
# path = os.path.join(parent_d, d)
for path in paths:
    os.mkdir(path)

# In[91]:

# '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/',
#

data_direc = "/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/images/"
    

# save_direcs = [
#         "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/presdarrays_both/", "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/presdarrays_both/"
# ]








for i, suffix in enumerate(to_append):
    model = keras.models.load_model("/home/3155/CNN_"+name+suffix)

    save_direc = paths[i]
    
    wid = 10
    abst_thr = 0.9999
    fpath_extractions_save = save_direc


    for i, file in enumerate(os.listdir(data_direc)): 
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:42]
        # real_image_filename = filename_start + "fits.fz"
        image_filename_full = data_direc + image_filename
        if os.path.isfile( #if it already exists
            fpath_extractions_save
            + str(filename_start)
            + "_predsarray.npy"
        ):
            print("skipping ", filename_start)
            continue

        print(image_filename_full)
        file1_array = np.load(image_filename_full)
        norm = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )
        image = norm
        path = (
            fpath_extractions_save
            + str(filename_start)
        )

        rff.save_predsarrays(image, model, wid, path)
        del norm
        del file1_array
        del image
        # del hdul #may not need this when we have with as
        gc.collect()

