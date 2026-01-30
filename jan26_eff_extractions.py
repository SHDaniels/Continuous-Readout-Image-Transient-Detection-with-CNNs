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

# internet's suggestion for searching with an incomplete path name


# In[12]:




# In[ ]:


# some stuff that will stay the same
wid = 10
# fpath_image = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/images/"
# fpath_info = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/info/"
# fpath_extractions = "/lustre/lrspec/ZTF_data/2022_1014_copy/full_transient_extractions/"

# image_directory = os.fsencode(fpath_image)


# In[6]:


model_best = keras.models.load_model("/home/3155/CNN_jan26")


# In[68]:




# In[91]:

# '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/',
#

data_direc = [
    "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/images/", "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/images/"
    ]

save_direc = [
        "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/presdarrays_flipped/", "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/presdarrays_flipped/"
]


for j in range(len(data_direc)):
    data_direc = data_direc[j]
    save_direc = save_direc[j]
    model = model_best
    wid = 10
    abst_thr = 0.9999
    fpath_extractions_save = save_direc
    

    for i, file in enumerate(os.listdir(data_direc)): 
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        real_image_filename = filename_start + "fits.fz"
        image_filename_full = data_direc + real_image_filename
        if os.path.isfile( #if it already exists
            fpath_extractions_save
            + str(filename_start)
            + "_extraction_predsarray_"
            + str(4)
            + "_predsarray.npy"
        ):
            print("skipping ", filename_start)
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

            rff.save_predsarrays(model, wid, path)
            del norm
            del file1_array
            del image
            # del hdul #may not need this when we have with as
            gc.collect()

