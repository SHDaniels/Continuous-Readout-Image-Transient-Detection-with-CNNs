#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
time to go from the implanted images to the lightcurve parameters all in one go
"""


# In[4]:


import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.ndimage as snm


import precision_recall as pr
import importlib

# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os
import implantation_functions as imp
from tensorflow import keras
import get_postage_stamps as gpsf
import fitting


# seed = np.random.seed(43)
# size = 2000
# center_values = np.random.randint(50, high=3000, size=size, dtype=int)
# sigmas = np.random.randint(3, high=10, size=size, dtype=int)
# intensities_neg = imp.get_intensities_neg(size, seed)


# # In[14]:


# fpath_image = "/lustre/lrspec/ZTF_data/training_aug24/implant_train/"
# path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/"
# size = 2000  # will cut to 1000


# counter = 0

# for iteration in range(1,4): #only gets through 1 though
#     for i, file in enumerate(os.listdir(fpath_image)):
#         image_filename = os.fsdecode(file)
#         filename_start = image_filename[:40]
#         image_filename_full = fpath_image+image_filename
#         print(image_filename)
#         hdul=fits.open(image_filename_full)
#         print(iteration)
#         image = hdul[iteration].data
#         # print(file1_array)

#         nrow, ncol = image.shape
#         imagepath = path_to_save+"images/"+str(filename_start)+"_"+str(iteration)+"_implanted_neg"
#         infopath = path_to_save+"info/"+str(filename_start)+"_"+str(iteration)+"_implanted_info_neg"
#         # stamps_implanted, stamp_info  = imp.get_10_trans_stamps(image, sigmas,
#                                                             # intensities_pos,
#                                                             # center_values_new, counter, plot=False)
#         implanted, transient_info = imp.implant_10_transients(image,
#                                                       sigmas, intensities_neg,
#                                                       center_values, counter=counter, norm=False,
#                                                       plot=False)

#         np.save(imagepath, implanted)
#         np.save(infopath, transient_info)
#         counter+=(len(transient_info)+1)
#         if counter>1990:
#             break
            
            
seed = np.random.seed(42)
size = 2000
center_values = np.random.randint(30, high=3042, size=size, dtype=int)
sigmas = np.random.randint(3, high=10, size=size, dtype=int)
intensities_pos = imp.get_intensities_pos(size, seed)

fpath_image = "/lustre/lrspec/ZTF_data/training_aug24/implant_train/"
path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/"
size = 2000  # will cut to 1000
transient_info_list = []
optimized_params_list = []
saved_tots = []

counter = 0

for iteration in range(1, 4):
    for i, file in enumerate(os.listdir(fpath_image)):
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        image_filename_full = fpath_image + image_filename
        imagepath = (
            path_to_save
            + "images/"
            + str(filename_start)
            + "_"
            + str(iteration)
            + "_implanted"
        )
        infopath = (
            path_to_save
            + "info/"
            + str(filename_start)
            + "_"
            + str(iteration)
            + "_implanted_info"
        )

        #         if os.path.isfile(infopath):
        #             print("skipping ",filename_start)
        #             continue

        print(image_filename)

        hdul = fits.open(image_filename_full)
        print(iteration)
        image = hdul[iteration].data
        # print(file1_array)

        nrow, ncol = image.shape
        # stamps_implanted, stamp_info  = imp.get_10_trans_stamps(image, sigmas,
        # intensities_pos,
        # center_values_new, counter, plot=False)
        implanted, transient_info = imp.implant_10_transients(
            image, sigmas, intensities_pos, center_values, counter=counter, norm=False, plot=False
        )

        np.save(imagepath, implanted)
        np.save(infopath, transient_info)
        counter += len(transient_info) + 1
        if counter > 1990:
            break