#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
gonna check out all the prediction images

"""


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from astropy.io import fits
import scipy.ndimage as snm


# In[2]:


from tqdm.auto import tqdm
import os
from pylab import cm
import matplotlib as mpl
import get_postage_stamps as gpsf
import pandas as pd


# In[3]:


import plot_time_series as pts
import precision_recall as pr
import importlib

# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name


# In[4]:


# some matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


# In[4]:


# image_filename_full = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/ztf_20210622237812_000677_zr_c12_o_0054.fits.fz"
# hdul=fits.open(image_filename_full)
# file1_array = hdul[1].data
# norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array))
# plt.imshow(norm)


# In[5]:


# some stuff that will stay the same
wid = 10
# fpath_image = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/images/"
# fpath_info = "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/info/"
# fpath_extractions = "/lustre/lrspec/ZTF_data/2022_1014_copy/full_transient_extractions/"

# image_directory = os.fsencode(fpath_image)


# In[6]:


# model_best=keras.models.load_model("/home/3155/CNN_jan2")


# In[5]:


direcs = ["/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/"]


#           '/lustre/lrspec/ZTF_data/science_extractions/to_extract/1026_minus_extras/',
#           '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20220808_minus_extras/',
#          '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/',
# '/lustre/lrspec/ZTF_data/science_extractions/to_extract/0820_minus_extras/',
#          '/lustre/lrspec/ZTF_data/science_extractions/to_extract/1014_minus_extras/',
#          '/lustre/lrspec/ZTF_data/science_extractions/to_extract/1017_minus_extras/']

direcnames = ["1020"]
# , '1026,', '20220808', '20210622', '0820_real_this_time', '1014', '1017']


# In[7]:


predsarray = np.load(
    "/lustre/lrspec/ZTF_data/science_extractions/1020/ztf_20221020297847_000449_zr_c16_x_0022._extraction_predsarray_4_predsarray.npy"
)


# In[10]:


predsarray.shape


# In[12]:


testimp = np.load(
    "/lustre/lrspec/ZTF_data/2022_1014_copy/implantations/images/ztf_20221014279051_000501_zr_c16_x_0025._implanted10.npy"
)


# In[16]:


nrow, ncol = testimp.shape
wid = 10


# In[14]:


nrow, ncol


# In[17]:


pimg_pos = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
pimg_bg = predsarray[:, 0].reshape(ncol - 2 * wid, nrow - 2 * wid)
pimg_streak = predsarray[:, 4].reshape(ncol - 2 * wid, nrow - 2 * wid)
pimg_neg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
pimg_cr = predsarray[:, 3].reshape(ncol - 2 * wid, nrow - 2 * wid)


# In[19]:


plt.imshow(pimg_pos > 0.9999)


# In[21]:


np.sum(pimg_pos > 0.9999)


# In[31]:


def make_obj_list(image, predsarray, abst_thr, fname):
    nrow, ncol = image.shape
    pimg_pos = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_bg = predsarray[:, 0].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_streak = predsarray[:, 4].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_neg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_cr = predsarray[:, 3].reshape(ncol - 2 * wid, nrow - 2 * wid)

    pimg_both = pimg_pos + pimg_neg

    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # for what counts as one object

    # FOR POS
    thrdet = pimg_pos > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_pos = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_pos = np.array(loc_list_pos)
    objsizes_pos = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_pos), fname)
    poses = np.full(len(loc_list_pos), "pos")
    if len(loc_list_pos) > 0:
        poslist = (
            np.vstack(
                (fnames, poses, loc_list_pos[:, 0], loc_list_pos[:, 1], objsizes_pos)
            )
        ).T
    else:
        poslist = np.zeros((1, 5))

    # FOR NEG
    thrdet = pimg_neg > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_neg = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_neg = np.array(loc_list_neg)

    objsizes_neg = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_neg), fname)

    if len(loc_list_neg) > 0:
        negs = np.full(len(loc_list_neg), "neg")
        neglist = (
            np.vstack(
                (fnames, negs, loc_list_neg[:, 0], loc_list_neg[:, 1], objsizes_neg)
            )
        ).T

    else:
        neglist = np.zeros((1, 5))

    # FOR CRS
    thrdet = pimg_cr > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_cr = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_cr = np.array(loc_list_cr)

    objsizes_cr = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_cr), fname)
    if len(loc_list_cr) > 0:
        crs = np.full(len(loc_list_cr), "cr")
        # print(np.shape(fnames))
        # print(np.shape(crs))
        # print(np.shape(loc_list_cr[:,0]))
        # print(np.shape(loc_list_cr[:,1]))
        # print(np.shape(objsizes_cr))
        crlist = (
            np.vstack((fnames, crs, loc_list_cr[:, 0], loc_list_cr[:, 1], objsizes_cr))
        ).T
    else:
        crlist = np.zeros((1, 5))
    # FOR BOTH
    thrdet = pimg_both > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_both = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_both = np.array(loc_list_both)

    objsizes_both = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_both), fname)
    boths = np.full(len(loc_list_both), "both")

    if len(loc_list_both) > 0:
        bothlist = (
            np.vstack(
                (fnames, boths, loc_list_both[:, 0], loc_list_both[:, 1], objsizes_both)
            )
        ).T
    else:
        bothlist = np.zeros((1, 5))

    del pimg_cr
    del pimg_pos
    del pimg_neg
    del pimg_bg
    del pimg_streak
    del thrdet
    del loc_list_pos
    del loc_list_neg
    del objsizes_pos
    del objsizes_neg
    del labs
    del threshed
    gc.collect()

    # np.save(path, n_loc_size_ray_b)
    listlist = np.vstack((poslist, neglist, crlist, bothlist))

    alllist = pd.DataFrame(listlist)

    alllist.rename(
        columns={
            0: "filename",
            1: "type",
            2: "row_from_nn",
            3: "col_from_nn",
            4: "detection_size",
        },
        inplace=True,
    )

    alllist[["row_from_nn", "col_from_nn", "detection_size"]] = alllist[
        ["row_from_nn", "col_from_nn", "detection_size"]
    ].apply(pd.to_numeric)
    # np.save(path, n_loc_size_ray_b)
    return poslist, neglist, crlist, bothlist, alllist


def make_obj_list_3(image, predsarray, abst_thr, fname):
    nrow, ncol = image.shape
    wid = 10
    subshape = (ncol - 2 * wid, nrow - 2 * wid)

    # Precompute transposed images only once
    pimg_pos_T = predsarray[:, 1].reshape(subshape).T
    pimg_neg_T = predsarray[:, 2].reshape(subshape).T
    pimg_cr_T = predsarray[:, 3].reshape(subshape).T
    pimg_both_T = (predsarray[:, 1] + predsarray[:, 2]).reshape(subshape).T

    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # for what counts as one object

    def process_image(threshed, obj_type):
        labs, nlabs = snm.label(threshed, structure)
        if nlabs == 0:
            return None  # Skip if no labels

        loc_list = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
        loc_list = np.array(loc_list)
        objsizes = pr.get_objsizes(labs, nlabs)

        fnames = np.full(len(loc_list), fname)
        types = np.full(len(loc_list), obj_type)
        return np.vstack((fnames, types, loc_list[:, 0], loc_list[:, 1], objsizes)).T

    # List of object types and corresponding thresholded images
    obj_data = [
        (pimg_pos_T > abst_thr, "pos"),
        (pimg_neg_T > abst_thr, "neg"),
        (pimg_cr_T > abst_thr, "cr"),
        (pimg_both_T > abst_thr, "both"),
    ]

    # Collect results
    result_list = []
    for threshed, obj_type in obj_data:
        data = process_image(threshed, obj_type)
        if data is not None:
            result_list.append(data)

    # Stack all results together
    if result_list:
        alllist = np.vstack(result_list)
    else:
        alllist = np.zeros((1, 5))  # Handle the case where no objects are found

    # Convert to DataFrame and apply column names and conversions
    alllist = pd.DataFrame(
        alllist,
        columns=["filename", "type", "row_from_nn", "col_from_nn", "detection_size"],
    )
    alllist[["row_from_nn", "col_from_nn", "detection_size"]] = alllist[
        ["row_from_nn", "col_from_nn", "detection_size"]
    ].apply(pd.to_numeric)

    return alllist


# In[ ]:


# ,
# , 1019

# In[87]:


for j in range(len(direcnames)):
    direc = direcs[j]
    direcname = direcnames[j]
    wid = 10
    abst_thr = 0.9999
    fpath_extractions = "/lustre/lrspec/ZTF_data/science_extractions/" + direcname + "/"

    for i, file in enumerate(os.listdir(direc)):
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        image_filename_full = direc + image_filename

        dpath = fpath_extractions + image_filename + str(4) + "alllist"
        if (
            image_filename_full
            == "/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/ztf_20221020157697_000640_zi_c01_x_0010.fits.fz"
        ):
            continue

        if os.path.isfile(dpath) == False:
            print("about to do ", image_filename_full)
            for iteration in range(1, 5):
                hdul = fits.open(image_filename_full)
                file1_array = hdul[iteration].data
                norm = (file1_array - np.min(file1_array)) / (
                    np.max(file1_array) - np.min(file1_array)
                )
                image = norm
                nrow, ncol = image.shape
                path = (
                    fpath_extractions
                    + str(filename_start)
                    + "_extraction_predsarray_"
                    + str(iteration)
                    + "_predsarray.npy"
                )

                predsarray = np.load(path)

                alllist = make_obj_list_3(image, predsarray, abst_thr, filename_start)

                del norm
                del file1_array
                del image
                del hdul

                dpath = fpath_extractions + image_filename + str(iteration) + "alllist"

                alllist.to_csv(dpath)
                del alllist
                gc.collect()
