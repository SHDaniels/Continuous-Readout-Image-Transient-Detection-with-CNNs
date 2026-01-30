#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""

trying this again in jan to see about the flipping - everything basically the same
except filepaths and
changed the fits opening to a with ...as
(also cleaned up imports)

gonna go through all the real images woooo


partially copied from last_full_image_extractions but for the larger set of implanted images 
with the new CNN

i make figures here but they are coppied and improved in plots_figures


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


def save_predsarrays(image, model, wid, path):
    """
    This function will take an image and a cnn model and return an array
    of the transients found in that image.
    image: np array
    model: cnn model
    abst_thr: absolute probability threshold to consider a transient probability as a transient
    mult_thr: comparative probability threshold to consider a transient probability as a transient
    (aka the trans prob has to be x times larger than (streak prob + backg prob)
    wid: desired postage stamp width
    bg_thresh: thresh to consider the average background on a column to be "background",
    something like .1, used as criterion for determining cosmic rays
    path: path to save the array

    returns (and saves): array of length (number of transient detections),
    with elements of (detection row, detection column), transient detection size,
    realness (aka not-cosmic-ray)
    so a list of tuples, then an int, then the probability that its column is a streak
    to flag as a probable cosmic ray
    """

    nrow, ncol = image.shape
    print("getting postage stamps")
    stamps = gpsf.get_postage_stamps_full(image, 10)
    stampbatches = np.array_split(
        stamps, stamps.shape[0] // 20000
    )  # (image.shape[0] - 2 * wid) )
    # I WANNA CHANGE THIS TO SPEED HIM UP
    # so to be faster there should be less batches
    # so i want the second argument to be smaller
    # so i want the denomitor of the second argument to be larger
    preds = []
    print("getting predictions:")
    for stamps in tqdm(
        stampbatches
    ):  # getting all the predictions. does not renormalize
        pred_l = model.predict_on_batch(stamps)
        preds.append(pred_l)
    predsarray = np.concatenate(preds)
    # i would like to save the predsarrays i think. for reproducibility
    np.save(path + "_predsarray", predsarray)  # TURN THIS BACK ON LATER
    del stamps
    del stampbatches
    del preds
    del predsarray
    gc.collect()
    return  # predsarray


# In[91]:

# '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/',
#

data_direc = [
    "/lustre/lrspec/ZTF_data/2022/1020/"
    ]

save_direc = [
    "1020jan26"
]
# ,
# , 1019

# In[87]:

original_extraction_direc = "/lustre/lrspec/ZTF_data/science_extractions/1020/"


for j in range(len(data_direc)):
    data_direc = data_direc[j]
    save_direc = save_direc[j]
    model = model_best
    wid = 10
    abst_thr = 0.9999
    fpath_extractions_save = "/lustre/lrspec/ZTF_data/science_extractions/" + save_direc + "/"
    

    for i, file in enumerate(os.listdir(original_extraction_direc)): #if we originally had it
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

            save_predsarrays(image, model, wid, path)
            del norm
            del file1_array
            del image
            # del hdul #may not need this when we have with as
            gc.collect()

