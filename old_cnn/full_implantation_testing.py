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
from tensorflow import keras
import get_postage_stamps as gpsf
import fitting


seed = np.random.seed(43)
size = 2000
center_values = np.random.randint(50, high=3000, size=size, dtype=int)
sigmas = np.random.randint(3, high=10, size=size, dtype=int)
intensities_neg = imp.get_intensities_neg(size, seed)


# In[14]:


fpath_image = "/lustre/lrspec/ZTF_data/training_aug24/implant_train/"
path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"
size = 2000  # will cut to 1000


counter = 0

# for iteration in range(1,4):
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
#                                                       center_values, counter=counter,
#                                                       plot=False)

#         np.save(imagepath, implanted)
#         np.save(infopath, transient_info)
#         counter+=(len(transient_info)+1)
#         if counter>1990:
#             break


# # In[11]:


# model_best=keras.models.load_model("/home/3155/CNN_sept2")


# In[68]:


# def save_predsarrays(image, model, wid, path):
#     '''
#     This function will take an image and a cnn model and return an array
#     of the transients found in that image.
#     image: np array
#     model: cnn model
#     abst_thr: absolute probability threshold to consider a transient probability as a transient
#     mult_thr: comparative probability threshold to consider a transient probability as a transient
#     (aka the trans prob has to be x times larger than (streak prob + backg prob)
#     wid: desired postage stamp width
#     bg_thresh: thresh to consider the average background on a column to be "background",
#     something like .1, used as criterion for determining cosmic rays
#     path: path to save the array

#     returns (and saves): array of length (number of transient detections),
#     with elements of (detection row, detection column), transient detection size,
#     realness (aka not-cosmic-ray)
#     so a list of tuples, then an int, then the probability that its column is a streak
#     to flag as a probable cosmic ray
#     '''


#     nrow, ncol = image.shape
#     print("getting postage stamps")
#     stamps = gpsf.get_postage_stamps_full(image, 10)
#     stampbatches = np.array_split(stamps,
#                           stamps.shape[0] // 20000) #(image.shape[0] - 2 * wid) )
#     #I WANNA CHANGE THIS TO SPEED HIM UP
#     #so to be faster there should be less batches
#     #so i want the second argument to be smaller
#     #so i want the denomitor of the second argument to be larger
#     preds = []
#     print("getting predictions:")
#     for stamps in stampbatches: #getting all the predictions. does not renormalize
#         pred_l = model.predict_on_batch(stamps);
#         preds.append(pred_l)
#     predsarray = np.concatenate(preds)
#     #i would like to save the predsarrays i think. for reproducibility
#     np.save(path+"_predsarray", predsarray) #TURN THIS BACK ON LATER
#     del stamps
#     del stampbatches
#     del preds
#     del predsarray
#     gc.collect()
#     return #predsarray


# In[91]:

# '/lustre/lrspec/ZTF_data/science_extractions/to_extract/20210622_minus_extras/',
#

direcs = ["/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/images/"]

# ,
# , 1019

# In[87]:


# for j in range(len(direcs)):
#     direc = direcs[j]
#     model = model_best
#     wid = 10
#     abst_thr = .9999
#     fpath_extractions = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays/"


#     for i, file in enumerate(os.listdir(direc)):
#         image_filename = os.fsdecode(file)
#         filename_start = image_filename[:40]
#         image_filename_full = direc+image_filename
#         if os.path.isfile(fpath_extractions+str(filename_start)+"_extraction_predsarray_predsarray_neg.npy"):
#             print("skipping ",filename_start)
#             continue

#         print(image_filename_full)
#         file1_array = np.load(image_filename_full)
#         norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array))
#         image = norm
#         nrow, ncol = image.shape
#         path = fpath_extractions+str(filename_start)+"_extraction_predsarray_neg"

#         save_predsarrays(image, model, wid, path)
#         del norm
#         del file1_array
#         del image
#         gc.collect()


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
        poslist = ["hi", "pos", 0, 0, 0]

    # FOR NEG
    thrdet = pimg_neg > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_neg = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_neg = np.array(loc_list_neg)

    objsizes_neg = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_neg), fname)
    negs = np.full(len(loc_list_neg), "neg")

    if len(loc_list_neg) > 0:
        neglist = (
            np.vstack(
                (fnames, negs, loc_list_neg[:, 0], loc_list_neg[:, 1], objsizes_neg)
            )
        ).T

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
        crlist = 0
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
    if crlist != 0:
        listlist = np.vstack((poslist, neglist, crlist, bothlist))
    else:
        listlist = np.vstack((poslist, neglist, bothlist))
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


# ,
# , 1019

# In[87]:


# for j in range(len(direcs)):
#     direc = direcs[j]
#     wid = 10
#     abst_thr = .9999
#     fpath_extractions = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays/"


#     for i, file in enumerate(os.listdir(direc)):
#         image_filename = os.fsdecode(file)
#         filename_start = image_filename[:40]
#         image_filename_full = direc+image_filename

#         print(image_filename_full)
#         iteration=1
#         file1_array = np.load(image_filename_full)
#         norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array))
#         image = norm
#         nrow, ncol = image.shape
#         path = fpath_extractions+str(filename_start)+"_extraction_predsarray_neg_predsarray.npy"

#         predsarray = np.load(path)

#         poslist, neglist, crlist, bothlist, alllist = make_obj_list(image, predsarray, abst_thr, filename_start)

#         del norm
#         del file1_array
#         del image

#         dpath = fpath_extractions+image_filename+str(iteration)+'alllist_neg'

#         alllist.to_csv(dpath)
#         del poslist
#         del neglist
#         del crlist
#         del bothlist
#         del alllist
#         gc.collect()

# Step 1: Set the folder path where your DataFrames are stored
folder_path = (
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays"
)
path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"

# Step 2: List all the files in the folder
files = [
    f for f in os.listdir(folder_path) if f.endswith("alllist_neg")
]  # Adjust this to your file format

# Step 3: Read all DataFrames and add the filename as a new column
df_detections = pd.DataFrame([])
counter = 0

for f in files:
    df = pd.read_csv(os.path.join(folder_path, f))
    filename_start = df["filename"][1]
    infopath = (
        path_to_save
        + "info/"
        + str(filename_start)
        + "_"
        + str(1)
        + "_implanted_info_neg.npy"
    )
    info = np.load(infopath)
    df_cleaned = df.drop_duplicates(subset=["filename", "row_from_nn", "col_from_nn"])

    for index in df_cleaned.index:
        for infos in info:
            if (
                np.abs(df_cleaned.loc[index]["row_from_nn"] + 10 - infos[1]) < 20
                and np.abs(df_cleaned.loc[index]["col_from_nn"] + 10 - infos[0]) < 20
            ):
                print(df_cleaned.loc[index]["row_from_nn"])
                print(infos[0])
                print(df_cleaned.loc[index]["col_from_nn"])
                print(infos[1])
                if counter == 0:
                    df_detections["filename"] = [df_cleaned.loc[index]["filename"]]
                    df_detections["detection_size"] = [
                        df_cleaned.loc[index]["detection_size"]
                    ]
                    df_detections["type"] = [df_cleaned.loc[index]["type"]]
                    df_detections["row_from_nn"] = [
                        df_cleaned.loc[index]["row_from_nn"]
                    ]
                    df_detections["col_from_nn"] = [
                        df_cleaned.loc[index]["col_from_nn"]
                    ]
                    df_detections["row_from_implant"] = [infos[0]]
                    df_detections["col_from_implant"] = [infos[1]]
                    df_detections["A_from_implant"] = [infos[2]]
                    df_detections["sig_from_implant"] = [infos[3]]
                    counter = 1
                else:
                    print("adding")
                    df_detections2 = pd.DataFrame([])

                    df_detections2["filename"] = [df_cleaned.loc[index]["filename"]]
                    df_detections2["detection_size"] = [
                        df_cleaned.loc[index]["detection_size"]
                    ]
                    df_detections2["type"] = [df_cleaned.loc[index]["type"]]
                    df_detections2["row_from_nn"] = [
                        df_cleaned.loc[index]["row_from_nn"]
                    ]
                    df_detections2["col_from_nn"] = [
                        df_cleaned.loc[index]["col_from_nn"]
                    ]
                    df_detections2["row_from_implant"] = [infos[0]]
                    df_detections2["col_from_implant"] = [infos[1]]
                    df_detections2["A_from_implant"] = [infos[2]]
                    df_detections2["sig_from_implant"] = [infos[3]]

                    df_detections = pd.concat(
                        [df_detections2, df_detections], ignore_index=True
                    )

# Step 4: Concatenate all DataFrames into one
# concatenated_df = pd.concat(dataframes, ignore_index=True)

# Display the concatenated DataFrame
# concatenated_df

df_detections.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/df_detections_neg"
)


def add_snr(df, half_aperture, tracking, fpath):
    """
    returns the dataframe with a new column of stdv ratios
    see get_sobel_ratio(stamp) and get_stdv_ratio
    """
    ratios_center = np.zeros(len(df))
    ratios_max = np.zeros(len(df))
    ratios_min = np.zeros(len(df))
    for index in df.index:
        # print(index)
        tot, maxcols = get_tot_npy(df, index, half_aperture, tracking, fpath)
        stdv = rff.get_stdvs(tot, do_all=False, df=df, index=index)
        avg = np.average(tot)
        row = int(df.loc[index]["row_from_nn"]) + 10
        max_val = np.max(tot[row - 10 : row + 10])
        min_val = np.min(tot[row - 10 : row + 10])
        signal_center = tot[row] - avg
        signal_max = max_val - avg
        signal_min = min_val - avg
        ratios_center[index] = signal_center / stdv
        ratios_max[index] = signal_max / stdv
        ratios_min[index] = signal_min / stdv
    df["snr_center"] = ratios_center
    df["snr_max"] = ratios_max
    df["snr_min"] = ratios_min
    return df


df_detections["quadrant"] = np.full(len(df_detections), 1)


def get_tot_npy(
    df,
    index,
    half_aperture=2,
    tracking=True,
    fpath="/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/",
):
    """
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    note: uses loc instead of iloc
    """
    wideval = half_aperture
    wid = 10
    fname = df.loc[index]["filename"] + "_1_implanted.npy"
    row = (
        int(df.loc[index]["row_from_nn"]) + 10
    )  # we add 10 here bc that's the cnn offset of returned values
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)

    path = fpath + fname
    # print(path)
    file1_array = np.load(path)
    # norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array

    background = rff.get_image_background(image)

    imagenobackground = image - np.full(np.shape(image), background)

    if tracking == False:
        zeropoint = col - 7
        averages = np.average(
            imagenobackground[row - 10 : row + 10, zeropoint : zeropoint + 14], axis=0
        )
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(
            0, imagenobackground.shape[0]
        ):  # this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1, half_aperture + 1):
                combined += imagenobackground[r, maxcol - i]
                combined += imagenobackground[r, maxcol + i]
            tot2[r] = combined
        return tot2, np.full(len(tot2), maxcol)

    # so we need to track the whole streak
    # for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col - 10
    for r in range(0, imagenobackground.shape[0]):  # this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint : zeropoint + 20]) + zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1, half_aperture + 1):
            combined += imagenobackground[r, maxcol - i]
            combined += imagenobackground[r, maxcol + i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined

    return tot, maxcols


df_detections_snr = add_snr(
    df_detections,
    half_aperture=2,
    tracking=True,
    fpath="/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/images/",
)

df_detections_snr.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/df_detections_snr_neg"
)

df_detections_snr_sorted = df_detections_snr.sort_values("snr_min")

df_detections_snr_cleaned = df_detections_snr_sorted.drop_duplicates(
    subset=["filename", "row_from_implant", "col_from_implant"], keep="first"
)

df_detections_fit = fitting.fit_df(df_detections_snr_cleaned)

df_detections_fit.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/df_detections_fit_neg"
)

percents_A = np.zeros(len(np.unique(df_fit_neg["A_from_implant"])))
percents_sig = np.zeros(len(np.unique(df_fit_neg["A_from_implant"])))
numbers = np.zeros(len(np.unique(df_fit_neg["A_from_implant"])))
diffs_M = np.zeros(len(np.unique(df_fit_neg["A_from_implant"])))
for i, A_val in enumerate(np.unique(df_fit_neg["A_from_implant"])):
    df = df_fit_neg[df_fit_neg["A_from_implant"] == A_val]
    if len(df) > 100:
        df = df[:100]
    percent_A = np.sum(
        ((df["A_from_implant"] - df["optimized_A"]) ** 2) / (df["A_from_implant"]) ** 2
    )
    percent_sig = np.sum(
        ((df["sig_from_implant"] - df["optimized_sig"]) ** 2)
        / (df["A_from_implant"]) ** 2
    )
    # percent_M = np.sum(((df['col_from_implant'] - df['optimized_M']) ** 2)/(df['A_from_implant'])**2) #WAS COLUMN
    diff_M = np.average(
        np.abs(df["col_from_implant"] - df["optimized_M"]) / df["row_from_implant"]
    )
    percents_A[i] = percent_A / len(df)
    percents_sig[i] = percent_sig / len(df)
    numbers[i] = len(df)
    diffs_M[i] = diff_M

df_percents_neg2 = pd.DataFrame(
    [
        np.unique(df_fit_neg["A_from_implant"]).T,
        numbers.T,
        percents_sig.T,
        diffs_M.T,
        percents_A.T,
    ]
)


df_percents.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/df_percents_neg_20"
)
