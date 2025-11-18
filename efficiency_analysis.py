#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
efficiency analysis
'''


# In[3]:


import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.ndimage as snm
from scipy.signal import find_peaks


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
from scipy.ndimage import sobel


def add_snr(df, half_aperture, tracking, fpath, extension):
    """
    returns the dataframe with a new column of stdv ratios
    see get_sobel_ratio(stamp) and get_stdv_ratio
    """
    ratios_center = np.zeros(len(df))
    ratios_max = np.zeros(len(df))
    ratios_min = np.zeros(len(df))
    for i, index in enumerate(df.index):
        # print(index)
        tot, maxcols = get_tot_npy(df, index, extension, half_aperture, tracking, fpath)
        stdv = rff.get_stdvs(tot, do_all=False, df=df, index=index)
        avg = np.average(tot)
        row = int(df.loc[index]["row_from_nn"]) + 10
        max_val = np.max(tot[row - 10 : row + 10])
        min_val = np.min(tot[row - 10 : row + 10])
        signal_center = tot[row] - avg
        signal_max = max_val - avg
        signal_min = min_val - avg
        ratios_center[i] = signal_center / stdv
        ratios_max[i] = signal_max / stdv
        ratios_min[i] = signal_min / stdv
    df["snr_center"] = ratios_center
    df["snr_max"] = ratios_max
    df["snr_min"] = ratios_min
    return df




def get_tot_npy(
    df,
    index, extension,
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
    fname = df.loc[index]["filename"] + extension
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


def get_stamp_from_df(index, fpath, df, extension, norm, quad=None):
    """
    returns a 20x20 numpy array of brightnesses centered around the detection
    located at index of dataframe df
    note that the filepath to the science images is hardcoded currently
    norm=True has the image normalized (not the stamp, the entire image)
    norm=False has it in terms of counts
    """
    fname = df.loc[index]["filename"] + extension
    row = int(df.loc[index]["row_from_nn"]) + 10
    col = int(df.loc[index]["col_from_nn"]) + 10
    if quad==None:
        quad = int(df.loc[index]["quadrant"])
    # if direcname == "0820":
    # fpath = (
    #     "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"
    #     + direcname
    #     + "_minus_extras/"
    # )
    # else:
    # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"/"
    path = fpath + fname
    # hdul = fits.open(path)
    file1_array = np.load(path)
    if norm:
        normalized = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )  # norm
        stamp = normalized[row - 10 : row + 10, col - 10 : col + 10]
        # plt.imshow(stamp)
        # plt.colorbar()
        return stamp
        background = rff.get_image_background(normalized)
    stamp = file1_array[row - 10 : row + 10, col - 10 : col + 10]
    # plt.imshow(stamp)
    # plt.colorbar()
    background = rff.get_image_background(file1_array)
    plt.imshow(stamp)
    return stamp, background


def add_sobel_ratio(fpath, df, extension):
    """
    returns the dataframe with a new column of sobel ratios
    see get_sobel_ratio(stamp)
    """
    sobels = []
    for index in df.index:
        stamp, bg = get_stamp_from_df(index, fpath, df, extension, norm=False)
        sobelr = get_sobel_ratio(stamp)
        sobels.append(sobelr)
    df["sobel_ratio"] = sobels
    return df


def add_snr_integrated(df, extension, fpath, half_aperture=2, tracking=True, wid=10):
    """
    returns the dataframe with new columns for outputs of
    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
    """
    column_names = ['snr_local', 'snr_global', 'noise_local', 'noise_global', 
                'local_average_outside', 'height', 'FWHM', 'signal_local']

    for col in column_names:
        df[col] = np.zeros(len(df))
    
    for index in df.index:
        # print(index)
        tot, maxcols = get_tot_npy(df, index, extension, half_aperture,  tracking, fpath)
        row = int(df.loc[index]["row_from_nn"]) + wid
        snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
        df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
    return df





def has_two_streaks(df, index, fpath, extension, prominence=15, return_properties=False):
    '''
    will return true if it has two streaks, otherwise false
    '''
    fname = df.loc[index]["filename"]+extension
    row = int(df.loc[index]["row_from_nn"])+10
    col = int(df.loc[index]["col_from_nn"])+10
    quad = int(df.loc[index]["quadrant"])
    # if direcname == "0820":
    # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
    # else:
        # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"/"
    path = fpath+fname
    # print(path)
    # hdul = fits.open(path)
    # file1_array = hdul[quad].data
    file1_array = np.load(path)

    averages = np.average(file1_array[row-15:row+15, col-15:col+15], axis=0)
    plt.plot(averages)
    peaks, properties = find_peaks(averages, prominence=prominence)
    # print(peaks)
    # print(properties['prominences'])
    if len(peaks)>1:
        if return_properties==True:
            return True, properties
        return True #, properties
    else:
        return False

    


# In[70]:


def add_twostreaks(df, extension, fpath, name="has_two_streaks",prominence=20):
    two_streaks = []
    for index in df.index:
        value = has_two_streaks(df, index, fpath, extension, prominence=prominence)
        two_streaks.append(value)
    df[name] = two_streaks
    return df





# #

# folder_path = (
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays"
# )
# path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"

# # Step 2: List all the files in the folder
# files = [
#     f for f in os.listdir(folder_path) if f.endswith("alllist_neg")
# ]  # Adjust this to your file format

# # Step 3: Read all DataFrames and add the filename as a new column
# df_detections = pd.DataFrame([])
# counter = 0

# for f in files:
#     df = pd.read_csv(os.path.join(folder_path, f))
#     filename_start = df["filename"][1]
#     infopath = (
#         path_to_save
#         + "info/"
#         + str(filename_start)
#         + "_"
#         + str(1)
#         + "_implanted_info_neg.npy"
#     )
#     info = np.load(infopath)
#     # df_cleaned = df.drop_duplicates(subset=["filename", "row_from_nn", "col_from_nn"])
#     df_cleaned = df
#     for index in df_cleaned.index:
#         for infos in info:
#             if (
#                 np.abs(df_cleaned.loc[index]["row_from_nn"] + 10 - infos[1]) < 20
#                 and np.abs(df_cleaned.loc[index]["col_from_nn"] + 10 - infos[0]) < 20
#             ):
#                 print(df_cleaned.loc[index]["row_from_nn"])
#                 print(infos[0])
#                 print(df_cleaned.loc[index]["col_from_nn"])
#                 print(infos[1])
#                 if counter == 0:
#                     df_detections["filename"] = [df_cleaned.loc[index]["filename"]]
#                     df_detections["detection_size"] = [
#                         df_cleaned.loc[index]["detection_size"]
#                     ]
#                     df_detections["type"] = [df_cleaned.loc[index]["type"]]
#                     df_detections["row_from_nn"] = [
#                         df_cleaned.loc[index]["row_from_nn"]
#                     ]
#                     df_detections["col_from_nn"] = [
#                         df_cleaned.loc[index]["col_from_nn"]
#                     ]
#                     df_detections["row_from_implant"] = [infos[0]]
#                     df_detections["col_from_implant"] = [infos[1]]
#                     df_detections["A_from_implant"] = [infos[2]]
#                     df_detections["sig_from_implant"] = [infos[3]]
#                     counter = 1
#                 else:
#                     print("adding")
#                     df_detections2 = pd.DataFrame([])

#                     df_detections2["filename"] = [df_cleaned.loc[index]["filename"]]
#                     df_detections2["detection_size"] = [
#                         df_cleaned.loc[index]["detection_size"]
#                     ]
#                     df_detections2["type"] = [df_cleaned.loc[index]["type"]]
#                     df_detections2["row_from_nn"] = [
#                         df_cleaned.loc[index]["row_from_nn"]
#                     ]
#                     df_detections2["col_from_nn"] = [
#                         df_cleaned.loc[index]["col_from_nn"]
#                     ]
#                     df_detections2["row_from_implant"] = [infos[0]]
#                     df_detections2["col_from_implant"] = [infos[1]]
#                     df_detections2["A_from_implant"] = [infos[2]]
#                     df_detections2["sig_from_implant"] = [infos[3]]

#                     df_detections = pd.concat(
#                         [df_detections2, df_detections], ignore_index=True
#                     )

# df_detections_neg = df_detections





# #now pos

# # Step 1: Set the folder path where your DataFrames are stored
# folder_path = (
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays"
# )
# path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"

# # Step 2: List all the files in the folder
# files = [
#     f for f in os.listdir(folder_path) if f.endswith("1alllist")
# ]  # Adjust this to your file format

# # Step 3: Read all DataFrames and add the filename as a new column
# df_detections = pd.DataFrame([])
# counter = 0

# for f in files:
#     df = pd.read_csv(os.path.join(folder_path, f))
#     filename_start = df["filename"][1]
#     infopath = (
#         path_to_save
#         + "info/"
#         + str(filename_start)
#         + "_"
#         + str(1)
#         + "_implanted_info.npy"
#     )
#     info = np.load(infopath)
#     # df_cleaned = df.drop_duplicates(subset=["filename", "row_from_nn", "col_from_nn"])
#     df_cleaned = df
    
#     for index in df_cleaned.index:
#         for infos in info:
#             if (
#                 np.abs(df_cleaned.loc[index]["row_from_nn"] + 10 - infos[0]) < 10
#                 and np.abs(df_cleaned.loc[index]["col_from_nn"] + 10 - infos[1]) < 10
#             ):
#                 print(df_cleaned.loc[index]["row_from_nn"])
#                 print(infos[0])
#                 print(df_cleaned.loc[index]["col_from_nn"])
#                 print(infos[1])
#                 if counter == 0:
#                     df_detections["filename"] = [df_cleaned.loc[index]["filename"]]
#                     df_detections["detection_size"] = [
#                         df_cleaned.loc[index]["detection_size"]
#                     ]
#                     df_detections["type"] = [df_cleaned.loc[index]["type"]]
#                     df_detections["row_from_nn"] = [
#                         df_cleaned.loc[index]["row_from_nn"]
#                     ]
#                     df_detections["col_from_nn"] = [
#                         df_cleaned.loc[index]["col_from_nn"]
#                     ]
#                     df_detections["row_from_implant"] = [infos[0]]
#                     df_detections["col_from_implant"] = [infos[1]]
#                     df_detections["A_from_implant"] = [infos[2]]
#                     df_detections["sig_from_implant"] = [infos[3]]
#                     counter = 1
#                 else:
#                     print("adding")
#                     df_detections2 = pd.DataFrame([])

#                     df_detections2["filename"] = [df_cleaned.loc[index]["filename"]]
#                     df_detections2["detection_size"] = [
#                         df_cleaned.loc[index]["detection_size"]
#                     ]
#                     df_detections2["type"] = [df_cleaned.loc[index]["type"]]
#                     df_detections2["row_from_nn"] = [
#                         df_cleaned.loc[index]["row_from_nn"]
#                     ]
#                     df_detections2["col_from_nn"] = [
#                         df_cleaned.loc[index]["col_from_nn"]
#                     ]
#                     df_detections2["row_from_implant"] = [infos[0]]
#                     df_detections2["col_from_implant"] = [infos[1]]
#                     df_detections2["A_from_implant"] = [infos[2]]
#                     df_detections2["sig_from_implant"] = [infos[3]]

#                     df_detections = pd.concat(
#                         [df_detections2, df_detections], ignore_index=True
#                     )

# # Step 4: Concatenate all DataFrames into one
# # concatenated_df = pd.concat(dataframes, ignore_index=True)

# # Display the concatenated DataFrame
# # concatenated_df

# df_detections_pos = df_detections



# # In[7]:


# # Step 1: Set the folder path where your DataFrames are stored
# folder_path = (
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays"
# )
# path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"

# # Step 2: List all the files in the folder
# files = [
#     f for f in os.listdir(folder_path) if f.endswith("alllist_neg")
# ]  # Adjust this to your file format

# # Step 3: Read all DataFrames and add the filename as a new column
# df_implants = pd.DataFrame([])
# counter = 0

# for f in files:
#     df = pd.read_csv(os.path.join(folder_path, f))
#     filename_start = df["filename"][1]
#     infopath = (
#         path_to_save
#         + "info/"
#         + str(filename_start)
#         + "_"
#         + str(1)
#         + "_implanted_info_neg.npy"
#     )
#     infos = np.load(infopath)
#     # print(info)
#     if counter == 0:
#          df_implants["filename"] = np.full(len(infos),filename_start)

#          df_implants["row_from_implant"] = infos[:,0]
#          df_implants["col_from_implant"] = infos[:,1]
#          df_implants["A_from_implant"] = infos[:,2]
#          df_implants["sig_from_implant"] = infos[:,3]
#          counter = 1
#     else:
#          print("adding")
#          df_implants2 = pd.DataFrame([])

#          df_implants2["filename"] = np.full(len(infos),filename_start)

#          df_implants2["row_from_implant"] = infos[:,0]
#          df_implants2["col_from_implant"] = infos[:,1]
#          df_implants2["A_from_implant"] = infos[:,2]
#          df_implants2["sig_from_implant"] = infos[:,3]

#          df_implants = pd.concat(
#             [df_implants2, df_implants], ignore_index=True
#                     )
# df_implants_neg = df_implants



# # In[8]:


# # Step 1: Set the folder path where your DataFrames are stored
# folder_path = (
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/predsarrays"
# )
# path_to_save = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/"

# # Step 2: List all the files in the folder
# files = [
#     f for f in os.listdir(folder_path) if f.endswith("npy1alllist")
# ]  # Adjust this to your file format

# # Step 3: Read all DataFrames and add the filename as a new column
# df_implants = pd.DataFrame([])
# counter = 0

# for f in files:
#     df = pd.read_csv(os.path.join(folder_path, f))
#     filename_start = df["filename"][1]
#     infopath = (
#         path_to_save
#         + "info/"
#         + str(filename_start)
#         + "_"
#         + str(1)
#         + "_implanted_info.npy"
#     )
#     infos = np.load(infopath)
#     # print(info)
#     if counter == 0:
#          df_implants["filename"] = np.full(len(infos),filename_start)

#          df_implants["row_from_implant"] = infos[:,0]
#          df_implants["col_from_implant"] = infos[:,1]
#          df_implants["A_from_implant"] = infos[:,2]
#          df_implants["sig_from_implant"] = infos[:,3]
#          counter = 1
#     else:
#          print("adding")
#          df_implants2 = pd.DataFrame([])

#          df_implants2["filename"] = np.full(len(infos),filename_start)

#          df_implants2["row_from_implant"] = infos[:,0]
#          df_implants2["col_from_implant"] = infos[:,1]
#          df_implants2["A_from_implant"] = infos[:,2]
#          df_implants2["sig_from_implant"] = infos[:,3]

#          df_implants = pd.concat(
#             [df_implants2, df_implants], ignore_index=True
#                     )
# df_implants_pos = df_implants




# # df_detections_pos = pd.read_csv("/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/df_detections")


# # In[12]:



# # In[13]:

# df_implants_neg_big = df_implants_neg[df_implants_neg["A_from_implant"]<=.9]


# # In[14]:


# df_implants_pos_big = df_implants_pos[df_implants_pos["A_from_implant"]>=1.1]


# # In[15]:


# print("size of initial num", len(df_implants_pos_big), len(df_implants_neg_big))


# # In[16]:


# df_detections_pos_big = df_detections_pos[df_detections_pos["A_from_implant"]>=1.1]
# df_detections_neg_big = df_detections_neg[df_detections_neg["A_from_implant"]<=.9]

# df_detections_pos_big.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_detections_pos_big"
# )
# df_detections_neg_big.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_detections_neg_big"
# )

# df_implants_neg_big.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_implants_neg_big"
# )

# df_implants_pos_big.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_implants_pos_big"
# )



# df_detections_neg_big=pd.read_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_detections_neg_big"
# )


# df_detections_pos_big=pd.read_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_detections_pos_big"
# )



# #pos
# df_nocr_pos = df_detections_pos_big[df_detections_pos_big["type"] != "cr"]
# df_nocr_pos.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_nocr_pos"
# )
# df_reduced_edges_pos= rff.reduce_edges(df_nocr_pos, 15)
# df_reduced_edges_pos.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_pos"
# )
# df_reduced_edges_size4_pos = rff.reduce_size(df_reduced_edges_pos, 4)
# df_reduced_edges_size4_pos.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_size4_pos"
# )

df_reduced_edges_size4_pos = pd.read_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_size4_pos"
)


count=0
for filename in np.unique(df_reduced_edges_size4_pos["filename"]):
    
    df_for_file = df_reduced_edges_size4_pos[df_reduced_edges_size4_pos["filename"]==filename]
    # print(df_for_file)
    fname = filename+"_1_implanted.npy"
    fpath = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/images/"
    path = fpath+fname
    # hdul=fits.open(path)
    # image = hdul[1].data
    image = np.load(path)
    #now have to scale it up...
    # plt.imshow(image)
    background = rff.get_image_background(image)
#         #subtracting background
    imagenobackground = image - np.full(np.shape(image),background)
    reduced_df = rff.reduce_max_3(df_for_file, imagenobackground)
    if count==0:
        reduced_df2 = reduced_df
        count=1
    else:
        reduced_df2 = pd.concat(
            [reduced_df2, reduced_df], ignore_index=True
                    )
df_reducemax_pos = reduced_df2

df_reducemax_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reducemax_pos")


# df_reducemax_pos=pd.read_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reducemax_pos")



df_reducemax_pos["quadrant"] = np.full(len(df_reducemax_pos), 1)


# In[32]:


fpath = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/images/"


# In[38]:


df_sobel_pos = add_sobel_ratio(fpath, df_reducemax_pos, extension="_1_implanted.npy")

df_sobel_pos.to_csv("/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_sobel_pos")


df_snr_pos = df_sobel_pos[df_sobel_pos["sobel_ratio"]>5]


# In[41]:


df_snr_pos = add_snr(df_snr_pos, half_aperture=2, tracking=True, fpath=fpath, extension="_1_implanted.npy")


# In[45]:


df_snr_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snr_pos")





df_snrcut_pos = df_snr_pos[np.abs(df_snr_pos["snr_center"])<25]





df_snrall_pos = add_snr_integrated(df_snrcut_pos, extension="_1_implanted.npy", fpath=fpath)


df_snrall_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snrall_pos")


 


df_wstreaks_pos = add_twostreaks(df_snrall_pos, extension="_1_implanted.npy", fpath=fpath, name="has_two_streaks",prominence=15)


df_wstreaks_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_wstreaks_pos")


df_onestreak_pos = df_wstreaks_pos[df_wstreaks_pos["has_two_streaks"]==False]


df_onestreak_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_onestreak_pos")


df_snrcut_pos = df_onestreak_pos[df_onestreak_pos["snr_local"]>1]

df_snrcut_pos.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snrcut_pos")
    
    
#     #NEG BELOW HERE
    
    
    
    

    
# df_nocr_neg = df_detections_neg_big[df_detections_neg_big["type"] != "cr"]
# df_nocr_neg.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_nocr_neg"
# )
# df_reduced_edges_neg= rff.reduce_edges(df_nocr_neg, 15)
# df_reduced_edges_neg.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_neg"
# )
# df_reduced_edges_size4_neg = rff.reduce_size(df_reduced_edges_neg, 4)
# df_reduced_edges_size4_neg.to_csv(
#     "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_size4_neg"
# )

df_reduced_edges_size4_neg= pd.read_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reduced_edges_size4_neg"
)


count=0
for filename in np.unique(df_reduced_edges_size4_neg["filename"]):
    
    df_for_file = df_reduced_edges_size4_neg[df_reduced_edges_size4_neg["filename"]==filename]
    # print(df_for_file)
    fname = filename+"_1_implanted_neg.npy"
    fpath = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/images/"
    path = fpath+fname
    # hdul=fits.open(path)
    # image = hdul[1].data
    image = np.load(path)
    #now have to scale it up...
    # plt.imshow(image)
    background = rff.get_image_background(image)
#         #subtracting background
    imagenobackground = image - np.full(np.shape(image),background)
    reduced_df = rff.reduce_max_3(df_for_file, imagenobackground)
    if count==0:
        reduced_df2 = reduced_df
        count=1
    else:
        reduced_df2 = pd.concat(
            [reduced_df2, reduced_df], ignore_index=True
                    )
df_reducemax_neg = reduced_df2

df_reducemax_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reducemax_neg")


# df_reducemax_neg=pd.read_csv(
    # "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_reducemax_neg")




df_reducemax_neg["quadrant"] = np.full(len(df_reducemax_neg), 1)


# In[32]:


fpath = "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/images/"


# In[38]:


df_sobel_neg = add_sobel_ratio(fpath, df_reducemax_neg, extension="_1_implanted_neg.npy")

df_sobel_neg.to_csv("/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_sobel_neg")


df_snr_neg = df_sobel_neg[df_sobel_neg["sobel_ratio"]>5]


# In[41]:


df_snr_neg = add_snr(df_snr_neg, half_aperture=2, tracking=True, fpath=fpath, extension="_1_implanted_neg.npy")


# In[45]:


df_snr_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snr_neg")





df_snrcut_neg = df_snr_neg[np.abs(df_snr_neg["snr_center"])<25]





df_snrall_neg = add_snr_integrated(df_snrcut_neg, extension="_1_implanted_neg.npy", fpath=fpath)


df_snrall_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snrall_neg")


 


df_wstreaks_neg = add_twostreaks(df_snrall_neg, extension="_1_implanted_neg.npy", fpath=fpath, name="has_two_streaks",prominence=15)


df_wstreaks_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_wstreaks_neg")


df_onestreak_neg = df_wstreaks_neg[df_wstreaks_neg["has_two_streaks"]==False]


df_onestreak_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_onestreak_neg")


df_snrcut_neg = df_onestreak_neg[df_onestreak_neg["snr_local"]>1]

df_snrcut_neg.to_csv(
    "/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/fullrange/df_snrcut_neg")

#todo at end:
#check duplicates
#see how many there are that were mischaracterized as pos/neg
