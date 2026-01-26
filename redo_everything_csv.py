import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt


import precision_recall as pr


# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os
import get_postage_stamps as gpsf
import fitting

from scipy.signal import find_peaks





wid = 10
abst_thr = .9999


folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020"

# #let's count how many pixels are classified as what
# total_num_pos2 = 0
# total_num_cr2 = 0
# total_num_both2 = 0
# total_num_neg2 = 0

# df_detections = pd.DataFrame([])
# counter = 0
# # # List all the files in the folder
# files = [f for f in os.listdir(folder_path) if f.endswith('alllist')]  # Adjust this to your file format
# for file in files:
#     df = pd.read_csv(os.path.join(folder_path, file))
    
# # #pos
#     df_nocr = df[df["type"] != "cr"]
#     df_noboth = df_nocr[df_nocr["type"] != "both"]
#     df_reduced_edges= rff.reduce_edges(df_noboth, 15)
    
#     quad = file[-8]
#     newpath = folder_path+"/redo_nov/"
    
#     xy_path = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"
#     xy_file = xy_path+file[:-20]+quad+".fits"
    
#     data_list = []
    
    
#     # closest_amounts_c = []
#     with fits.open(xy_file) as hdul_xy:
#         for i, rrow in df_reduced_edges.iterrows():
#             col = rrow["col_from_nn"]
#             target_x = col+10
#         # print(xy_file)
#             data = hdul_xy[1].data  
#             stored_x = data["X"][:25]  # Column "X"
#             stored_y = data["Y"][:25]
#             closest_idx = np.argmin(np.abs(stored_x - target_x))
#             closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
#             if np.abs(closest_x-target_x)<=10:
#                 # plt.imshow(np.sqrt(file1_array[(int(closest_y))-20:(int(closest_y))+20, (int(closest_x))-20:(int(closest_x))+20]))
#                 # plt.scatter(20,20)
#                 # plt.show()    
#                     # Retrieve the closest (x, y) match
#                 data_list.append({"filename": rrow["filename"], "quad":quad, "type":rrow["type"],"row_from_nn":rrow["row_from_nn"],  
#                                   "col_from_nn":rrow["col_from_nn"], "detection_size":rrow["detection_size"],
#                                  "closest_x": closest_x, "closest_y":closest_y, "Diff":np.abs(closest_x-target_x)})
#                 dframe = pd.DataFrame(data_list)
#                 dframe.to_csv(newpath+file+"_matched_corrected")
    
newpath = folder_path+"/redo_nov/"



# # # # Step 1: Set the folder path where your DataFrames are stored

path_to_save = newpath

# # Step 2: List all the files in the folder
# files = [
#     f for f in os.listdir(newpath) if f.endswith("corrected")
# ]  # Adjust this to your file format

# # Step 3: Read all DataFrames and add the filename as a new column
# dataframes = []
# for f in files:
#     df = pd.read_csv(os.path.join(newpath, f))
#     # df["quadrant"] = f[-8]  # Add the filename as a new column
#     dataframes.append(df)

# # Step 4: Concatenate all DataFrames into one
# concatenated_df = pd.concat(dataframes, ignore_index=True)


# concatenated_df.to_csv(newpath+"df_all_corrected")
                                          
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




# def add_snr_integrated(df, half_aperture=2, tracking=True, wid=10):
#     """
#     returns the dataframe with new columns for outputs of
#     snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
#     """
#     column_names = ['snr_local', 'snr_global', 'noise_local', 'noise_global', 
#                 'local_average_outside', 'height', 'FWHM', 'signal_local']

#     for col in column_names:
#         df[col] = np.zeros(len(df))
    
#     for index in df.index:
#         # print(index)
#         tot = get_tot_csv(df, index)
#         row = int(df.loc[index]["row_from_nn"]) + wid
#         snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
#         df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
#         if index%10000==0:
#             print(index)
#     return df




def add_snr_integrated_new(df, half_aperture=2, tracking=True, wid=10):
    """
    returns the dataframe with new columns for outputs of
    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
    """
    column_names = ['snr_local_new', 'snr_global_new', 'noise_local_new', 'noise_global_new', 
                'local_average_outside_new', 'height_new', 'FWHM_new', 'signal_local_new']

    for col in column_names:
        df[col] = np.zeros(len(df))
    
    for index in df.index:
        # print(index)
        tot = get_tot_csv(df, index)
        row = int(df.loc[index]["row_from_nn"]) + wid
        if half_aperture==2:
            snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
            df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
        elif half_aperture==3:
            snr_local_hap3, snr_global_hap3, noise_local_hap3, noise_global_hap3, local_average_outside_hap3, height_hap3, FWHM_hap3, signal_local_hap3 = rff.get_snr_integrated(tot, row)
            df.loc[index, column_names] = [snr_local_hap3, snr_global_hap3, noise_local_hap3, noise_global_hap3, local_average_outside_hap3, height_hap3, FWHM_hap3, signal_local_hap3]
        elif half_aperture==4:
            snr_local_hap4, snr_global_hap4, noise_local_hap4, noise_global_hap4, local_average_outside_hap4, height_hap4, FWHM_hap4, signal_local_hap4 = rff.get_snr_integrated(tot, row)
            df.loc[index, column_names] = [snr_local_hap4, snr_global_hap4, noise_local_hap4, noise_global_hap4, local_average_outside_hap4, height_hap4, FWHM_hap4, signal_local_hap4]
        if index%10000==0:
            print(index)
    return df



def has_two_streaks(df, index, fpath, extension, prominence=15, return_properties=False):
    '''
    will return true if it has two streaks, otherwise false
    '''
    fname = df.loc[index]["filename"]+extension
    row = int(df.loc[index]["row_from_nn"])+10
    col = int(df.loc[index]["col_from_nn"])+10
    quad = int(df.loc[index]["quad"])
    # if direcname == "0820":
    # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
    # else:
        # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"/"
    path = fpath+fname
    # print(path)
    if extension=="fits.fz":
        hdul = fits.open(path)
        file1_array = hdul[quad].data
    # file1_array = np.load(path)

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
        tot = get_tot_csv(df, index)
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


def get_tot_csv(
    df,
    index, lc_filename=None,
    half_aperture=2,
    fpath="/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/",
):
    """
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    put in lc_filename yourself or it will take it from the dataframe 
    (in which case df and index are ignored)
    """
    # Handle filename logic 
    if lc_filename == None:
        lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
        df_row = df.loc[index]
        lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quad'])+'_'+f'{int(df_row["closest_x"]):04}'+'_'+f'{int(df_row["closest_y"]):04}'+'.csv'
    

    lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'  
    lc_df = pd.read_csv(lightcurve_path + lc_filename)
    
    # Calculate flux and error
    tot = lc_df['flux_hap_'+str(half_aperture)]
    

    return tot

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
        quad = int(df.loc[index]["quad"])
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
    # file1_array = np.load(path)
    if extension=="fits.fz":
        hdul = fits.open(path)
        file1_array = hdul[quad].data
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



# df_reducemax= pd.read_csv(newpath+"df_all")



fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+"1020_minus_extras"+"/"

# In[38]:


# df_sobel = add_sobel_ratio(fpath, df_reducemax, extension="fits.fz")

# df_sobel.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_sobel")

# df_sobel = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_sobel")



# df_snr_neg = df_sobel_neg[df_sobel_neg["sobel_ratio"]>5]


# # In[41]:


# df_snr_neg = add_snr(df_snr_neg, half_aperture=2, tracking=True, fpath=fpath, extension="_1_implanted_neg.npy")


# # In[45]:


# df_snr_neg.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_neg")





# df_snrcut_neg = df_snr_neg[np.abs(df_snr_neg["snr_center"])<25]




# df_snrall = add_snr_integrated(df_sobel)

# df_snrall = add_snr_integrated(concatenated_df)

# df_snrall.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snrall_corrected")


 


# df_wstreaks = add_twostreaks(df_snrall, extension="fits.fz", fpath=fpath, name="has_two_streaks",prominence=15)


# df_wstreaks.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_wstreaks")


# df_onestreak = df_wstreaks[df_wstreaks["has_two_streaks"]==False]


# df_onestreak.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_onestreak")


# df_snrcut = df_onestreak[df_onestreak["snr_local"]>1]

# df_snrcut.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snrcut")
    
    
# df_snrall_corrected = pd.read_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snrall_corrected")

# df_snrall_corrected_new = add_snr_integrated_new(df_snrall_corrected)

# df_snrall_corrected_new.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snrall_corrected_new")

# df_snrall_corrected_new=pd.read_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snrall_corrected_new")

# df_snr_3 = add_snr_integrated_new(df_snrall_corrected_new, half_aperture=3)
# df_snr_3.to_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_3")

df_snr_3=pd.read_csv(
    "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_3")

df_snr_4 = add_snr_integrated_new(df_snr_3, half_aperture=4)


df_snr_4.to_csv(
    "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_4")
