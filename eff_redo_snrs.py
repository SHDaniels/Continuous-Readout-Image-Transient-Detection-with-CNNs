import os
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
# import implantation_functions as imp
# import get_postage_stamps as gpsf
# from scipy.signal import savgol_filter

def match_stars_eff(df):
    closest_xs = np.zeros(len(df))
    closest_ys= np.zeros(len(df))
    Diffs = np.zeros(len(df))
    brightness_ranking = np.zeros(len(df))
    xy_path = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"
    quad = str(1)
    i=0
    for index, rrow in df.iterrows():

        xy_file = xy_path+df.loc[index]['filename'][:35]+quad+".fits"
        with fits.open(xy_file) as hdul_xy:
            col = rrow["col_from_nn"]
            target_x = col+10
            # print(xy_file)
            data = hdul_xy[1].data  
            stored_x = data["X"][:25]  # Column "X"
            stored_y = data["Y"][:25]
            closest_idx = np.argmin(np.abs(stored_x - target_x))
            
            closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
            closest_xs[i]=closest_x
            closest_ys[i]=closest_y
            Diffs[i]=stored_x[closest_idx] - target_x
            brightness_ranking[i]= closest_idx
            # print(closest_idx)
            # print(df.loc[index]['brightness_ranking'])
            i+=1 #yes there is a better way to do this
    df['closest_x'] = closest_xs
    df['closest_y']= closest_ys
    df['Diff'] = Diffs
    df['brightness_ranking'] = brightness_ranking
    return df

# def add_snr_integrated(df, extension=None,  half_aperture=2, tracking=True, wid=10):
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
#         tot, maxcols = rff.get_tot_npy(df, index, df.loc[index]["type"])
#         row = int(df.loc[index]["row_from_nn"]) + wid
#         snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
#         df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
#     return df






base_path = "/lustre/lrspec/ZTF_data/efficiency_testing/"
# df_pos = pd.read_csv(base_path+"df_detections_pos_only")
# df_neg = pd.read_csv(base_path+"df_detections_neg_only")

# both = pd.concat([df_neg,df_pos], ignore_index=True)

# noedges = rff.reduce_edges(both, 15)

# matched = match_stars_eff(noedges)

# matched['quadrant']=np.full(len(matched), 1)

# snrs = add_snr_integrated(matched[matched["Diff"]<5])


# snrs.to_csv(base_path+"df_with_snrs_corrected_again")

# def highpass_filter(data, window=101, order=3):
#     """Apply Savitzky-Golay filter for smoothing or derivative"""

#     return data - savgol_filter(data, window, order, axis=0) #was 1
     



# def add_hpf_snr(df,  half_aperture=2):
#     """
#     returns the dataframe with new columns for outputs of
#     snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
#     """
#     wid=10
#     column_names = ['snr_local_hpf', 'snr_global_hpf', 'noise_local_hpf', 'noise_global_hpf', 
#                 'local_average_outside_hpf', 'height_hpf', 'FWHM_hpf', 'signal_local_hpf']

#     for col in column_names:
#         df[col] = np.zeros(len(df))
    
#     for index in df.index:
#         # print(index)
#         tot_original, maxcols = rff.get_tot_npy(df, index, df.loc[index]["type"])
#         tot = highpass_filter(tot_original, window=101, order=3)
#         row = int(df.loc[index]["row_from_nn"]) + wid
#         snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
#         df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
#     return df

# snrs = pd.read_csv(base_path+"df_with_snrs_corrected_again")

# more = add_hpf_snr(snrs)

# more.to_csv(base_path+"df_with_hpf_snrs")

def add_snr_integrated(df, extension=None,  half_aperture=2, tracking=True, wid=10):
    """
    returns the dataframe with new columns for outputs of
    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
    """
    column_names = ['snr_local', 'snr_global', 'noise_local', 'noise_global', 
                'local_average_outside', 'height', 'FWHM', 'signal_local']
    if half_aperture==3:
        column_names = ['snr_local_3', 'snr_global_3', 'noise_local_3', 'noise_global_3', 
                'local_average_outside_3', 'height_3', 'FWHM_3', 'signal_local_3']   
    elif half_aperture==4:
        column_names = ['snr_local_4', 'snr_global_4', 'noise_local_4', 'noise_global_4', 
                'local_average_outside_4', 'height_4', 'FWHM_4', 'signal_local_4']    
    for col in column_names:
        df[col] = np.zeros(len(df))
    
    for index in df.index:
        # print(index)
        tot, maxcols = rff.get_tot_npy(df, index, df.loc[index]["type"], half_aperture=half_aperture)
        row = int(df.loc[index]["row_from_nn"]) + wid
        snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_integrated(tot, row)
        df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
    return df


# snrs = pd.read_csv(base_path+"df_with_snrs_corrected_again")

# snrs_3 = add_snr_integrated(snrs)
# snrs_3.to_csv(base_path+"df_snrs_3")

# snrs_4 = add_snr_integrated(snrs_3)
# snrs_4.to_csv(base_path+"df_snrs_4")

snrs_4 = pd.read_csv(base_path+"df_snrs_4")
w_sob = rff.add_sobel_ratio(snrs_4, eff=True)
w_sob.to_csv(base_path+"df_snr_sobel_eff")