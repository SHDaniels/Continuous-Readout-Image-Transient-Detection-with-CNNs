'''i am gonna do cut at diff <5, then cut on size 1, 
then de-duplicate in a different script
...well maybe in this script actually
and i can start with matched
and sobel we shall ignore bc need to recalculate'''

import review_findings_functions as rff
import os
import numpy as np
import pandas as pd
import gc
import sys




def add_snr_both(df):
    """
    returns the dataframe with new columns for outputs of
    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM 
    currently hardcoded half ap = 2
    """
    column_names = ['snr_local_new', 'snr_global_new', 'noise_local_new', 'noise_global_new', 
                'local_average_outside_new', 'height_new', 'FWHM_new', 'signal_local_new']

    for col in column_names:
        df[col] = np.zeros(len(df))
    
    for closest_x, group in df.groupby('closest_x'):  #
        # print(index)
        tot = rff.get_tot_csv(df, group.index[0]) #will be the same for all so just using the first
        for index in group.index:
            row = int(df.loc[index]["row_from_nn"]) + wid
           
            snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local = rff.get_snr_both(tot, row)
            df.loc[index, column_names] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local]
            if index%10000==0:
            print(index)
    return df





folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"

# df_detections = pd.DataFrame([])
counter = 0
# # List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('matched')]  # Adjust this to your file format
for file in files:
    # if os.path.exists(folder_path+file+"_sobel"): #if we already have it in sobel, use that
    #     newfile = file+"_sobel"
    #     sobel_flag = True
    # else:
    #     newfile = file
    #     sobel_flag = False
    if os.path.exists(folder_path+file+"_diffcut5_size1_dedup_sobel_snr"): 
            continue  
    print(file)
    df = pd.read_csv(os.path.join(folder_path, file))
    if len(df)>0:
        df_diffcut = df[df['Diff']<5]
        df_diffcut.to_csv(folder_path+file+"_diffcut5")
        df_size1 = df_diffcut[df_diffcut['detection_size']>1]
        df_size1.to_csv(folder_path+file+"_diffcut5_size1")
        df_dedup = rff.make_aggregated_transient_detections_df(df_size1)
        df_dedup.to_csv(folder_path+file+"_diffcut5_size1_dedup")
        df_sobel = rff.add_sobel_ratio_fast(df_dedup, direcname='1020')
        df_sobel.to_csv(folder_path+file+"_diffcut5_size1_dedup_sobel")
        df_snr = add_snr_both(df_sobel)
        df_snr.to_csv(folder_path+file+"_diffcut5_size1_dedup_sobel_snr")

        
    
