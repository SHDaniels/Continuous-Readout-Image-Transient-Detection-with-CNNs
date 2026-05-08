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







folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"

# df_detections = pd.DataFrame([])
counter = 0
# # List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('matched')]  # Adjust this to your file format
for file in files:

    if os.path.exists(folder_path+file+"_diffcut5_size1_dedup_snr"): 
            continue  
    print(file)
    try:
        df = pd.read_csv(os.path.join(folder_path, file))
    except pandas.errors.EmptyDataError:
        print("empty data error from "+file)
        continue
    if len(df) == 0:
        print(f"Skipping {file}: empty DataFrame")
        continue
    
    df_diffcut = df[df['Diff']<5]
    if len(df_diffcut) == 0:
        print(f"Skipping {file}: no rows after Diff cut")
        continue
    df_diffcut.to_csv(folder_path+file+"_diffcut5")
    
    df_size1 = df_diffcut[df_diffcut['detection_size']>1]
    if len(df_size1) == 0:
        print(f"Skipping {file}: no rows after size cut")
        continue
    df_size1.to_csv(folder_path+file+"_diffcut5_size1")
    
    df_dedup = rff.make_aggregated_transient_detections_df(df_size1)
    if len(df_dedup) == 0:
        print(f"Skipping {file}: no rows after deduplication")
        continue
    df_dedup.to_csv(folder_path+file+"_diffcut5_size1_dedup")
    try:
        df_snr = rff.add_snr_both(df_dedup)
        if len(df_snr) == 0:
            print(f"Skipping {file}: no rows after SNR calculation")
            continue
        df_snr.to_csv(folder_path+file+"_diffcut5_size1_dedup_snr")
    except:
        print("no light curve file or other snr error? "+file)
        
    
