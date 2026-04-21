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
    if os.path.exists(folder_path+file+"_sobel"): #if we already have it don't remake
        continue    
    print(file)
    df = pd.read_csv(os.path.join(folder_path, file))
    if len(df)>0:
        # df_nocr = df[df["type"] != "cr"]
        # df_nosat = df_nocr[df_nocr["type"] != "sat"]
        df_sobel = rff.add_sobel_ratio_fast(df, direcname='1020')
        df_sobel.to_csv(folder_path+file+"_sobel")
    
