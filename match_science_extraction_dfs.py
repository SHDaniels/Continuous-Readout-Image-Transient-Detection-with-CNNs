import review_findings_functions as rff
import os
import numpy as np
import pandas as pd
import gc
from astropy.io import fits

folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"

# df_detections = pd.DataFrame([])
counter = 0
# # List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('alllist')]  # Adjust this to your file format
for file in files:
    if os.path.exists(folder_path+file+"_matched"): #if we already have it don't remake
        continue
    df = pd.read_csv(os.path.join(folder_path, file))
    df_nocr = df[df["type"] != "cr"]
    df_nosat = df_nocr[df_nocr["type"] != "sat"]
    
# #pos
    # df_nocr = df[df["type"] != "cr"]
    # df_noboth = df_nocr[df_nocr["type"] != "both"]
    # df_reduced_edges= rff.reduce_edges(df_noboth, 15)
    
    quad = file[-9] #41
    
    xy_path = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"
    xy_file = xy_path+file[:-15]+quad+".fits"
    
    data_list = []
    
    
    # closest_amounts_c = []
    with fits.open(xy_file) as hdul_xy:
        for i, rrow in df_nosat.iterrows():
            col = rrow["col_from_nn"]
            target_x = col+10
        # print(xy_file)
            data = hdul_xy[1].data  
            stored_x = data["X"][:25]  # Column "X"
            stored_y = data["Y"][:25]
            closest_idx = np.argmin(np.abs(stored_x - target_x))
            closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
            if np.abs(closest_x-target_x)<=10:
                # plt.imshow(np.sqrt(file1_array[(int(closest_y))-20:(int(closest_y))+20, (int(closest_x))-20:(int(closest_x))+20]))
                # plt.scatter(20,20)
                # plt.show()    
                    # Retrieve the closest (x, y) match
                data_list.append({"filename": rrow["filename"], "quad":quad, "type":rrow["type"],"row_from_nn":rrow["row_from_nn"],  
                                  "col_from_nn":rrow["col_from_nn"], "detection_size":rrow["detection_size"],
                                 "closest_x": closest_x, "closest_y":closest_y, "Diff":np.abs(closest_x-target_x)})
    dframe = pd.DataFrame(data_list)
    dframe.to_csv(folder_path+file+"_matched")
    del dframe
    del data_list
    gc.collect()
                
