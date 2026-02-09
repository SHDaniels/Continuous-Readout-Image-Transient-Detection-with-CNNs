#!/usr/bin/env python
# coding: utf-8

# In[21]:
'''
concatenate dataframes from all the predictions
must run with folder path and save name when you run the script!
'''

import numpy as np
import pandas as pd
import os
import review_findings_functions as rff
from astropy.io import fits
import matplotlib.pyplot as plt
import sys

eff=True

# Step 1: Set the folder path where your DataFrames are stored
folder_path = sys.argv[1]

#name to save (saves in same path)
save_name = sys.argv[2]

extension = sys.argv[3] #alllist


# # Step 1: Set the folder path where your DataFrames are stored
# folder_path = "/lustre/lrspec/ZTF_data/science_extractions/" + save_direc + "/"

# #name to save (saves in same path)
# save_name = "intermediate_df_flipped"

# In[26]:
concat_list = []

files = [f for f in os.listdir(folder_path) if f.endswith(extension)]  # Adjust this to your file format
for file in files:
    df = pd.read_csv(os.path.join(folder_path, file))
    
# #pos
    df_nocr = df[df["type"] != "cr"]
    df_noboth = df_nocr[df_nocr["type"] != "both"]
    df_reduced_edges= rff.reduce_edges(df_noboth, 15)
    
    xy_path = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"
    
    if eff==False:
        quad = file[-9]
        xy_file = xy_path+file[:-30]+quad+".fits"
    elif eff==True:
        quad = file[41]
        xy_file = xy_path+file[:35]+quad+".fits"
    
    
    
    
    
    data_list = []
    
    counter=0
    # closest_amounts_c = []
    with fits.open(xy_file) as hdul_xy:
        for i, rrow in df_reduced_edges.iterrows():
            col = rrow["col_from_nn"]
            target_x = col+10
        # print(xy_file)
            data = hdul_xy[1].data  
            stored_x = data["X"][:25]  # Column "X"
            stored_y = data["Y"][:25]
            closest_idx = np.argmin(np.abs(stored_x - target_x))
            closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
            # if np.abs(closest_x-target_x)<=10:
                # plt.imshow(np.sqrt(file1_array[(int(closest_y))-20:(int(closest_y))+20, (int(closest_x))-20:(int(closest_x))+20]))
                # plt.scatter(20,20)
                # plt.show()    
                    # Retrieve the closest (x, y) match
            
            data_list.append({"filename": rrow["filename"], "quad":quad, "type":rrow["type"],"row_from_nn":rrow["row_from_nn"],  
                              "col_from_nn":rrow["col_from_nn"], "detection_size":rrow["detection_size"],
                             "closest_x": closest_x, "closest_y":closest_y, "Diff":(closest_x-target_x), "brightness_ranking":closest_idx})
    dframe = pd.DataFrame(data_list)
            # dframe.to_csv(newpath+file+"_matched_corrected")
    concat_list.append(dframe)

concatenated_df = pd.concat(concat_list, ignore_index=True)


# In[29]:


concatenated_df['quadrant']=concatenated_df['quad']


# In[30]:

concatenated_df.to_csv(folder_path+save_name)

concatenated_df = rff.add_num_in_file_in_star(concatenated_df)

concatenated_df.to_csv(folder_path+save_name)
# In[34]:


# In[45]:


# In[4]:


# rff.match_stars_eff(concatenated_df)


