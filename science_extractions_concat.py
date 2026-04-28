#!/usr/bin/env python
# coding: utf-8

# In[21]:
'''
concatenate dataframes from all the predictions
'''

import numpy as np
import pandas as pd
import os
import review_findings_functions as rff
from astropy.io import fits
import matplotlib.pyplot as plt
import sys


# Step 1: Set the folder path where your DataFrames are stored
folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"

#name to save (saves in same path)
save_name = "concat"

extension = "_dedup_snr"#"matched"


# # Step 1: Set the folder path where your DataFrames are stored
# folder_path = "/lustre/lrspec/ZTF_data/science_extractions/" + save_direc + "/"

# #name to save (saves in same path)
# save_name = "intermediate_df_flipped"

# In[26]:
concat_list = []

files = [f for f in os.listdir(folder_path) if f.endswith(extension)]  # Adjust this to your file format
for file in files:
    df = pd.read_csv(os.path.join(folder_path, file))
    concat_list.append(df)

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


