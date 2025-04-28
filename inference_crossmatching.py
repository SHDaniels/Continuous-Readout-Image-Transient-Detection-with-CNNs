#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
getting a dataframe with wcs coordinates for the inference data
'''


# In[2]:


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import pandas as pd
from astropy.wcs import WCS
import shutil
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


# In[3]:


import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/3155/rwe')

import review_findings_functions as rff


# In[4]:


#matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


# In[33]:


fpath_header = "/lustre/lrspec/ZTF_data/corrected_coordinates/new_training_headers/"



# In[5]:


df_lessthan13 = pd.read_csv(
    "/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/df_lessthan13"
)


# In[12]:


threesig_array = (df_lessthan13['snr_max']>3) | (df_lessthan13['snr_min']<(-3))


# In[13]:


df_lessthan13_threesig = df_lessthan13[threesig_array]


# In[19]:


np.sum(df_lessthan13_threesig['type']=='both')


# In[22]:


np.sum(df_lessthan13_threesig['type']=='pos')+np.sum( df_lessthan13_threesig['type']=='neg')


# In[25]:


posneg = (df_lessthan13_threesig['type']=='pos') | (df_lessthan13_threesig['type']=='neg')


# In[26]:


df_lessthan13_threesig_posneg = df_lessthan13_threesig[posneg]


# In[27]:


df_lessthan13_threesig_posneg


# In[63]:


df = df_lessthan13_threesig_posneg


# In[69]:


(df['filename'])[:-6]


# In[75]:


for index, row in df.iterrows():
    if row['filename'][:-6]=='ztf_20221020294271_000449_zr_c09_x' and row['quadrant']==3:
        print('hi')
        #df.drop(row)


# In[44]:


#check that col is the x position: first and second possible transients 
#have same col and are same streak
rff.plot_time_series_again_aperture(df,3) 


# In[ ]:


# List to store dictionaries for DataFrame creation
data_list = []

for index, row in df.iterrows():
    image_filename = row['filename']
    print(image_filename[:-6])
    quad = row['quadrant']
    
    xy_file = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/" + image_filename[:-6] + "_" + str(quad) + ".fits"
    header_file = fpath_header + image_filename[:-6] + "_" + str(quad) + ".wcs"
    print(header_file)

    try:
        with fits.open(xy_file) as hdul_xy:
            data = hdul_xy[1].data  
            stored_x = data["X"]
            stored_y = data["Y"]
    except Exception as e:
        print(f"Could not open xy_file: {xy_file} — {e}")
        stored_x = stored_y = None

    try:
        with fits.open(header_file) as hdul_header:
            wcs = WCS(hdul_header[0].header, relax=True)
    except Exception as e:
        print(f"Could not open header_file: {header_file} — {e}")
        wcs = None

    if stored_x is not None and stored_y is not None and wcs is not None:
        target_x = row['col_from_nn'] + 10
        closest_idx = np.argmin(np.abs(stored_x - target_x))
        closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
        ra, dec = wcs.all_pix2world(closest_x, closest_y, True)
    else:
        closest_x = closest_y = ra = dec = np.nan

    diff = np.abs(closest_x - (row['col_from_nn'] + 10)) if not np.isnan(closest_x) else np.nan
    data_list.append({**dict(enumerate(row)), 
                      "closest_x": closest_x, 
                      "closest_y": closest_y, 
                      "Diff": diff, 
                      "star_ra": ra, 
                      "star_dec": dec})


# In[61]:


data_list


# In[33]:





# In[ ]:


# Create DataFrame
df = pd.DataFrame(data_list)

# Display DataFrame structure
df.head()


# In[53]:


df_lessthan13_threesig_posneg.head()


# In[ ]:


df.rename(columns={5: 'filename', 
                   6: 'type',
                  7: 'row_from_nn', 
                   8: 'col_from_nn',
                  9:'detection_size',
                  10:'quadrant',
                  11:'sobel_ratio',
                  12:'snr_center',
                  13:'snr_max',
                  14:'snr_min'}, inplace=True)


# In[ ]:


df


# In[38]:


df.to_csv("/home/3155/inference_star_positions")

