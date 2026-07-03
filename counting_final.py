#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
figure out how many transient candidates get cut at each stage
'''


# In[1]:


import numpy as np
import pandas as pd
# from astropy.io import fits
# import matplotlib.pyplot as plt
# import precision_recall as pr
import gc  # garbage collector
import review_findings_functions as rff
import os
# import importlib
# import implantation_functions as imp
# import corner
# plt.rcParams["figure.figsize"] = (10, 7)
# plt.rcParams["image.interpolation"] = "nearest"
# plt.rcParams["image.cmap"] = "gist_gray"


# # In[13]:


# #let's count the number of pixels >.99
# folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"
# pixels_count = 0
# transients_count = 0
# # # List all the files in the folder
# files = [f for f in os.listdir(folder_path) if f.endswith('alllist')] 

# for file in files:
#     it = pd.read_csv(folder_path+file)
#     pixels_count+= np.sum(it['detection_size'])
#     transients_count+=len(it)
#     del it
#     gc.collect()
    

# print("pixels, transients ", pixels_count, transients_count )                        


# # In[14]:





# # In[16]:





# In[24]:


#now let's count the diffcut but only the good ones
#we can do size 1 here as well
predictions = pd.read_csv('/lustre/lrspec/ZTF_data/random_forest_training/predictions_updated.csv', names=['filename', 'rf'])

folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"
transients_count_diffcut = 0
transients_count_diffcut_nosize1 = 0
# # List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('diffcut5')] 

for file in files:
    #figure out whether it's in the random forest
    df = pd.read_csv(folder_path+file)
    for filename, group in df.groupby('filename'):     
        for quadnum, quadgroup in group.groupby('quad'):
            for starx, stargroup in quadgroup.groupby('closest_x'):
                lc_filename = 'lightcurve_'+filename[:-1]+'_'+str(quadnum)+'_'+f'{int(starx):04}'+'_'+f'{int(stargroup.iloc[0]["closest_y"]):04}'+'.csv'
                # lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quadrant'])+'_'+f'{int(df_row["closest_x"]):04}'+'_'+f'{int(df_row["closest_y"]):04}'+'.csv'
                matching_row = predictions[predictions['filename']==lc_filename]
                # print(lc_filename)
                # print(matching_row['rf'].values[0])
                if len(matching_row)==1:
                    if matching_row['rf'].values[0]==0:
                        transients_count_diffcut+=len(stargroup)
                        transients_count_diffcut_nosize1+=len(stargroup[stargroup['detection_size']>1])
                elif len(matching_row)==0:
                    # concat.loc[i, 'random_forest_result']=None #actually can just leave this as a NaN
                    
                    print('didnt find it')
                    print(matching_row)
                else:
                    
                    print('something strange is happening...')
                    print(matching_row)
                    print(len(matching_row))
    
    del df
    gc.collect()
    

print("transients_count_diffcut, transients_count_diffcut_nosize1 ", transients_count_diffcut, transients_count_diffcut_nosize1)    


# In[25]:


print("transients_count_diffcut, transients_count_diffcut_nosize1 ", transients_count_diffcut, transients_count_diffcut_nosize1)    


# In[26]:


#finally count deduplicated
predictions = pd.read_csv('/lustre/lrspec/ZTF_data/random_forest_training/predictions_updated.csv', names=['filename', 'rf'])

folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"
transients_count_dedup = 0
# transients_count_diffcut_nosize1 = 0
# # List all the files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('dedup')] 

for file in files:
    #figure out whether it's in the random forest
    df = pd.read_csv(folder_path+file)
    for filename, group in df.groupby('filename'):     
        for quadnum, quadgroup in group.groupby('quad'):
            for starx, stargroup in quadgroup.groupby('closest_x'):
                lc_filename = 'lightcurve_'+filename[:-1]+'_'+str(quadnum)+'_'+f'{int(starx):04}'+'_'+f'{int(stargroup.iloc[0]["closest_y"]):04}'+'.csv'
                # lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quadrant'])+'_'+f'{int(df_row["closest_x"]):04}'+'_'+f'{int(df_row["closest_y"]):04}'+'.csv'
                matching_row = predictions[predictions['filename']==lc_filename]
                # print(lc_filename)
                # print(matching_row['rf'].values[0])
                if len(matching_row)==1:
                    if matching_row['rf'].values[0]==0:
                        transients_count_dedup+=len(stargroup)
                elif len(matching_row)==0:
                    # concat.loc[i, 'random_forest_result']=None #actually can just leave this as a NaN
                    
                    print('didnt find it')
                    print(matching_row)
                else:
                    
                    print('something strange is happening...')
                    print(matching_row)
                    print(len(matching_row))
    
    del df
    gc.collect()
    

print("transients_count_dedup", transients_count_dedup)    


# In[27]:


print("transients_count_dedup", transients_count_dedup)    


# In[29]:


np.save('/home/3155/counts.npy', np.array([pixels_count, transients_count, transients_count_diffcut, transients_count_diffcut_nosize1, transients_count_dedup]))


# In[30]:


np.load('/home/3155/counts.npy')

