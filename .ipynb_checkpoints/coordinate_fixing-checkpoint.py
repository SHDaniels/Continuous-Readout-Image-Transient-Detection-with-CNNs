#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
i have to figure out which beginning-crm images work
cause some of them are messed up somehow
and then i will run them through colab to fix the coordinates

'''


# In[1]:


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import subprocess
from astropy.wcs import WCS


# In[21]:


#matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"





#training data
training_path = '/lustre/lrspec/ZTF_data/training_aug24/implant_train'
all_data_path = '/lustre/lrspec/ZTF_data/2022/1017'

# In[17]:


training_filenames = []

files = [
    f for f in os.listdir(training_path) if f.startswith("ztf_20221017")
]  # Adjust this to your file format
for file in files:
    image_filename = os.fsdecode(file)
    image_filename_start = image_filename[:-13]
    if len(training_filenames)>0:
        if image_filename_start in training_filenames:
            continue
        else:
            training_filenames.append(image_filename_start)
    else:
        training_filenames.append(image_filename_start)


# In[18]:


training_filenames


# In[ ]:


'''
to copy and paste:
scp shard@darwin.hpc.udel.edu:/lustre/lrspec/ZTF_data/2022/1020/ztf_20221020138241_000686_zi_c04_x.fits.fz Downloads/

'''


# In[23]:


for name in training_filenames:
    file = all_data_path+'/'+name+'.fits.fz'
    print(file)




#we can calculate the xylists here probs


# In[38]:


'''
Now we're going to calculate where the beginnings of the brightest streaks are.
To do this, we take the rolling mean along each column
and see when it changes by >70 over a different of ~100 pixels
(both of these parameters can be adjusted, but these ones work for me)

and it keeps the position of the greatest change
note that pixels separated by <8 are considered to be part of the same streak
note too that in the case of overlapping streaks, only the brighter one is retained
(this is because the inherent variability of some streaks is quite high,
and allowing more than one position along each streak meant that there were
too many which were clearly not from new stars)

this cell will take a while to run
'''
def get_diffs(array, quad):
    diffs = []
    for x_coord in range(np.shape(array)[1]):
        if quad==1 or quad==2: #moving from top to bottom if we have quad 1 or 2
            columns = range(0, np.shape(array)[0]-220)
        elif quad==3 or quad==4: #moving from bottom to top if we have quad 3 or 4
            columns = range(np.shape(array)[0]-220, 0, -1)
        for i in columns:
            mean_1 = np.mean(array[i:i+100,x_coord])
            mean_2 = np.mean(array[i+120:i+220,x_coord])
            if quad==1 or quad==2:
                diff = mean_2-mean_1
            else:
                diff = mean_1-mean_2
            if diff>30: #diff>100 did not give us enough stars for astrometry.net to be confident
                if len(diffs)>0:
                    last_x_coord = diffs[-1][0]
                    last_y_coord = diffs[-1][1]
                    last_diff = diffs[-1][2]
                    if np.abs(x_coord-last_x_coord) < 8:
                        # if np.abs((i+40)-last_y_coord) > 100 and diff>last_diff: #new star starting partway
                        #     diffs.append(np.array((x_coord,i+100, diff)))
                        # else:
                        if last_diff < diff:
                            diffs[-1]=np.array((x_coord,i+100, diff)) #replace if this is just a brighter version of the last one
                        else: #if the last one was brighter and they're the same, just ignore this one
                            continue
                    else:
                        diffs.append(np.array((x_coord,i+100, diff)))
                else:
                    diffs.append(np.array((x_coord,i+100, diff)))
            continue
    return diffs




#we have to convert this into a fits bintable file

def write_xylist(diffs, output_file):
    """
    Writes an XY list file from a given star catalog.
    """
    diffs = np.array(diffs)
    sorted_arr = diffs[diffs[:, 2].argsort()[::-1]]

    # Create FITS columns
    col1 = fits.Column(name='X', format='D', array=sorted_arr[:, 0])
    col2 = fits.Column(name='Y', format='D', array=sorted_arr[:, 1])

    # Create a FITS BINTABLE
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)

  # Save to a FITS file
    hdu.writeto(output_file+'.fits', overwrite=True)
    print(f"XY list saved to {output_file}.fits")


# In[33]:


# #here is where we do all the stuff




for name in training_filenames:
    file = all_data_path+'/'+name+'.fits.fz'
    print(file)
    hdul_start = fits.open(file)
    for quad in range(1,5):
        array = hdul_start[quad].data
        xy_file = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"+name+"_"+str(quad)
        # plt.imshow(np.sqrt(hdul_start[quad].data))
        diffs = get_diffs(array, quad)
        print("first few diffs:",diffs[:10])
        write_xylist(diffs, xy_file)

