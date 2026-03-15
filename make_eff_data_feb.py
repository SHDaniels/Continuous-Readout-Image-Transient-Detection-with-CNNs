import numpy as np
import pandas as pd
from astropy.io import fits
# import matplotlib.pyplot as plt



import precision_recall as pr


# importlib.reload(plot_time_series)
import gc  # garbage collector
import glob  # i don't know if there is a better way to do this but this was the

# internet's suggestion for searching with an incomplete path name
import review_findings_functions as rff
import os
import implantation_functions as imp


sat_df= pd.read_csv('/home/3155/satsofar')

num_images = 8
wid = 10
n = 50
size = num_images * n * 3   #multiply by 3 for the total considering quads

# HERES WHERE THE STUFF HAPPENS
path_to_save_pos = "/lustre/lrspec/ZTF_data/efficiency_testing/new_eff_data/full_image_implantations_pos/"
fpath_image = "/lustre/lrspec/ZTF_data/2022/1020/"
path_to_save_neg = "/lustre/lrspec/ZTF_data/efficiency_testing/new_eff_data/full_image_implantations_neg/"


seed = np.random.seed(20)  # so that i can keep the same random
# for the center of the transient - want at least half of the transient in the image
# center_values_new = np.random.randint(35, high=3035, size=size, dtype=int)
sigmas_pos = np.random.randint(3, high=10, size=size, dtype=int)
sigmas_neg = np.random.randint(3, high=10, size=size, dtype=int)
intensities_pos = imp.get_intensities_pos_feb_eff(size, seed)
intensities_neg = imp.get_intensities_neg_feb_eff(size, seed)
counter = 0

for fname in np.unique(sat_df['filename']):
    
    image_filename = fname+'fits.fz'
    filename_start = image_filename[:40]
    image_filename_full = fpath_image + image_filename
    # print(image_filename)
    hdul = fits.open(image_filename_full)
    for iteration in range(1, 5):
        if iteration in sat_df[sat_df['filename']==fname]['quad'].values: #don't use the ones with sats
            print(fname)
            print("skipping quad ", iteration)
            continue
        file1_array = hdul[iteration].data
        norm = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )
        image = norm
        # plt.imshow(image)
        nrow, ncol = image.shape
    
        for thetype in ["pos", "neg"]:
            # print(type)
            
                
            if thetype=="pos":
                sigmas = sigmas_pos
                imagepath = (
                    path_to_save_pos
                    + "images/"
                    + str(filename_start)
                    + "_"
                    + str(iteration)
                    + ".npy")
                
                infopath = (
                    path_to_save_pos
                    + "info/"
                    + str(filename_start)
                    + "_"
                    + str(iteration)
                    + "_info.npy")
                

                stamps_implanted, stamp_info, = imp.implant_50_transients(
                image, sigmas, intensities_pos, counter, norm=False
                ) 
                np.save(imagepath, stamps_implanted)
                np.save(infopath, stamp_info)
                
            elif thetype=="neg":
                sigmas = sigmas_neg
                imagepath = (
                    path_to_save_neg
                    + "images/"
                    + str(filename_start)
                    + "_"
                    + str(iteration)
                    + ".npy")
                
                infopath = (
                    path_to_save_neg
                    + "info/"
                    + str(filename_start)
                    + "_"
                    + str(iteration)
                    + "_info.npy")
                

                stamps_implanted, stamp_info = imp.implant_50_transients(
                image, sigmas, intensities_neg, counter, norm=False
                ) 
                np.save(imagepath, stamps_implanted)
                np.save(infopath, stamp_info)



        counter += n 