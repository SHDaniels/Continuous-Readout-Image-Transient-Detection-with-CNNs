#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
from astropy.io import fits
import os
import get_postage_stamps as gpsf
import gc
import glob
import review_findings_functions as rff
import sys

name = sys.argv[1]
cnn_index = int(sys.argv[2])  # 0-4 from SLURM_ARRAY_TASK_ID
file_task_id = int(sys.argv[3])  # NEW: which subset of files to process
total_file_tasks = int(sys.argv[4])  # NEW: total number of file tasks

suffix = f"_{cnn_index}"

path_start = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_"
save_direc = path_start + name + suffix + "/"

if not os.path.exists(save_direc):
    os.mkdir(save_direc)

data_direc = "/lustre/lrspec/ZTF_data/2022/1020/"
path_save_long_term = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"

model = keras.models.load_model(f"/home/3155/CNN_{name}{suffix}")

wid = 10
fpath_extractions_save = save_direc

# Get all files and filter them
all_files = [f for f in os.listdir(data_direc) if f.endswith('.fits.fz') and not f.endswith('x.fits.fz')]
all_files.sort()  # Ensure consistent ordering

# Divide files among tasks
files_per_task = len(all_files) // total_file_tasks + 1
start_idx = file_task_id * files_per_task
end_idx = min((file_task_id + 1) * files_per_task, len(all_files))
my_files = all_files[start_idx:end_idx]

print(f"Task {file_task_id}/{total_file_tasks}: Processing {len(my_files)} files (indices {start_idx}-{end_idx})")

for image_filename in my_files:
    filename_start = image_filename[:40]
    image_filename_full = data_direc + image_filename
    
    # Check if already exists
    if os.path.isfile(
        fpath_extractions_save + str(filename_start) + "_extraction_predsarray_4_predsarray.npy"
    ):
        print("skipping ", filename_start)
        continue
    
    if os.path.isfile(
        path_save_long_term + str(filename_start) + "_extraction_predsarray_4_predsarray.npy"
    ):
        print("skipping - exists in average", filename_start)
        continue
    
    if os.path.isfile("/lustre/lrspec/ZTF_data/training_aug24/implant_train/" + image_filename):
        print("skipping - in training data ", filename_start)
        continue
    
    print(image_filename_full)
    
    for iteration in range(1, 5):
        with fits.open(image_filename_full) as hdul:
            file1_array = hdul[iteration].data
            norm = (file1_array - np.min(file1_array)) / (
                np.max(file1_array) - np.min(file1_array)
            )
            image = norm
            path = (
                fpath_extractions_save
                + str(filename_start)
                + "_extraction_predsarray_"
                + str(iteration)
            )
            rff.save_predsarrays(image, model, wid, path)
    
    del norm
    del file1_array
    del image
    gc.collect()

print(f"Task {file_task_id} complete!")