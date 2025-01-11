import numpy as np
import pandas as pd
import os


#let's count how many pixels are classified as what
# total_num_pos = 0
# total_num_cr = 0
# total_num_streak = 0
# total_num_bg = 0
# total_num_neg = 0
# total_num_both = 0
# total_num_leftover = 0

# wid = 10
# abst_thr = .9999
# folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020"
# files = [f for f in os.listdir(folder_path) if f.endswith('extraction_predsarray_1_predsarray.npy') or f.endswith('extraction_predsarray_2_predsarray.npy')or f.endswith('extraction_predsarray_3_predsarray.npy') or f.endswith('extraction_predsarray_4_predsarray.npy')]  # Adjust this to your file format

    
    
# for file in files:
#     image_filename = os.fsdecode(file)
#     filename_start = image_filename[:40]
#     image_filename_full = folder_path+'/'+image_filename
#     print(image_filename_full)
#     if image_filename_full=='/lustre/lrspec/ZTF_data/science_extractions/1020/ztf_20221020157697_000640_zi_c01_x_0010._extraction_predsarray_4_predsarray.npy':
#         continue
#     predsarray = np.load(image_filename_full)
#     pimg_pos = predsarray[:,1]
#     pimg_bg = predsarray[:,0]
#     pimg_streak = predsarray[:,4]
#     pimg_neg = predsarray[:,2]
#     pimg_cr = predsarray[:,3]

#     pimg_both = pimg_pos+pimg_neg
    
#     num_pos = np.sum(pimg_pos > abst_thr)
#     num_cr = np.sum(pimg_cr > abst_thr)
#     num_streak = np.sum(pimg_streak > abst_thr)
#     num_neg = np.sum(pimg_neg > abst_thr)
#     num_bg = np.sum(pimg_bg > abst_thr)
#     num_both = np.sum(pimg_both > abst_thr)
#     num_leftover = len(pimg_pos)-num_both-num_bg-num_streak-num_cr

#     total_num_pos +=num_pos
#     total_num_cr +=num_cr
#     total_num_streak +=num_streak
#     total_num_bg +=num_bg
#     total_num_neg +=num_neg
#     total_num_both +=num_both
#     total_num_leftover +=num_leftover

#     del pimg_cr
#     del pimg_pos
#     del pimg_neg
#     del pimg_bg
#     del pimg_streak
#     del predsarray
#     gc.collect()
    
# np.save('/lustre/lrspec/ZTF_data/science_extractions/1020/numbers_predsarrays.npy',[total_num_pos,total_num_cr,total_num_streak,total_num_bg,total_num_neg,total_num_both,total_num_leftover])
        
        
folder_path = '/lustre/lrspec/ZTF_data/science_extractions/1020'

# #let's count how many pixels are classified as what
# total_num_pos2 = 0
# total_num_cr2 = 0
# total_num_both2 = 0
# total_num_neg2 = 0

# # List all the files in the folder
# files = [f for f in os.listdir(folder_path) if f.endswith('alllist')]  # Adjust this to your file format
# for file in files:
#     df = pd.read_csv(os.path.join(folder_path, file))
#     num_cr = len(df[df['type'] == 'cr'])
#     num_pos = len(df[df['type'] == 'pos'])
#     num_neg = len(df[df['type'] == 'neg'])
#     num_both = len(df[df['type'] == 'both'])
    
#     total_num_pos2 +=num_pos
#     total_num_cr2 +=num_cr
#     total_num_neg2 +=num_neg
#     total_num_both2 +=num_both
    
    
# np.save('/lustre/lrspec/ZTF_data/science_extractions/1020/numbers_alllist.npy',[total_num_pos2,total_num_cr2,total_num_neg2,total_num_both2])

# files = [f for f in os.listdir(folder_path) if f.endswith('alllist_reduced')]  # Adjust this to your file format

# total_num_pos3 = 0
# total_num_both3 = 0
# total_num_neg3 = 0

# # Read and process each DataFrame file
# for f in files:

#     df = pd.read_csv(os.path.join(folder_path, f))
#     num_pos = len(df[df['type'] == 'pos'])
#     num_neg = len(df[df['type'] == 'neg'])
#     num_both = len(df[df['type'] == 'both'])
    
#     total_num_pos3 +=num_pos
#     total_num_neg3 +=num_neg
#     total_num_both3 +=num_both
    
# np.save('/lustre/lrspec/ZTF_data/science_extractions/1020/numbers_reduced.npy',[total_num_pos3,total_num_neg3,total_num_both3])

files = [f for f in os.listdir(folder_path) if f.endswith('_snr')]  # Adjust this to your file format

total_num_pos4 = 0
total_num_both4 = 0
total_num_neg4 = 0

# Read and process each DataFrame file
for f in files:

    df = pd.read_csv(os.path.join(folder_path, f))
    num_pos = len(df[df['type'] == 'pos'])
    num_neg = len(df[df['type'] == 'neg'])
    num_both = len(df[df['type'] == 'both'])
    
    total_num_pos4 +=num_pos
    total_num_neg4 +=num_neg
    total_num_both4 +=num_both
    
np.save('/lustre/lrspec/ZTF_data/science_extractions/1020/numbers_reducedsobelwsnr.npy',[total_num_pos4,total_num_neg4,total_num_both4])