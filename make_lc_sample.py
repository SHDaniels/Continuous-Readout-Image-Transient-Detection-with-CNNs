#i want to copy specific light curves into a zip file
#i bet i can make a list and then zip it
#tar -cvf allfiles.tar -T mylist.txt
#so i need to make a list

from astropy.io import fits 
import os
import numpy as np

lcpath = '/lustre/lrspec/ZTF_data/lightcurves/1020/'

file_starts = [ #we want 1/3 in each field, technically twice as much i as g
    'ztf_20221020138241_000686_zi_c01_x_1.fits',
    'ztf_20221020141806_000686_zi_c02_x_2.fits',
    'ztf_20221020148032_000686_zg_c03_x_3.fits',
    'ztf_20221020138241_000686_zi_c10_x_4.fits',
    'ztf_20221020141806_000686_zi_c11_x_1.fits',
    'ztf_20221020148032_000686_zg_c12_x_2.fits',
    
    'ztf_20221020154120_000640_zi_c04_x_3.fits',
    'ztf_20221020157697_000640_zi_c05_x_4.fits',
    'ztf_20221020163819_000640_zg_c06_x_1.fits',
    'ztf_20221020154120_000640_zi_c13_x_2.fits',
    'ztf_20221020157697_000640_zi_c14_x_3.fits',
    'ztf_20221020163819_000640_zg_c15_x_4.fits',
    
    'ztf_20221020294271_000449_zr_c07_x_1.fits',
    'ztf_20221020297847_000449_zr_c08_x_2.fits',
    'ztf_20221020294271_000449_zr_c09_x_3.fits',
    'ztf_20221020297847_000449_zr_c16_x_4.fits',
    'ztf_20221020294271_000449_zr_c01_x_2.fits',
    'ztf_20221020297847_000449_zr_c02_x_3.fits',
]

def get_filenames_of_lc_file(fits_file):
    filename_list = []
    quad = fits_file[-6]
    fname = fits_file[:-6]
    with fits.open('/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/'+fits_file) as hdul:
        if len(hdul[1].data)>25: #only take top 25
            data = hdul[1].data[:25]
        else:
            data = hdul[1].data
        for pair in data:
            X = pair[0] 
            Y = pair[1]
            # print(X,Y)
            
            num = np.random.randint(0,29)
            # print(f'{int(num):04}') 
            # print(f'{int(X):04}')
            lc_filename = 'lightcurve_'+fname+f'{int(num):04}'+'_'+str(quad)+'_'+f'{int(X):04}'+'_'+f'{int(Y):04}'+'.csv'
            #check if exists? or no
            if not os.path.isfile(lcpath+lc_filename):
                num = 0
                lc_filename = 'lightcurve_'+fname+f'{int(num):04}'+'_'+str(quad)+'_'+f'{int(X):04}'+'_'+f'{int(Y):04}'+'.csv'

            filename_list.append(lc_filename)
    return filename_list


def add_to_list(filename):
    with open('filenames_lcs.txt', 'a') as file:
        file.write('\n')
        file.write(filename)
    return



for fits_file in file_starts:
    filename_list = get_filenames_of_lc_file(fits_file)
    for filename in filename_list:
        add_to_list(filename)

    