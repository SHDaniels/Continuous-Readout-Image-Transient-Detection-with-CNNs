'''
run pca denoising on all our LCS
'''

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.fft import fft, ifft, fftfreq

predictions = pd.read_csv('/lustre/lrspec/ZTF_data/random_forest_training/predictions_w_fname.csv')



def apply_fft_lowpass(data, fs=1, fc=0.0003):
    """
    data: 1D numpy array (single sensor)
    fs: sampling frequency
    fc: cutoff frequency
    """
    n = len(data)
    fft_vals = fft(data)
    freqs = fftfreq(n, d=1)
    
    # Create a mask: keep frequencies where |freq| <= fc
    mask = np.abs(freqs) <= fc
    
    # Apply mask (zero out high frequencies)
    fft_vals_filtered = fft_vals * mask
    
    # Transform back to time domain (real part)
    return np.real(ifft(fft_vals_filtered))


original_direc = '/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat/'
new_direc = '/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat_denoised_rf/'

all_filestarts = np.load('/home/3155/all_filestarts.npy')
#need to do it per-chip
for filestart in all_filestarts:
    filelist = [file for file in os.listdir(original_direc) if file.startswith(filestart)]
    lc_dict = {
        file: np.load(os.path.join(original_direc, file))
        for file in filelist if np.sum(predictions[predictions['filename_new']==file]['binary'].values)<1
    }
    df = pd.DataFrame.from_dict(lc_dict, orient='index')  # rows=filenames, cols=time points
    
    print(filestart, flush=True)
    print('shape df: ', np.shape(df), flush=True)
        
    data_matrix = df
    df_lowpass = pd.DataFrame(
            [apply_fft_lowpass(data_matrix.iloc[i].values, 1, fc=0.0003) for i in range(data_matrix.T.shape[1])]
        )
    
    # print('shape df_lowpass: ', np.shape(df_lowpass))
    
    X = df_lowpass.values.T
    X_centered = X - np.mean(X, axis=0)
    Xdo = X_centered
    
    # print('shape Xdo: ', np.shape(Xdo))
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_centered)
    #IF YOU WANT TO SCALE
    #Xdo = X_scaled
    
    pca = PCA()
    pca.fit(Xdo)
    
    k = 10
    pca_noise = PCA(n_components=k)
    pca_noise.fit(Xdo)
    # the common noise is the 20-components reconstructions of the lowpass data
    common_noise = pca_noise.inverse_transform(pca_noise.transform(Xdo))
    
    X_denoised = data_matrix.T - common_noise
    
    # print('shape X_denoised: ', np.shape(X_denoised))
    
    df_denoised = pd.DataFrame(X_denoised.T, index=df.index, columns=df.columns)
    df_denoised.to_csv('/lustre/lrspec/ZTF_data/df_denoised_temp')
    
    for i, row in df_denoised.iterrows():
        row.to_csv(os.path.join(new_direc, i[:-4]+'_denoised'), header=False, index=False)