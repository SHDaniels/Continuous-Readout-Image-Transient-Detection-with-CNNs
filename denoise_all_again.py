'''
run pca denoising on all our LCs
'''
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.fft import fft, ifft, fftfreq

predictions = pd.read_csv('/lustre/lrspec/ZTF_data/random_forest_training/predictions_w_fname.csv')

def apply_fft_lowpass_series(s, fs=1, fc=0.0003):
    """
    s: pandas Series (single light curve), index = time points / columns
    Returns a Series with the same index/name.
    """
    data = s.values
    n = len(data)
    fft_vals = fft(data)
    freqs = fftfreq(n, d=1)
    mask = np.abs(freqs) <= fc
    fft_vals_filtered = fft_vals * mask
    filtered = np.real(ifft(fft_vals_filtered))
    return pd.Series(filtered, index=s.index, name=s.name)

original_direc = '/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat/'
new_direc = '/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat_denoised_rf_again/'
all_filestarts = np.load('/home/3155/all_filestarts.npy')

for filestart in all_filestarts:
    filelist = [file for file in os.listdir(original_direc) if file.startswith(filestart)]

    lc_dict = {
        file: np.load(os.path.join(original_direc, file))
        for file in filelist
        if np.sum(predictions[predictions['filename_new'] == file]['binary'].values) < 1
    }

    # rows = filenames, cols = time points; index carries filenames explicitly
    df = pd.DataFrame.from_dict(lc_dict, orient='index')

    print(filestart, flush=True)
    print('shape df: ', df.shape, flush=True)

    if df.empty:
        print('  no non-binary files found, skipping', flush=True)
        continue

    # sanity check: no unexpected NaNs from unequal-length light curves
    n_nan_rows = df.isna().any(axis=1).sum()
    if n_nan_rows > 0:
        print(f'  WARNING: {n_nan_rows} rows contain NaNs (unequal-length curves?)', flush=True)

    # --- lowpass filter, row by row, staying in DataFrame form ---
    # df.apply(axis=1) passes each row as a Series (index=time cols, name=filename)
    # and returns a Series per row, so the filename label travels with the data.
    df_lowpass = df.apply(apply_fft_lowpass_series, axis=1, fs=1, fc=0.0003)
    df_lowpass.index = df.index  # explicit, but should already match
    df_lowpass.columns = df.columns

    assert df_lowpass.index.equals(df.index), "lowpass output index mismatch!"

    # --- PCA on lowpassed data ---
    # rows=filenames, cols=time -> PCA expects samples as rows, so this matches (X = files x time)
    X_df = df_lowpass
    X_centered_df = X_df - X_df.mean(axis=0)

    Xdo = X_centered_df.values  # PCA needs a bare array, but we keep the index/columns alongside

    pca = PCA()
    pca.fit(Xdo)

    k = 10
    pca_noise = PCA(n_components=k)
    pca_noise.fit(Xdo)
    common_noise = pca_noise.inverse_transform(pca_noise.transform(Xdo))

    # rebuild common_noise as a DataFrame with the SAME index/columns as X_centered_df
    common_noise_df = pd.DataFrame(
        common_noise, index=X_centered_df.index, columns=X_centered_df.columns
    )

    # subtract using DataFrame arithmetic -> pandas aligns on index/columns automatically
    # and would raise/produce NaNs (visibly) rather than silently misaligning positionally
    df_denoised = df - common_noise_df

    assert df_denoised.index.equals(df.index), "denoised output index mismatch!"
    print('shape df_denoised: ', df_denoised.shape, flush=True)

    # df_denoised.to_csv(f'/lustre/lrspec/ZTF_data/df_denoised_temp_{filestart}.csv')

    os.makedirs(new_direc, exist_ok=True)
    for fname, row in df_denoised.iterrows():
        # namespace with filestart to avoid collisions across groups after truncating '.npy'
        out_name = f'{filestart}_{fname[:-4]}_denoised'
        row.to_csv(os.path.join(new_direc, out_name), header=False, index=False)