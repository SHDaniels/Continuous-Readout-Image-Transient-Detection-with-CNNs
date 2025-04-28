#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
this notebook will get info about all of the stars included in 
the CNN training data
'''


# In[41]:


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




# df = pd.read_csv("/home/3155/training_star_positions")
df = pd.read_csv("/home/3155/inference_star_positions")
band = []
for i in range(len(df)):
    band.append(df.loc[i]['filename'][27])
df["Band"] = band



def query_gaia_for_stars(df, radius=12):
    """
    Query Gaia DR3 to match stars in the given DataFrame based on RA and Dec.
    Adds Gaia source ID, magnitude (G, BP-RP, RP, BP), and variability information.
    Additionally checks if the star is listed in the Gaia DR3 vari_summary table.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'ra' and 'dec' columns.
        radius (float): Search radius in arcseconds.
        
    Returns:
        pd.DataFrame: Updated DataFrame with Gaia data.
    """
    gaia_columns = [
        'source_id', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'phot_rp_mean_mag', 'phot_bp_mean_mag', 'phot_variable_flag'
    ]
    
    
    original_data = np.load('/home/3155/training_star_info_partial.npy',allow_pickle=True)
    new_data = []
    
    for i, row in df.iterrows():
        if i>=len(original_data):
            if pd.isna(row['star_ra']) or pd.isna(row['star_dec']):
                print("NaN coordinate, skipping Gaia query for this row.")
                new_data.append([None, None, None, None, None, None, None])
                continue
            
            coord = SkyCoord(ra=row['star_ra']*u.deg, dec=row['star_dec']*u.deg, frame='icrs')
            
            if row["Band"]=="g":
                query = f"""
                SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
                WHERE CONTAINS(POINT('ICRS', ra, dec), 
                           CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
                ORDER BY phot_bp_mean_mag ASC 
                """
            elif row["Band"]=="i":
                query = f"""
                SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
                WHERE CONTAINS(POINT('ICRS', ra, dec), 
                           CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
                ORDER BY phot_rp_mean_mag ASC
                """
            elif row["Band"]=="r":
                query = f"""
                SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
                WHERE CONTAINS(POINT('ICRS', ra, dec), 
                           CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
                ORDER BY phot_rp_mean_mag ASC
                """
            else:
                print("band not recognized error: ", row["Band"])
            print("launching job")
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            print("got results")
            if len(results) > 0:
                best_match = results[0]
                source_id = best_match['source_id']
                
                # Check if the star is in the vari_summary table
                var_query = f"""
                SELECT COUNT(*) AS var_count FROM gaiadr3.vari_summary
                WHERE source_id = {source_id}
                """
                var_job = Gaia.launch_job(var_query)
                var_results = var_job.get_results()
                is_variable = var_results['var_count'][0] > 0
                
                new_data.append([
                    source_id, best_match['phot_g_mean_mag'],
                    best_match['bp_rp'], best_match['phot_rp_mean_mag'],
                    best_match['phot_bp_mean_mag'], best_match['phot_variable_flag'],
                    is_variable
                ])
            else:
                new_data.append([None, None, None, None, None, None, None])
            np.save("/home/3155/training_star_info_partial_2", new_data)
        all_data = np.vstack((original_data,new_data))
        df[['gaia_source_id', 'g_mag', 'bp_rp', 'rp_mag_ztf_ri', 'bp_mag_ztf_g', 'variability', 'in_vari_summary']] = all_data
    # df.to_csv("/home/3155/training_star_info_partial")
    return df


# In[47]:





# In[56]:


df = query_gaia_for_stars(df)
# df.to_csv("/home/3155/training_star_info_r12_bands")
df.to_csv("/home/3155/inference_star_info_r12_bands_2")


