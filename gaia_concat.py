'''
crossmatch new concat

'''

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import precision_recall as pr
import gc  # garbage collector
import review_findings_functions as rff
import os
import importlib
import implantation_functions as imp
import corner
from astropy.wcs import WCS
import shutil
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"

def add_ra_dec(olddf):
    fpath_header = "/lustre/lrspec/ZTF_data/corrected_coordinates/new_training_headers/"
    df = olddf.copy()
    df['star_ra']=np.full(len(df), np.nan)
    df['star_dec']=np.full(len(df), np.nan)
    for index, row in df.iterrows():
        image_filename = row['filename']
        # print(image_filename[:-6])
        quad = row['quadrant']
        
        xy_file = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/" + image_filename[:-6] + "_" + str(quad) + ".fits"
        header_file = fpath_header + image_filename[:-6] + "_" + str(quad) + ".wcs"
        # print(header_file)
    
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
            closest_x, closest_y = row['closest_x'], row['closest_y']
            ra, dec = wcs.all_pix2world(closest_x, closest_y, True)
            df.loc[index, 'star_ra' ] = ra
            df.loc[index, 'star_dec' ] = dec
    return df


def query_gaia_for_stars(olddf, radius=9, partial = "/lustre/lrspec/ZTF_data/gaia_partial.npy"):
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
    df = olddf.copy()
    # df[['gaia_source_id', 'g_mag', 'bp_rp', 'r_mag', 'i_mag', 'variability', 'in_vari_summary']
    gaia_columns = [
        'source_id', 'ra', 'dec', 'phot_g_mean_mag', 'bp_rp', 'phot_rp_mean_mag', 'phot_bp_mean_mag', 'phot_variable_flag'
    ]
    
    new_data = []
    if partial!=None:
        partial = np.load(partial, allow_pickle=True)
        partial_1 = np.load("/lustre/lrspec/ZTF_data/gaia_partial_1.npy", allow_pickle=True) 
        partial_2 = np.load("/lustre/lrspec/ZTF_data/gaia_partial_2.npy", allow_pickle=True) 
        partial_3 = np.load("/lustre/lrspec/ZTF_data/gaia_partial_3.npy", allow_pickle=True) 

        starting_row = len(partial)+len(partial_1)+len(partial_2)+len(partial_3)
    else:
        starting_row = 0
        
    for _, row in df[starting_row:].iterrows():
        if pd.isna(row['star_ra']) or pd.isna(row['star_dec']):
                print("NaN coordinate, skipping Gaia query for this row.")
                new_data.append([None, None, None, None, None])
                continue
        coord = SkyCoord(ra=row['star_ra']*u.deg, dec=row['star_dec']*u.deg, frame='icrs')
        
        query = f"""
        SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                       CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
        ORDER BY ztf_ri_gaia_rp_mag ASC
        """ #just swapped the order so now it's by RI
        if row['Band']=='g':
            query = f"""
        SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                       CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
        ORDER BY phot_bp_mean_mag ASC
        """ 
        else:
            query = f"""
        SELECT TOP 1 {', '.join(gaia_columns)} FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                       CIRCLE('ICRS', {row['star_ra']}, {row['star_dec']}, {radius}/3600.0))=1
        ORDER BY phot_rp_mean_mag ASC
        """ 
        
        job = Gaia.launch_job(query)
        results = job.get_results()
        
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
                source_id, 
                    best_match['phot_bp_mean_mag'], best_match['phot_rp_mean_mag'],
                best_match['phot_variable_flag'],
                is_variable
            ])
        else:
            new_data.append([None, None, None, None, None])
        np.save("/lustre/lrspec/ZTF_data/gaia_partial_4", new_data)
    
    df[['gaia_source_id',  'ztf_g_gaia_bp_rp_mag', 'ztf_ri_gaia_rp_mag',  'variability', 'in_vari_summary']] = new_data
    return df

nodupsconcat = pd.read_csv('/home/3155/nodupsconcat')

nodupsconcat = add_ra_dec(nodupsconcat)

nodupsconcat = query_gaia_for_stars(nodupsconcat)

nodupsconcat.to_csv('/home/3155/nodupsconcat_gaia')