#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
extracting all light curves and saving them in csv files
per image

'''


# In[13]:


import numpy as np
import pandas as pd
from astropy.io import fits
import os
import review_findings_functions as rff


# In[ ]:


#let's figure out where the x,y positions of the stars are


# In[115]:


import glob


# In[124]:





# In[120]:


def process_xy_positions_efficient(xy_positions, image_path, quad):
    """
    Efficiently process multiple x,y positions by loading the image once
    
    Parameters:
    -----------
    xy_positions : array-like
        Array of (x, y) coordinates, shape (N, 2)
    image_path : str
        Path to FITS file
    quad : int
        Quadrant number
    tracking : bool
        Whether to track streak position
    output_csv : str
        Output CSV filename
        
    Returns:
    --------
    nothing, saves light curves
    """
    xy_positions = np.array(xy_positions)
    
    filename = image_path[-47:-8]
    
    print(f"Loading FITS file: {image_path}")
    print(f"Processing {np.shape(xy_positions)[0]} positions...")
    
    # Load image once
    try:
        hdul = fits.open(image_path)
        file1_array = hdul[quad].data
        hdul.close()
        
        # Calculate background once
        print("Calculating background...")
        background = rff.get_image_background(file1_array)
        imagenobackground = file1_array - background
        
        print(f"Image shape: {imagenobackground.shape}")
        print(f"Background value: {background:.2f}")
        
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        return None
    
    # Process all positions
    
    for i, (x_pos, y_pos) in enumerate(xy_positions):
        try:
            tot_2,tot_3,tot_4,tot_5, maxcols = get_tot_from_xy_efficient(
                imagenobackground, int(x_pos), int(y_pos), tracking=True)
            
            # Create DataFrame for this position
            position_df = pd.DataFrame({
                'position_id': i,
                'x_input': x_pos,
                'y_input': y_pos,
                'row_index': range(len(tot_2)),
                'flux_hap_2': tot_2,
                'flux_hap_3': tot_3,
                'flux_hap_4': tot_4,
                'flux_hap_5': tot_5,
                'max_col': maxcols
            })
            
            

            individual_filename = f"/lustre/lrspec/ZTF_data/lightcurves/1020/lightcurve_{filename}_{quad}_{int(x_pos):04}_{int(y_pos):04}.csv"
            print("saving")
            print(individual_filename)
            position_df.to_csv(individual_filename, index=False)

                
        except Exception as e:
            print(f"Error processing position {i} ({x_pos}, {y_pos}): {e}")
            continue
    
    return


# In[125]:


def get_tot_from_xy_efficient(imagenobackground, x_pos, y_pos, tracking=True):
    """
    Extract flux from a single x,y position using pre-loaded image data
    
    Parameters:
    -----------
    imagenobackground : array
        Pre-loaded and background-subtracted image data
    x_pos : int
        X coordinate (column)
    y_pos : int
        Y coordinate (row)
    half_aperture : int
        Half-width of aperture
    tracking : bool
        Whether to track streak position
        
    Returns:
    --------
    tot : array
        Flux values for each row
    maxcols : array
        Column positions used for each row
    """
    if tracking == False:
        # Fixed position approach
        zeropoint = x_pos - 7
        search_start = max(0, zeropoint)
        search_end = min(imagenobackground.shape[1], zeropoint + 14)
        
        if search_end > search_start and y_pos - 10 >= 0 and y_pos + 10 < imagenobackground.shape[0]:
            averages = np.average(
                imagenobackground[y_pos - 10 : y_pos + 10, search_start : search_end], axis=0
            )
            maxind = np.argmax(averages)
            maxcol = search_start + maxind
        else:
            maxcol = x_pos
            
        tot = np.zeros(imagenobackground.shape[0])
        
        for r in range(imagenobackground.shape[0]):
            combined = imagenobackground[r, maxcol] if maxcol < imagenobackground.shape[1] else 0
            for i in range(1, half_aperture + 1):
                if maxcol - i >= 0:
                    combined += imagenobackground[r, maxcol - i]
                if maxcol + i < imagenobackground.shape[1]:
                    combined += imagenobackground[r, maxcol + i]
            tot[r] = combined
        
        return tot, np.full(len(tot), maxcol)
    
    else:
        # Tracking approach
        tot_2 = np.zeros(imagenobackground.shape[0])
        tot_3 = np.zeros(imagenobackground.shape[0])
        tot_4 = np.zeros(imagenobackground.shape[0])
        tot_5 = np.zeros(imagenobackground.shape[0])
        maxcols = np.zeros(imagenobackground.shape[0])
        zeropoint = x_pos - 10
        
        for r in range(imagenobackground.shape[0]):
            search_start = max(0, zeropoint)
            search_end = min(imagenobackground.shape[1], zeropoint + 20)
            
            if search_end > search_start:
                maxcol = np.argmax(imagenobackground[r, search_start:search_end]) + search_start
            else:
                maxcol = x_pos
                
            maxcols[r] = maxcol
            combined = imagenobackground[r, maxcol] if maxcol < imagenobackground.shape[1] else 0
            
            half_aperture = 2
            for i in range(1, half_aperture + 1):
                if maxcol - i >= 0:
                    combined += imagenobackground[r, maxcol - i]
                if maxcol + i < imagenobackground.shape[1]:
                    combined += imagenobackground[r, maxcol + i]
            
            tot_2[r] = combined
            
            half_aperture = 3
            for i in range(1, half_aperture + 1):
                if maxcol - i >= 0:
                    combined += imagenobackground[r, maxcol - i]
                if maxcol + i < imagenobackground.shape[1]:
                    combined += imagenobackground[r, maxcol + i]
            
            tot_3[r] = combined
            
            half_aperture = 4
            for i in range(1, half_aperture + 1):
                if maxcol - i >= 0:
                    combined += imagenobackground[r, maxcol - i]
                if maxcol + i < imagenobackground.shape[1]:
                    combined += imagenobackground[r, maxcol + i]
            
            tot_4[r] = combined
            
            half_aperture = 5
            for i in range(1, half_aperture + 1):
                if maxcol - i >= 0:
                    combined += imagenobackground[r, maxcol - i]
                if maxcol + i < imagenobackground.shape[1]:
                    combined += imagenobackground[r, maxcol + i]
            
            tot_5[r] = combined
            
            
        return tot_2,tot_3,tot_4,tot_5, maxcols
    


# In[14]:


def load_xy_positions_from_fits(bintable_path):
    """
    Load X,Y positions from FITS binary table
    
    Parameters:
    -----------
    bintable_path : str
        Path to FITS file containing binary table with X,Y coordinates
        
    Returns:
    --------
    xy_positions : array
        Array of (x, y) coordinates
    """
    try:
        with fits.open(bintable_path) as hdul:
            # Try to find the binary table (usually extension 1, but could vary)
            table_data = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if hasattr(hdu.data, 'dtype') and len(hdu.data.dtype.names or []) > 0:
                        table_data = hdu.data
                        print(f"Found binary table in extension {i}")
                        break
            
            if table_data is None:
                print(f"No binary table found in {bintable_path}")
                return None
            
            # Extract X,Y coordinates
            # Handle different possible column names
            x_col = None
            y_col = None
            
            for col_name in table_data.dtype.names:
                if col_name.upper() in ['X', 'X_POS', 'XPOS', 'COL']:
                    x_col = col_name
                elif col_name.upper() in ['Y', 'Y_POS', 'YPOS', 'ROW']:
                    y_col = col_name
            
            if x_col is None or y_col is None:
                print(f"Could not find X,Y columns in {bintable_path}")
                print(f"Available columns: {table_data.dtype.names}")
                return None
            
            # Extract positions
            x_vals = table_data[x_col]
            y_vals = table_data[y_col]
            xy_positions = np.column_stack([x_vals, y_vals])
            
            print(f"Loaded {len(xy_positions)} positions from {bintable_path}")
            print(f"X range: {x_vals.min():.1f} - {x_vals.max():.1f}")
            print(f"Y range: {y_vals.min():.1f} - {y_vals.max():.1f}")
            
            return xy_positions
            
    except Exception as e:
        print(f"Error loading positions from {bintable_path}: {e}")
        return None


# In[127]:


image_directory = '/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/'
xy_directory = '/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/'
lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/lightcurve_'

image_files = os.listdir(image_directory)

for image_filename in image_files:
    
    if image_filename[0:3] == 'ztf': #for somet reason there were other files in here
        # Create full path to image file
        image_path = os.path.join(image_directory, image_filename)
        
        # Extract filename parts for bintable
        bintable_filename = image_filename[-47:-13]  
        
        filename = image_filename[-47:-8]  
        
        # Check if lightcurve files already exist
        if glob.glob(lightcurve_path + filename + '*'+'_4_'+'*'):  # Fixed: use glob instead of is_file
            print('light curves already exist - skipping')
            print(lightcurve_path + filename)
            continue
        
        for quad in range(1, 5):
            bintable_path = xy_directory + bintable_filename + '_' + str(quad) + '.fits'  # Fixed: removed extra str() and parenthesis
            xy_positions = load_xy_positions_from_fits(bintable_path)
            process_xy_positions_efficient(xy_positions, image_path, quad)


#
