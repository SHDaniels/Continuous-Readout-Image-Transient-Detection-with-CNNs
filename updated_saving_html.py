import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import review_findings_functions as rff
import os
from astropy.io import fits
import importlib
import shutil
import json


# In[2]:


import plotly.io as pio



df = pd.read_csv('/home/3155/best_cands_nosim')


# def make_html_file(*figs, filename, titles=None, instructions=None,
#                    fits_filename=None, fits_url_base='./fits/', image_fig_indices=None):
#     """
#     Save multiple Plotly figures in one HTML file with optional FITS download
#     and robust contrast/brightness controls for image figures.
#     """
#     num_figs = len(figs)
#     if num_figs == 0:
#         print("✗ No figures provided")
#         return False
    
#     if titles is None:
#         titles = [None] * num_figs
#     elif len(titles) != num_figs:
#         print(f"✗ Number of titles ({len(titles)}) doesn't match number of figures ({num_figs})")
#         return False
    
#     instructions_html = f'<div class="instructions">{instructions}</div>' if instructions else ""

#     # FITS download HTML
#     fits_html = ""
#     if fits_filename:
#         fits_url = fits_url_base + fits_filename
#         fits_html = f'''
#         <div class="fits-download-section">
#             <button class="fits-download-btn" onclick="downloadFits()">Download FITS</button>
#             <span class="fits-filename">{fits_filename}</span>
#         </div>
#         '''
    
#     # Prepare figure divs
#     figure_divs = ""
#     for i, (fig, title) in enumerate(zip(figs, titles)):
#         include_js = 'cdn' if i == 0 else False
#         fig_html = fig.to_html(include_plotlyjs=include_js, full_html=False, div_id=f"fig{i}")

#         control_panel = ""
#         if image_fig_indices and i in image_fig_indices:
#             control_panel = f'''
#             <div class="control-panel">
#                 <label>Contrast:</label>
#                 <input type="range" id="contrast-slider-fig{i}" min="0.1" max="10.0" step="0.1" value="1.0"
#                        oninput="adjustContrastFromSlider('fig{i}', this.value)">
#                 <input type="number" id="contrast-input-fig{i}" min="0.1" max="10.0" step="0.1" value="1.0"
#                        onchange="adjustContrastFromInput('fig{i}', this.value)">
#                 <label>Brightness:</label>
#                 <input type="range" id="brightness-slider-fig{i}" min="-2.0" max="2.0" step="0.1" value="0.0"
#                        oninput="adjustBrightnessFromSlider('fig{i}', this.value)">
#                 <input type="number" id="brightness-input-fig{i}" min="-2.0" max="2.0" step="0.1" value="0.0"
#                        onchange="adjustBrightnessFromInput('fig{i}', this.value)">
#                 <button class="reset-btn" onclick="resetContrast('fig{i}')">Reset</button>
#             </div>
#             '''
#         figure_divs += f'<div class="figure-container">{"<div class=figure-title>"+title+"</div>" if title else ""}{control_panel}{fig_html}</div>'
    
#     # Full HTML with robust JS
#     html = f'''
# <!DOCTYPE html>
# <html>
# <head>
#     <meta charset="utf-8">
#     <title>Combined Figures</title>
#     <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
#     <style>
#         body {{ font-family: Arial; background-color: #f8f9fa; margin:10px; }}
#         .figure-container {{ margin: 10px 0; padding:10px; background:white; border-radius:4px; }}
#         .figure-title {{ font-weight:bold; margin-bottom:5px; font-size:16px; }}
#         .control-panel {{ margin:10px 0; display:flex; gap:10px; flex-wrap:wrap; }}
#         .fits-download-btn {{ padding:6px 12px; background:#ff9800; color:white; border:none; border-radius:4px; cursor:pointer; }}
#         .fits-download-btn:hover {{ background:#f57c00; }}
#     </style>
# </head>
# <body>
# {instructions_html}
# {fits_html}
# {figure_divs}

# <script>
# var originalImageData = {{}};

# // FITS download
# function downloadFits() {{
#     var link = document.createElement('a');
#     link.href = '{fits_url_base}{fits_filename if fits_filename else ""}';
#     link.download = '{fits_filename if fits_filename else ""}';
#     document.body.appendChild(link);
#     link.click();
#     document.body.removeChild(link);
# }}

# // Contrast/Brightness helpers
# function adjustContrastFromSlider(plotId, value) {{
#     document.getElementById('contrast-input-'+plotId).value = value;
#     applyContrast(plotId, parseFloat(value), parseFloat(document.getElementById('brightness-slider-'+plotId)?.value || 0));
# }}
# function adjustContrastFromInput(plotId, value) {{
#     document.getElementById('contrast-slider-'+plotId).value = value;
#     applyContrast(plotId, parseFloat(value), parseFloat(document.getElementById('brightness-slider-'+plotId)?.value || 0));
# }}
# function adjustBrightnessFromSlider(plotId, value) {{
#     document.getElementById('brightness-input-'+plotId).value = value;
#     applyContrast(plotId, parseFloat(document.getElementById('contrast-slider-'+plotId)?.value || 1.0), parseFloat(value));
# }}
# function adjustBrightnessFromInput(plotId, value) {{
#     document.getElementById('brightness-slider-'+plotId).value = value;
#     applyContrast(plotId, parseFloat(document.getElementById('contrast-slider-'+plotId)?.value || 1.0), parseFloat(value));
# }}
# function resetContrast(plotId) {{
#     document.getElementById('contrast-slider-'+plotId).value = 1.0;
#     document.getElementById('contrast-input-'+plotId).value = 1.0;
#     document.getElementById('brightness-slider-'+plotId).value = 0.0;
#     document.getElementById('brightness-input-'+plotId).value = 0.0;
#     applyContrast(plotId, 1.0, 0.0);
# }}

# function applyContrast(plotId, contrast=1.0, brightness=0.0) {{
#     var plot = document.getElementById(plotId);
#     if(!plot) return;
#     var traces = plot.data;
#     if(!traces) return;

#     for(var i=0; i<traces.length; i++) {{
#         var trace = traces[i];
#         if(trace.type !== 'heatmap' || !trace.z) continue;

#         if(!originalImageData[plotId]) originalImageData[plotId] = [];
#         if(!originalImageData[plotId][i]) {{
#             // Deep copy to normal arrays
#             originalImageData[plotId][i] = Array.from(trace.z, row => Array.from(row));
#         }}

#         var zorig = originalImageData[plotId][i];
#         var zflat = [];
#         for(var r=0; r<zorig.length; r++) zflat = zflat.concat(zorig[r]);

#         var zmin = Math.min(...zflat);
#         var zmax = Math.max(...zflat);
#         var range = zmax - zmin;
#         if(range === 0) range = 1;

#         var gamma = 1.0 / contrast;

#         var newZ = zorig.map(row => row.map(v => {{
#             let norm = (v - zmin)/range;
#             norm = Math.pow(Math.min(Math.max(norm,0),1), gamma);
#             norm += brightness*0.1; // brightness adjustment
#             return zmin + Math.min(Math.max(norm,0),1)*range;
#         }}));

#         // Update zmin/zmax for Plotly
#         var zmid = zflat[Math.floor(zflat.length/2)];
#         var newRange = range / Math.pow(contrast, 1.5);
#         var newZmin = zmid - newRange/2;
#         var newZmax = zmid + newRange/2;

#         Plotly.restyle(plotId, {{z:[newZ], zmin:newZmin, zmax:newZmax}}, [i]);
#     }}
# }}
# </script>
# </body>
# </html>
# '''
#     with open(filename, 'w') as f:
#         f.write(html)
    
#     print(f"✓ HTML saved: {filename} ({os.path.getsize(filename)} bytes)")
#     return True



def make_html_file(*figs, filename, titles=None, instructions=None,
                   fits_filename=None, fits_url_base='./fits/', image_fig_indices=None):
    """
    Save multiple Plotly figures in one HTML file with optional FITS download
    """
    num_figs = len(figs)
    if num_figs == 0:
        print("✗ No figures provided")
        return False
    
    if titles is None:
        titles = [None] * num_figs
    elif len(titles) != num_figs:
        print(f"✗ Number of titles ({len(titles)}) doesn't match number of figures ({num_figs})")
        return False
    
    instructions_html = f'<div class="instructions">{instructions}</div>' if instructions else ""

    # FITS download HTML
    fits_html = ""
    if fits_filename:
        fits_url = fits_url_base + fits_filename
        fits_html = f'''
        <div class="fits-download-section">
            <button class="fits-download-btn" onclick="downloadFits()">Download FITS</button>
            <span class="fits-filename">{fits_filename}</span>
        </div>
        '''
    
    # Prepare figure divs
    figure_divs = ""
    for i, (fig, title) in enumerate(zip(figs, titles)):
        include_js = 'cdn' if i == 0 else False
        fig_html = fig.to_html(include_plotlyjs=include_js, full_html=False, div_id=f"fig{i}")

        control_panel = ""

        figure_divs += f'<div class="figure-container">{"<div class=figure-title>"+title+"</div>" if title else ""}{control_panel}{fig_html}</div>'
    
    # Full HTML with robust JS
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Combined Figures</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{ font-family: Arial; background-color: #f8f9fa; margin:10px; }}
        .figure-container {{ margin: 10px 0; padding:10px; background:white; border-radius:4px; }}
        .figure-title {{ font-weight:bold; margin-bottom:5px; font-size:16px; }}
        .control-panel {{ margin:10px 0; display:flex; gap:10px; flex-wrap:wrap; }}
        .fits-download-btn {{ padding:6px 12px; background:#ff9800; color:white; border:none; border-radius:4px; cursor:pointer; }}
        .fits-download-btn:hover {{ background:#f57c00; }}
    </style>
</head>
<body>
{instructions_html}
{fits_html}
{figure_divs}

<script>
var originalImageData = {{}};

// FITS download
function downloadFits() {{
    var link = document.createElement('a');
    link.href = '{fits_url_base}{fits_filename if fits_filename else ""}';
    link.download = '{fits_filename if fits_filename else ""}';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}}



</script>
</body>
</html>
'''
    with open(filename, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML saved: {filename} ({os.path.getsize(filename)} bytes)")
    return True



    
    
    
    
    

for index in df.index:
# index=0
# if index==0:
    
    # Get FITS filename and quadrant
    fits_file = df.loc[index]['filename'] + 'fits.fz'
    quadrant = df.loc[index]['quadrant']
    
    # #copy the file over as well so we can access it 
    # shutil.copyfile('/lustre/lrspec/ZTF_data/2022/1020/'+fits_file, 
    #                 '/lustre/lrspec/ZTF_data/html_files/fits_files/'+fits_file)
    
    instructions_html = f'''
    <p>Data points surrounding the center of the transient candidate are shown as white markers.</p>
    
    <p>You can interact with the charts:</p>
    <ul>
        <li>Hover over data points to see detailed values</li>
        <li>Click and drag to zoom into specific areas</li>
        <li>Double-click to reset zoom</li>
    </ul>
    <p><strong>Additional Note:</strong> Half-aperture=2 was determined to be the most reasonable aperture size (width of the streak), but a much larger aperture is shown in the second plot</p> 
    <p> because in some cases the streak spreads past the chosen aperture, causing false positives.</p>
    
    <p><strong>Star Details:</strong> gaia_source_id: {df.loc[index]['gaia_source_id']:.1f}, known variable: {df.loc[index]['in_vari_summary']}, band: {df.loc[index]['Band']}, local snr: {df.loc[index]['snr_local']:.3f}, global snr: {df.loc[index]['snr_global']:.3f}</p>
    <p><strong>FITS Quadrant:</strong> {quadrant}, row:{df.loc[index]['row_from_nn']+10}, col: {df.loc[index]['col_from_nn']+10} </p>
    '''
    
    fig_1 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=2)
    fig_2 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=5)
    fig_5 = rff.plot_whole_image_plotly(
        df, index,
        direcname="1020",
        half_aperture=2,
        tracking=True,
        title=None,
        plot_tracking=False,
    )
    
    path = '/lustre/lrspec/ZTF_data/html_files/with_images/'
    filename = path + "cand_" + str(index) + "_with_images.html"
    
    make_html_file(
        fig_1, fig_2, fig_5, 
        filename=filename, 
        instructions=instructions_html,
        fits_filename=fits_file,
        fits_url_base='../fits_files/',
        image_fig_indices=[2]  # Only figure 3 (index 2) gets download/contrast controls
    )