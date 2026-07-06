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



df = pd.read_csv("/home/3155/best_cands")




def make_html_file(*figs, filename, titles=None, instructions=None,
                   fits_filename=None, fits_url_base='./fits/', image_fig_indices=None,
                   prev_file=None, next_file=None):
    """
    Save multiple Plotly figures in one HTML file with optional FITS download and navigation
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

    # Navigation buttons HTML
    nav_html = '<div class="navigation-buttons">'
    if prev_file:
        nav_html += f'<button class="nav-btn prev-btn" onclick="window.location.href=\'{prev_file}\'">← Previous</button>'
    else:
        nav_html += '<button class="nav-btn prev-btn" disabled>← Previous</button>'
    
    if next_file:
        nav_html += f'<button class="nav-btn next-btn" onclick="window.location.href=\'{next_file}\'">Next →</button>'
    else:
        nav_html += '<button class="nav-btn next-btn" disabled>Next →</button>'
    nav_html += '</div>'

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
    
    # Full HTML with robust JS and navigation styling
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
        
        /* Navigation button styles */
        .navigation-buttons {{ 
            display: flex; 
            justify-content: space-between; 
            gap: 10px; 
            margin: 20px 0; 
            padding: 10px;
            background: white;
            border-radius: 4px;
            position: sticky;
            top: 10px;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .nav-btn {{ 
            padding: 10px 20px; 
            background: #2196F3; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px;
            font-weight: bold;
            transition: background 0.3s;
        }}
        .nav-btn:hover:not(:disabled) {{ 
            background: #1976D2; 
        }}
        .nav-btn:disabled {{ 
            background: #ccc; 
            cursor: not-allowed; 
            opacity: 0.6;
        }}
        .prev-btn {{
            margin-right: auto;
        }}
        .next-btn {{
            margin-left: auto;
        }}
    </style>
</head>
<body>
{nav_html}
{instructions_html}
{fits_html}
{figure_divs}
{nav_html}

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

// Keyboard navigation
document.addEventListener('keydown', function(event) {{
    if (event.key === 'ArrowLeft' && '{prev_file}') {{
        window.location.href = '{prev_file}';
    }} else if (event.key === 'ArrowRight' && '{next_file}') {{
        window.location.href = '{next_file}';
    }}
}});

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
    <p> A high-pass filtered version of the light curve with half aperture=2 is also plotted in blue.</p>
    
    
    <p><strong>Star Details:</strong> Type of transient (brightening (pos) or dimming (neg)): {df.loc[index]['real_type_local']}, local snr (filtered): {df.loc[index]['snr_local_filtered2']:.3f}, global snr (filtered): {df.loc[index]['snr_global_filtered2']:.3f}, local snr (unfiltered): {df.loc[index]['snr_local']:.3f}, global snr (unfiltered): {df.loc[index]['snr_global']:.3f}, central point in light curve from snr calc: {df.loc[index]['central_point']:.1f}, number of candidates in streak: {df.loc[index]['num_in_file_in_star']:.0f}</p>
    #<p><strong>Star Details:</strong> gaia_source_id: {df.loc[index]['gaia_source_id']:.1f}, known variable: {df.loc[index]['in_vari_summary']}, band: {df.loc[index]['Band']}</p>
    <p><strong>FITS Quadrant:</strong> {quadrant}, approximate row: {df.loc[index]['row_from_nn']+10}, approximate col: {df.loc[index]['col_from_nn']+10} </p>
    '''
    #<p><strong>Star Details:</strong> gaia_source_id: {df.loc[index]['gaia_source_id']:.1f}, known variable: {df.loc[index]['in_vari_summary']}, band: {df.loc[index]['Band']}, local snr: {df.loc[index]['snr_local']:.3f}, global snr: {df.loc[index]['snr_global']:.3f}</p>

    fig_1 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=2, old_lines=False)
    fig_2 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=5,old_lines=False)
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
    
    # Determine previous and next filenames
    prev_file = None
    next_file = None
    
    if index > df.index.min():
        prev_index = index - 1
        prev_file = f"cand_{prev_index}_with_images.html"
    
    if index < df.index.max():
        next_index = index + 1
        next_file = f"cand_{next_index}_with_images.html"
    
    make_html_file(
        fig_1, fig_2, fig_5, 
        filename=filename, 
        instructions=instructions_html,
        fits_filename=fits_file,
        fits_url_base='../fits_files/',
        image_fig_indices=[2],  # Only figure 3 (index 2) gets download/contrast controls
        prev_file=prev_file,
        next_file=next_file
    )
    
#     fig_1 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=2, old_lines=False)
#     fig_2 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=5,old_lines=False)
#     fig_5 = rff.plot_whole_image_plotly(
#         df, index,
#         direcname="1020",
#         half_aperture=2,
#         tracking=True,
#         title=None,
#         plot_tracking=False,
#     )
    
#     path = '/lustre/lrspec/ZTF_data/html_files/with_images/'
#     filename = path + "cand_" + str(index) + "_with_images.html"
    
#     # Determine previous and next filenames
#     prev_file = None
#     next_file = None
    
#     if index > df.index.min():
#         prev_index = index - 1
#         prev_file = f"cand_{prev_index}_with_images.html"
    
#     if index < df.index.max():
#         next_index = index + 1
#         next_file = f"cand_{next_index}_with_images.html"
    
#     make_html_file(
#         fig_1, fig_2, fig_5, 
#         filename=filename, 
#         instructions=instructions_html,
#         fits_filename=fits_file,
#         fits_url_base='../fits_files/',
#         image_fig_indices=[2],  # Only figure 3 (index 2) gets download/contrast controls
#         prev_file=prev_file,
#         next_file=next_file
#     )