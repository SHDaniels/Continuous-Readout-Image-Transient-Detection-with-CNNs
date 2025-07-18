#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
saving html files
'''


# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import review_findings_functions as rff
import os
from astropy.io import fits
import importlib


# In[2]:


import plotly.io as pio



df = pd.read_csv('/home/3155/best_cands_nosim')




def make_html_file(*figs, filename, titles=None, instructions=None):
    """
    save multiple Plotly figures in one HTML file
    
    Parameters:
    *figs: Variable number of Plotly figure objects
    filename: output HTML filename
    titles: optional list of titles for each figure (if None, no titles shown)
    instructions: optional string with instructional text to display above figures
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
    
    # Prepare instructions HTML
    instructions_html = ""
    if instructions:
        instructions_html = f'''
        <div class="instructions">
            <div class="instructions-content">{instructions}</div>
        </div>
        '''
    
    try:
        # Method 1: Try using to_html() for all figures
        try:
            html_parts = []
            
            for i, fig in enumerate(figs):
                if i == 0:
                    # First figure includes plotly.js
                    html_content = fig.to_html(include_plotlyjs='cdn', div_id=f"fig{i+1}")
                else:
                    # Subsequent figures don't include plotly.js
                    html_content = fig.to_html(include_plotlyjs=False, div_id=f"fig{i+1}")
                
                html_parts.append(html_content)
            
            # Combine them with minimal HTML structure
            figure_divs = ""
            for i, (html_part, title) in enumerate(zip(html_parts, titles)):
                if title:
                    figure_divs += f'<div class="figure-title">{title}</div>'
                figure_divs += f'<div class="figure-container">{html_part}</div>'
            
            combined_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Combined Figures</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 10px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .instructions {{
            margin: 20px 0;
            padding: 20px;
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .instructions h2 {{
            margin-top: 0;
            color: #1976d2;
            font-size: 20px;
        }}
        .instructions-content {{
            color: #333;
            font-size: 14px;
        }}
        .instructions-content p {{
            margin: 10px 0;
        }}
        .instructions-content ul, .instructions-content ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .figure-container {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .figure-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            padding-bottom: 5px;
            border-bottom: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    {instructions_html}
    {figure_divs}
</body>
</html>
'''
            
            with open(filename, 'w') as f:
                f.write(combined_html)
            
            if os.path.exists(filename):
                print(f"✓ Combined save successful: {filename}")
                print(f"File size: {os.path.getsize(filename)} bytes")
                print(f"Number of figures: {num_figs}")
                return True
                
        except Exception as e:
            print(f"Method 1 failed: {e}")
            pass
        
        # Method 2: Manual construction if Method 1 fails
        try:
            # Get the plotly JSON data manually
            import json
            
            # Extract figure data and layout for all figures
            fig_data_list = []
            for fig in figs:
                fig_json = fig.to_json()
                fig_data_list.append(json.loads(fig_json))
            
            # Build plot divs
            plot_divs = ""
            plot_scripts = ""
            
            for i, (fig_data, title) in enumerate(zip(fig_data_list, titles)):
                if title:
                    plot_divs += f'<div class="figure-title">{title}</div>'
                
                plot_divs += f'<div class="figure-container"><div id="plot{i+1}" class="plot-div"></div></div>'
                
                plot_scripts += f'''
        // Plot {i+1}
        var fig{i+1}_data = {json.dumps(fig_data['data'])};
        var fig{i+1}_layout = {json.dumps(fig_data['layout'])};
        Plotly.newPlot('plot{i+1}', fig{i+1}_data, fig{i+1}_layout);
        '''
            
            # Build HTML with embedded JavaScript
            combined_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Combined Figures</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 10px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .instructions {{
            margin: 20px 0;
            padding: 20px;
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .instructions h2 {{
            margin-top: 0;
            color: #1976d2;
            font-size: 20px;
        }}
        .instructions-content {{
            color: #333;
            font-size: 14px;
        }}
        .instructions-content p {{
            margin: 10px 0;
        }}
        .instructions-content ul, .instructions-content ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .figure-container {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .figure-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            padding-bottom: 5px;
            border-bottom: 1px solid #dee2e6;
        }}
        .plot-div {{
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>
    {instructions_html}
    {plot_divs}
    
    <script>
        {plot_scripts}
    </script>
</body>
</html>
'''
            
            with open(filename, 'w') as f:
                f.write(combined_html)
            
            if os.path.exists(filename):
                print(f"✓ Manual combined save successful: {filename}")
                print(f"File size: {os.path.getsize(filename)} bytes")
                print(f"Number of figures: {num_figs}")
                return True
            else:
                print("✗ Manual combined save failed - file not created")
                return False
                
        except Exception as e:
            print(f"✗ Manual combined save failed with error: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Combined save failed with error: {e}")
        return False





for index in df.index:
    
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
    '''
    fig_1 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=2)
    fig_2 = rff.plot_lc_from_csv_with_transient_df_plotly(None, df, index, hap=5)
    # fig_3 = rff.plot_whole_streak_plotly(
    #     df, index,
    #     direcname="1020",
    #     half_aperture=2,
    #     tracking=True,
    #     title=None,
    #     plot_tracking=False,
    # )
    # fig_4 = rff.plot_transient_streak_plotly(
    #     df, index,
    #     direcname="1020",
    #     half_aperture=2,
    #     tracking=True,
    #     title=None,
    #     plot_tracking=False,
    # )
    fig_5 = rff.plot_whole_image_plotly(
        df, index,
        direcname="1020",
        half_aperture=2,
        tracking=True,
        title=None,
        plot_tracking=False,
    )
    path = '/lustre/lrspec/ZTF_data/html_files/with_images/'
    filename = path+"cand_"+str(index)+"_with_images.html"
    # path = '/lustre/lrspec/ZTF_data/html_files/without_images/'
    # filename = path+"cand_"+str(index)+"_without_images.html"
    
    make_html_file(fig_1, fig_2, fig_5,filename=filename,instructions=instructions_html)


