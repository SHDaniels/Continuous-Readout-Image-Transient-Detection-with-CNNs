#!/usr/bin/env python
# coding: utf-8

"""
Script to add contrast controls and download buttons to existing HTML files
containing Plotly figures.
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

def add_controls_to_html(html_content):
    """
    Add contrast controls and download buttons to HTML file.
    
    Parameters:
    html_content: string containing the HTML content
    
    Returns:
    Modified HTML content as string
    """
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all figure containers
    figure_containers = soup.find_all('div', class_='figure-container')
    
    if not figure_containers:
        print("  ⚠ No figure containers found")
        return html_content
    
    # Add controls to each figure container
    for i, container in enumerate(figure_containers):
        if i==2:
            # Find the plot div inside this container
            plot_div = container.find('div', id=re.compile(r'(fig\d+|plot\d+)'))
            
            if not plot_div:
                continue
            
            plot_id = plot_div.get('id', f'plot{i+1}')
            
            # Create control panel
            control_panel = soup.new_tag('div', **{'class': 'control-panel'})
            
            # Download button
            download_btn = soup.new_tag('button', **{
                'class': 'download-btn',
                'onclick': f'downloadPlot("{plot_id}")'
            })
            download_btn.string = f'Download Figure {i+1}'
            
            # Contrast controls (only for image plots - typically the last figures)
            contrast_controls = soup.new_tag('div', **{'class': 'contrast-controls'})
            
            contrast_label = soup.new_tag('label', **{'for': f'contrast-{plot_id}'})
            contrast_label.string = 'Contrast: '
            
            contrast_slider = soup.new_tag('input', **{
                'type': 'range',
                'id': f'contrast-slider-{plot_id}',
                'class': 'contrast-slider',
                'min': '0.1',
                'max': '5.0',
                'step': '0.1',
                'value': '1.0',
                'oninput': f'adjustContrastFromSlider("{plot_id}", this.value)'
            })
            
            contrast_input = soup.new_tag('input', **{
                'type': 'number',
                'id': f'contrast-input-{plot_id}',
                'class': 'contrast-input',
                'min': '0.01',
                'max': '100',
                'step': '0.1',
                'value': '1.0',
                'onchange': f'adjustContrastFromInput("{plot_id}", this.value)'
            })
            
            reset_btn = soup.new_tag('button', **{
                'class': 'reset-btn',
                'onclick': f'resetContrast("{plot_id}")'
            })
            reset_btn.string = 'Reset'
            
            # Assemble contrast controls
            contrast_controls.append(contrast_label)
            contrast_controls.append(contrast_slider)
            contrast_controls.append(contrast_input)
            contrast_controls.append(reset_btn)
            
            # Assemble control panel
            control_panel.append(download_btn)
            control_panel.append(contrast_controls)
            
            # Insert control panel before the plot div
            plot_div.insert_before(control_panel)
    
    # Add CSS styles
    style_tag = soup.find('style')
    if style_tag:
        additional_css = """
        .control-panel {
            margin: 10px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        .download-btn, .reset-btn {
            padding: 8px 16px;
            background-color: #2196f3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        .download-btn:hover, .reset-btn:hover {
            background-color: #1976d2;
        }
        .reset-btn {
            background-color: #757575;
            padding: 6px 12px;
        }
        .reset-btn:hover {
            background-color: #616161;
        }
        .contrast-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .contrast-slider {
            width: 150px;
        }
        .contrast-input {
            width: 70px;
            padding: 4px 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        """
        style_tag.string += additional_css
    
    # Add JavaScript functions
    body_tag = soup.find('body')
    if body_tag:
        script_tag = soup.new_tag('script')
        script_tag.string = """
        // Store original image data for each plot
        var originalImageData = {};
        
        // Download plot as PNG
        function downloadPlot(plotId) {
            Plotly.downloadImage(plotId, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: plotId + '_' + new Date().getTime()
            });
        }
        
        // Sync slider and input, then adjust contrast
        function adjustContrastFromSlider(plotId, value) {
            var inputElement = document.getElementById('contrast-input-' + plotId);
            inputElement.value = value;
            applyContrast(plotId, value);
        }
        
        function adjustContrastFromInput(plotId, value) {
            // Validate input
            var numValue = parseFloat(value);
            if (isNaN(numValue) || numValue < 0.01) {
                numValue = 0.01;
            } else if (numValue > 100) {
                numValue = 100;
            }
            
            // Update input to validated value
            var inputElement = document.getElementById('contrast-input-' + plotId);
            inputElement.value = numValue;
            
            // Update slider if value is in range
            var sliderElement = document.getElementById('contrast-slider-' + plotId);
            if (numValue >= parseFloat(sliderElement.min) && numValue <= parseFloat(sliderElement.max)) {
                sliderElement.value = numValue;
            }
            
            applyContrast(plotId, numValue);
        }
        
        // Apply contrast adjustment to plot
        function applyContrast(plotId, value) {
            var plotElement = document.getElementById(plotId);
            if (!plotElement || !plotElement.data) return;
            
            // Store original data and colorscale settings if not already stored
            if (!originalImageData[plotId]) {
                originalImageData[plotId] = {
                    data: [],
                    zmin: [],
                    zmax: [],
                    zmid: []
                };
                for (var i = 0; i < plotElement.data.length; i++) {
                    var trace = plotElement.data[i];
                    if (trace.type === 'heatmap' || trace.type === 'image' || trace.z) {
                        // Store original z data
                        originalImageData[plotId].data[i] = JSON.parse(JSON.stringify(trace.z));
                        
                        // Calculate and store data statistics
                        var flatData = [];
                        for (var j = 0; j < trace.z.length; j++) {
                            for (var k = 0; k < trace.z[j].length; k++) {
                                if (trace.z[j][k] !== null && trace.z[j][k] !== undefined && !isNaN(trace.z[j][k])) {
                                    flatData.push(trace.z[j][k]);
                                }
                            }
                        }
                        
                        if (flatData.length > 0) {
                            flatData.sort(function(a, b) { return a - b; });
                            var minVal = flatData[0];
                            var maxVal = flatData[flatData.length - 1];
                            var medianVal = flatData[Math.floor(flatData.length / 2)];
                            
                            // Use median as center point for contrast
                            originalImageData[plotId].zmin[i] = minVal;
                            originalImageData[plotId].zmax[i] = maxVal;
                            originalImageData[plotId].zmid[i] = medianVal;
                        }
                    }
                }
            }
            
            // Apply contrast adjustment using colorscale limits
            var contrastValue = parseFloat(value);
            
            for (var i = 0; i < plotElement.data.length; i++) {
                if (originalImageData[plotId].data[i]) {
                    var zmid = originalImageData[plotId].zmid[i];
                    var zmin = originalImageData[plotId].zmin[i];
                    var zmax = originalImageData[plotId].zmax[i];
                    
                    // Calculate new limits based on contrast
                    // Higher contrast = narrower range around median
                    // Lower contrast = wider range
                    var range = (zmax - zmin) / 2;
                    var newRange = range / contrastValue;
                    
                    var newZmin = zmid - newRange;
                    var newZmax = zmid + newRange;
                    
                    // Update colorscale limits
                    Plotly.restyle(plotId, {
                        'zmin': newZmin,
                        'zmax': newZmax
                    }, [i]);
                }
            }
        }
        
        // Reset contrast to original
        function resetContrast(plotId) {
            var slider = document.getElementById('contrast-slider-' + plotId);
            var input = document.getElementById('contrast-input-' + plotId);
            slider.value = 1.0;
            input.value = 1.0;
            
            if (originalImageData[plotId]) {
                var plotElement = document.getElementById(plotId);
                for (var i = 0; i < plotElement.data.length; i++) {
                    if (originalImageData[plotId].data[i]) {
                        // Reset to original colorscale limits
                        var zmin = originalImageData[plotId].zmin[i];
                        var zmax = originalImageData[plotId].zmax[i];
                        
                        Plotly.restyle(plotId, {
                            'zmin': zmin,
                            'zmax': zmax
                        }, [i]);
                    }
                }
            }
        }
        """
        body_tag.append(script_tag)
    
    return str(soup)


def process_html_files(input_dir, output_dir=None, in_place=False):
    """
    Process all HTML files in a directory, adding controls.
    
    Parameters:
    input_dir: directory containing HTML files
    output_dir: directory to save modified files (if None and not in_place, creates 'modified' subdir)
    in_place: if True, modifies files in place (overwrites originals)
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"✗ Input directory does not exist: {input_dir}")
        return
    
    # Determine output directory
    if in_place:
        output_path = input_path
        print(f"⚠ WARNING: Modifying files in place at {input_path}")
    elif output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_path}")
    else:
        output_path = input_path / "modified"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_path}")
    
    # Find all HTML files
    html_files = list(input_path.glob("*.html"))
    
    if not html_files:
        print(f"✗ No HTML files found in {input_dir}")
        return
    
    print(f"\n🔍 Found {len(html_files)} HTML files")
    print("=" * 60)
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for html_file in html_files:
        try:
            print(f"\n📄 Processing: {html_file.name}")
            
            # Read original file
            with open(html_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Add controls
            modified_content = add_controls_to_html(original_content)
            
            # Write modified file
            output_file = output_path / html_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            # Verify file was created
            if output_file.exists():
                size_diff = len(modified_content) - len(original_content)
                print(f"  ✓ Success! Size change: +{size_diff} bytes")
                success_count += 1
            else:
                print(f"  ✗ Failed to create output file")
                error_count += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {error_count}")
    print(f"  📁 Total: {len(html_files)}")
    
    if success_count > 0:
        print(f"\n✓ Modified files saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = '/lustre/lrspec/ZTF_data/html_files/with_images/'
    OUTPUT_DIR = None  # Set to None to create 'modified' subdirectory
    IN_PLACE = False  # Set to True to overwrite original files (use with caution!)
    
    # Option to process a single file for testing
    TEST_SINGLE_FILE = True
    TEST_FILE = '/lustre/lrspec/ZTF_data/html_files/with_images/cand_0_with_images.html'
    
    if TEST_SINGLE_FILE:
        print("🧪 TEST MODE: Processing single file")
        print("=" * 60)
        
        test_path = Path(TEST_FILE)
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = add_controls_to_html(content)
            
            output_file = test_path.parent / f"{test_path.stem}_modified.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(modified)
            
            print(f"✓ Test file created: {output_file}")
        else:
            print(f"✗ Test file not found: {TEST_FILE}")
    else:
        # Process all files in directory
        process_html_files(INPUT_DIR, OUTPUT_DIR, IN_PLACE)