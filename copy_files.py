#!/usr/bin/env python3
"""
Copy files modified in the last 12 hours from source to destination folder.
Stops after copying 100,000 files.
"""

import os
import shutil
import time
from pathlib import Path

# Configuration
SOURCE_DIR = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
DEST_DIR = "/lustre/lrspec/ZTF_data/lightcurves/selection_new_ap"  # Change this to your desired destination
MAX_FILES = 100000
TIME_WINDOW = 18 * 3600  # 12 hours in seconds

def copy_recent_files(source_dir, dest_dir, max_files, time_window):
    """
    Copy files modified within time_window to dest_dir.
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
        max_files: Maximum number of files to copy
        time_window: Time window in seconds
    """
    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    current_time = time.time()
    cutoff_time = current_time - time_window
    files_copied = 0
    files_skipped = 0
    
    print(f"Scanning directory: {source_dir}")
    print(f"Looking for files modified in the last {time_window/3600:.1f} hours")
    print(f"Will copy up to {max_files:,} files to: {dest_dir}\n")
    
    try:
        # Walk through source directory
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                # Check if we've reached the limit
                if files_copied >= max_files:
                    print(f"\nReached maximum of {max_files:,} files. Stopping.")
                    return files_copied, files_skipped
                
                source_path = os.path.join(root, filename)
                
                try:
                    # Get file modification time
                    mtime = os.path.getmtime(source_path)
                    
                    # Check if file was modified within the time window
                    if mtime >= cutoff_time:
                        # Preserve directory structure in destination
                        rel_path = os.path.relpath(source_path, source_dir)
                        dest_path = os.path.join(dest_dir, rel_path)
                        
                        # Create subdirectories if needed
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(source_path, dest_path)
                        files_copied += 1
                        
                        if files_copied % 1000 == 0:
                            print(f"Copied {files_copied:,} files...")
                    else:
                        files_skipped += 1
                        
                except (OSError, IOError) as e:
                    print(f"Error processing {source_path}: {e}")
                    continue
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user.")
        return files_copied, files_skipped
    
    return files_copied, files_skipped

if __name__ == "__main__":
    start_time = time.time()
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory does not exist: {SOURCE_DIR}")
        exit(1)
    
    # Run the copy operation
    copied, skipped = copy_recent_files(SOURCE_DIR, DEST_DIR, MAX_FILES, TIME_WINDOW)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Copy operation completed!")
    print(f"Files copied: {copied:,}")
    print(f"Files skipped (too old): {skipped:,}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Destination: {DEST_DIR}")
    print(f"{'='*50}")