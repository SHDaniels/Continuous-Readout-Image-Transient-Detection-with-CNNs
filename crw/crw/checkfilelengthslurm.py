import os
import subprocess
import pandas as pd
import argparse
import crw_config as config

def get_file_info_batch(directory, batch_size=10000, name=None):
    """Efficiently get file info for millions of files using find + wc"""
    all_files = []
    if name is None:
        name = "*csv"
    print(directory+name)

    # Use find to get all CSV files (bypasses argument limit)
    cmd = ['find', directory, '-name', name, '-type', 'f']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    files = result.stdout.strip().split('\n')

    print(f"Found {len(files)} files")

    # Process in batches to avoid memory issues
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(files)//batch_size)+1}")

        wc_cmd = ['wc', '-l'] + batch
        wc_result = subprocess.run(wc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        for line in wc_result.stdout.strip().split('\n'):
            if line and 'total' not in line:
                path = line.lstrip().strip().split(" ")[-1]
                filename = os.path.basename(path).split("_")
                datetime = filename[2]
                run = filename[3]
                thisfilter = filename[4]
                choosechip = filename[5].replace("c","")
                sequencenumber = filename[7]
                amp = filename[8]
                x = filename[9]
                y = filename[10]
                parts = line.split()
                if len(parts) == 2:
                    count, path = parts
                    all_files.append({
                        'datetime': datetime,
                        'run': "%06d"%int(run),
                        'thisfilter': thisfilter,
                        'choosechip': "%02d"%int(choosechip),
                        'sequencenumber': "%04d"%int(sequencenumber),
                        'amp': amp,
                        'x': x,
                        'y': y,
                        'filename': os.path.basename(path),
                        'lines': int(count)-1,
                    })

    return pd.DataFrame(all_files)

def get_file_info_batch_chunked(directory, chunk_id, total_chunks, batch_size=10000, name=None):
    """Process only one chunk of files (for SLURM array)"""
    if name is None:
        name = "*csv"
    
    # Get all files
    cmd = ['find', directory, '-name', name, '-type', 'f']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    files = result.stdout.strip().split('\n')
    files = [f for f in files if f]
    
    # Calculate chunk boundaries
    chunk_size = (len(files) + total_chunks - 1) // total_chunks
    start_idx = chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(files))
    
    if start_idx >= len(files):
        print(f"Chunk {chunk_id}: no files")
        return pd.DataFrame()
    
    chunk_files = files[start_idx:end_idx]
    print(f"Chunk {chunk_id}: processing {len(chunk_files)} files (indices {start_idx}-{end_idx-1})")
    

    # Process just this chunk using your existing logic
    all_files = []
    for i in range(0, len(chunk_files), batch_size):
        batch = chunk_files[i:i+batch_size]
        wc_cmd = ['wc', '-l'] + batch
        wc_result = subprocess.run(wc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        for line in wc_result.stdout.strip().split('\n'):
            if line and 'total' not in line:
                path = line.lstrip().strip().split(" ")[-1]
                filename = os.path.basename(path).split("_")
                datetime = filename[2]
                run = filename[3]
                thisfilter = filename[4]
                choosechip = filename[5].replace("c","")
                sequencenumber = filename[7]
                amp = filename[8]
                x = filename[9]
                y = filename[10]
                parts = line.split()
                if len(parts) == 2:
                    count, path = parts
                    all_files.append({
                        'datetime': datetime,
                        'run': "%06d"%int(run),
                        'thisfilter': thisfilter,
                        'choosechip': "%02d"%int(choosechip),
                        'sequencenumber': "%04d"%int(sequencenumber),
                        'amp': amp,
                        'x': x,
                        'y': y,
                        'filename': os.path.basename(path),
                        'lines': int(count)-1,
                    })
    
    return pd.DataFrame(all_files)

def getfilesizeintodf(path='/lustre/lrspec/ZTF_data/lightcurves/1020/', batch_size=10000, name=None, chunk_id=None, total_chunks=None):
    """Main function - works with or without SLURM chunking"""
    if chunk_id is not None and total_chunks is not None:
        # SLURM mode: process only one chunk
        df = get_file_info_batch_chunked(path, chunk_id, total_chunks, batch_size, name)
        output_file = f"{config.ROOT}lightcurvefiles_info_chunk_{chunk_id:04d}.csv"
        df.to_csv(output_file, index=False)
        print(f"Chunk {chunk_id} saved to {output_file}")
    else:
        # Original mode: process everything
        df = get_file_info_batch(path, batch_size=batch_size, name=name)
        df.to_csv(config.ROOT + 'lightcurvefiles_info_slurm.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', type=int, help='Chunk ID for SLURM array')
    parser.add_argument('--total-chunks', type=int, help='Total number of chunks')
    args = parser.parse_args()
    
    getfilesizeintodf(
        path='/lustre/lrspec/ZTF_data/lightcurves/1020/',
        batch_size=10000,
        name='*.csv',
        chunk_id=args.chunk,
        total_chunks=args.total_chunks
    )
