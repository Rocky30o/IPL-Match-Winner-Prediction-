# src/download_data.py
"""
Data Download Script
--------------------
This script downloads the IPL Matches dataset from a reliable public GitHub repository
and saves it locally under the 'data/' directory.

This is the first step of our machine learning pipeline.
"""

import os
import urllib.request

def download_dataset():
    # URL of the raw IPL matches CSV file on GitHub (sourced from Kaggle originally)
    DATA_URL = "https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/matches.csv"
    
    # Define paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    output_path = os.path.join(data_dir, 'matches.csv')
    
    # Ensure the 'data/' directory exists
    print(f"[*] Ensuring directory exists: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"[*] Downloading dataset from: {DATA_URL}")
    try:
        # Download the file using urllib (standard Python library)
        urllib.request.urlretrieve(DATA_URL, output_path)
        print(f"[+] Dataset successfully downloaded and saved to: {output_path}")
        
        # Verify file size
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"[+] Downloaded file size: {file_size_kb:.2f} KB")
        return True
    except Exception as e:
        print(f"[-] Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
