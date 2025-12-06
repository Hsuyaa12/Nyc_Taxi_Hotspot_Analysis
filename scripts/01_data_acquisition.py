"""
Data Acquisition Script
Downloads NYC TLC Parquet files and taxi zone lookup data.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.data_config import (
    NYC_TLC_BASE_URL, TAXI_ZONE_URL, DATA_YEAR, DATA_MONTHS,
    RAW_DATA_PATH, SAMPLE_DATA_PATH, TAXI_ZONE_PATH,
    YELLOW_TAXI_PATTERN, TAXI_ZONE_FILENAME, SAMPLE_FRACTION
)
from config.spark_config import create_spark_session


def download_file(url, destination, chunk_size=8192, max_retries=3):
    """
    Download a single file from URL to destination with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    destination : str
        Local file path to save to
    chunk_size : int
        Size of chunks to download (default: 8KB)
    max_retries : int
        Maximum number of retry attempts (default: 3)
    
    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file was downloaded (check size)
            if os.path.getsize(destination) > 0:
                return True
            else:
                print(f"Warning: Downloaded file {destination} is empty")
                return False
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False
    
    return False


def download_nyc_taxi_data(year, months, output_dir):
    """
    Download NYC taxi trip data Parquet files for specified year and months.
    
    Parameters:
    -----------
    year : int
        Year to download (e.g., 2019)
    months : list
        List of month numbers (1-12)
    output_dir : str
        Directory to save downloaded files
    
    Returns:
    --------
    list
        List of successfully downloaded file paths
    """
    downloaded_files = []
    failed_files = []
    
    # Create list of URLs and destinations
    download_tasks = []
    for month in months:
        filename = YELLOW_TAXI_PATTERN.format(year=year, month=month)
        url = f"{NYC_TLC_BASE_URL}/{filename}"
        destination = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(destination):
            print(f"File {filename} already exists. Skipping download.")
            downloaded_files.append(destination)
        else:
            download_tasks.append((url, destination, filename))
    
    # Download files in parallel
    if download_tasks:
        print(f"Downloading {len(download_tasks)} files...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(download_file, url, dest): filename
                for url, dest, filename in download_tasks
            }
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        downloaded_files.append(
                            os.path.join(output_dir, filename)
                        )
                        print(f"✓ Successfully downloaded {filename}")
                    else:
                        failed_files.append(filename)
                except Exception as e:
                    print(f"✗ Error downloading {filename}: {e}")
                    failed_files.append(filename)
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successfully downloaded: {len(downloaded_files)} files")
    if failed_files:
        print(f"  Failed downloads: {len(failed_files)} files")
        print(f"  Failed files: {failed_files}")
    
    return downloaded_files


def download_taxi_zones(output_dir):
    """
    Download taxi zone lookup CSV file.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the file
    
    Returns:
    --------
    str
        Path to downloaded file, or None if failed
    """
    destination = os.path.join(output_dir, TAXI_ZONE_FILENAME)
    
    if os.path.exists(destination):
        print(f"Taxi zone file already exists. Skipping download.")
        return destination
    
    print(f"Downloading taxi zone lookup file...")
    success = download_file(TAXI_ZONE_URL, destination)
    
    if success:
        print(f"✓ Successfully downloaded taxi zone lookup")
        return destination
    else:
        print(f"✗ Failed to download taxi zone lookup")
        return None


def extract_sample(spark, input_path, output_path, sample_fraction=SAMPLE_FRACTION):
    """
    Extract a sample from Parquet file for local development.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    input_path : str
        Path to input Parquet file
    output_path : str
        Path to save sample Parquet file
    sample_fraction : float
        Fraction of data to sample (default: 0.01 = 1%)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"Extracting {sample_fraction*100}% sample from {os.path.basename(input_path)}...")
        df = spark.read.parquet(input_path)
        
        # Sample the data
        sample_df = df.sample(fraction=sample_fraction, seed=42)
        
        # Save sample
        sample_df.write.mode("overwrite").parquet(output_path)
        
        # Get count for verification
        count = sample_df.count()
        print(f"✓ Sample extracted: {count:,} rows saved to {output_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error extracting sample: {e}")
        return False


def verify_downloads(file_paths):
    """
    Verify that downloaded files exist and have reasonable sizes.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to verify
    
    Returns:
    --------
    dict
        Dictionary with verification results
    """
    results = {
        "total_files": len(file_paths),
        "existing_files": 0,
        "total_size_gb": 0,
        "missing_files": []
    }
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 0:
                results["existing_files"] += 1
                results["total_size_gb"] += size / (1024 ** 3)  # Convert to GB
            else:
                results["missing_files"].append(file_path)
        else:
            results["missing_files"].append(file_path)
    
    return results


def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC Taxi Data Acquisition")
    print("=" * 60)
    
    # Download taxi trip data
    print(f"\nDownloading Yellow Taxi data for year {DATA_YEAR}...")
    downloaded_files = download_nyc_taxi_data(
        year=DATA_YEAR,
        months=DATA_MONTHS,
        output_dir=RAW_DATA_PATH
    )
    
    # Download taxi zones
    print(f"\nDownloading taxi zone lookup...")
    zone_file = download_taxi_zones(TAXI_ZONE_PATH)
    
    # Verify downloads
    print(f"\nVerifying downloads...")
    verification = verify_downloads(downloaded_files)
    print(f"  Files downloaded: {verification['existing_files']}/{verification['total_files']}")
    print(f"  Total size: {verification['total_size_gb']:.2f} GB")
    
    if verification['missing_files']:
        print(f"  Warning: {len(verification['missing_files'])} files missing or empty")
    
    # Extract sample from first month's data
    if downloaded_files:
        print(f"\nExtracting sample for local development...")
        spark = create_spark_session()
        first_file = downloaded_files[0]
        sample_filename = f"sample_{os.path.basename(first_file)}"
        sample_path = os.path.join(SAMPLE_DATA_PATH, sample_filename)
        
        extract_sample(spark, first_file, sample_path)
        spark.stop()
    
    print("\n" + "=" * 60)
    print("Data acquisition complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

