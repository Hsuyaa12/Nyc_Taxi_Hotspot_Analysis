"""
NYC Taxi Hotspot Analysis - Main Pipeline
Runs the complete analysis from data acquisition to visualization.

Supports two processing modes:
1. Spark mode (default): Uses sampling (35%) for K-Means due to memory limits
2. Dask mode: Processes 100% of data using out-of-core computation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ COMPLETED: {description}")


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description='NYC Taxi Hotspot Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Modes:
  --mode spark  : Uses Spark with 35% sampling for K-Means (default)
  --mode dask   : Uses Dask to process 100% of data (no sampling)
  --mode both   : Runs both for comparison

Examples:
  python main.py                    # Run with Spark (default)
  python main.py --mode dask        # Run with Dask (100% data)
  python main.py --mode both        # Run both for comparison
        """
    )
    parser.add_argument(
        '--mode', 
        choices=['spark', 'dask', 'both'],
        default='spark',
        help='Processing mode: spark (sampling), dask (full data), or both'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download if already exists'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NYC TAXI HOTSPOT ANALYSIS - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"\nProcessing Mode: {args.mode.upper()}")
    
    if args.mode == 'spark':
        print("  → Using Spark with 35% sampling for K-Means")
    elif args.mode == 'dask':
        print("  → Using Dask with 100% data (no sampling)")
    else:
        print("  → Running both modes for comparison")
    
    print("\nPipeline Steps:")
    print("  1. Download NYC taxi data (12 months, 84.6M trips)")
    print("  2. Clean and validate data")
    print("  3. Engineer features")
    print("  4. Run hotspot analysis (aggregation method)")
    if args.mode in ['spark', 'both']:
        print("  5a. Run K-Means clustering (Spark - 35% sample)")
    if args.mode in ['dask', 'both']:
        print("  5b. Run K-Means clustering (Dask - 100% data)")
    print("  6. Run benchmark comparison (Spark vs sklearn)")
    print("  7. Generate visualizations")
    print("\n" + "=" * 70)
    
    # Get script directory
    script_dir = Path(__file__).parent / "scripts"
    
    # Define pipeline steps
    base_steps = [
        (script_dir / "01_data_acquisition.py", "Data Acquisition"),
        (script_dir / "02_data_cleaning.py", "Data Cleaning"),
        (script_dir / "03_feature_engineering.py", "Feature Engineering"),
        (script_dir / "05_clustering_locations.py", "Location-Based Hotspot Analysis"),
    ]
    
    # Clustering steps based on mode
    spark_clustering = (script_dir / "05_clustering_proper_kmeans.py", "Spark K-Means (35% sample)")
    dask_clustering = (script_dir / "09_clustering_dask_full.py", "Dask K-Means (100% data)")
    
    # Final steps
    final_steps = [
        (script_dir / "07_benchmark_sklearn_vs_spark.py", "Benchmark Comparison"),
        (script_dir / "08_visualizations_simple.py", "Visualizations"),
    ]
    
    # Build step list based on mode
    steps = base_steps.copy()
    
    if args.mode == 'spark':
        steps.append(spark_clustering)
    elif args.mode == 'dask':
        steps.append(dask_clustering)
    else:  # both
        steps.append(spark_clustering)
        steps.append(dask_clustering)
    
    steps.extend(final_steps)
    
    # Run each step
    for script_path, description in steps:
        if not script_path.exists():
            print(f"\n⚠ Warning: {script_path} not found, skipping...")
            continue
        
        run_script(str(script_path), description)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)
    print("\nResults:")
    print("  - Figures: results/figures/")
    print("  - Reports: results/reports/")
    print("  - Models: results/models/")
    
    if args.mode == 'spark':
        print("\nK-Means Results (Spark - 35% sample):")
        print("  - results/reports/cluster_statistics_proper.json")
    elif args.mode == 'dask':
        print("\nK-Means Results (Dask - 100% data):")
        print("  - results/reports/cluster_statistics_dask_full.json")
    else:
        print("\nK-Means Results Comparison:")
        print("  - Spark (35%): results/reports/cluster_statistics_proper.json")
        print("  - Dask (100%): results/reports/cluster_statistics_dask_full.json")


if __name__ == "__main__":
    main()
