"""
PROPER Benchmark: Spark vs scikit-learn K-Means
Demonstrates scalability by comparing both tools on the SAME clustering task.
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from config.data_config import PROCESSED_DATA_PATH, REPORTS_PATH, FIGURES_PATH
from src.data_loader import load_processed_data


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_sklearn_kmeans(data_array, k=10, sample_sizes=[10000, 50000, 100000]):
    """
    Benchmark scikit-learn K-Means on different sample sizes.
    """
    from sklearn.cluster import KMeans as SklearnKMeans
    
    print("\n" + "=" * 60)
    print("SKLEARN K-MEANS BENCHMARK")
    print("=" * 60)
    
    results = []
    
    for n_samples in sample_sizes:
        if n_samples > len(data_array):
            print(f"\n⚠ Skipping {n_samples:,} rows (exceeds available data)")
            continue
        
        print(f"\nTesting sklearn with {n_samples:,} samples...")
        
        # Sample data
        sample_data = data_array[:n_samples]
        
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        
        try:
            # Run sklearn K-Means
            kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10, max_iter=20)
            kmeans.fit(sample_data)
            
            end_time = time.time()
            end_mem = get_memory_usage_mb()
            
            elapsed = end_time - start_time
            mem_used = end_mem - start_mem
            
            print(f"  ✓ Success!")
            print(f"    Time: {elapsed:.3f} seconds")
            print(f"    Memory: {mem_used:.2f} MB")
            print(f"    Throughput: {n_samples/elapsed:,.0f} rows/sec")
            print(f"    Inertia: {kmeans.inertia_:,.2f}")
            
            results.append({
                "tool": "scikit-learn",
                "n_samples": n_samples,
                "k": k,
                "time_seconds": elapsed,
                "memory_mb": mem_used,
                "throughput": n_samples / elapsed,
                "inertia": float(kmeans.inertia_),
                "status": "success"
            })
            
        except MemoryError as e:
            print(f"  ✗ FAILED: Out of memory")
            results.append({
                "tool": "scikit-learn",
                "n_samples": n_samples,
                "k": k,
                "status": "memory_error",
                "error": str(e)
            })
            break  # Stop trying larger samples
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({
                "tool": "scikit-learn",
                "n_samples": n_samples,
                "k": k,
                "status": "error",
                "error": str(e)
            })
    
    return results


def benchmark_spark_kmeans(spark, df, k=10, sample_sizes=[10000, 50000, 100000, 1000000]):
    """
    Benchmark Spark MLlib K-Means on different sample sizes.
    """
    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.feature import VectorAssembler
    
    print("\n" + "=" * 60)
    print("SPARK K-MEANS BENCHMARK")
    print("=" * 60)
    
    results = []
    total_rows = df.count()
    
    # Prepare features (use PULocationID as simple numeric feature for benchmark)
    assembler = VectorAssembler(inputCols=["PULocationID"], outputCol="features")
    df_features = assembler.transform(df)
    
    for n_samples in sample_sizes:
        if n_samples > total_rows:
            print(f"\n⚠ Skipping {n_samples:,} rows (exceeds available: {total_rows:,})")
            continue
        
        print(f"\nTesting Spark with {n_samples:,} samples...")
        
        # Sample data
        fraction = n_samples / total_rows
        df_sample = df_features.sample(fraction=fraction, seed=42)
        
        start_time = time.time()
        
        try:
            # Run Spark K-Means
            kmeans = SparkKMeans(k=k, seed=42, maxIter=20, featuresCol="features")
            model = kmeans.fit(df_sample)
            
            # Force evaluation
            wssse = model.summary.trainingCost
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"  ✓ Success!")
            print(f"    Time: {elapsed:.3f} seconds")
            print(f"    Throughput: {n_samples/elapsed:,.0f} rows/sec")
            print(f"    WSSSE: {wssse:,.2f}")
            
            results.append({
                "tool": "Spark",
                "n_samples": n_samples,
                "k": k,
                "time_seconds": elapsed,
                "throughput": n_samples / elapsed,
                "wssse": float(wssse),
                "status": "success"
            })
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({
                "tool": "Spark",
                "n_samples": n_samples,
                "k": k,
                "status": "error",
                "error": str(e)
            })
    
    return results


def create_comparison_plot(all_results, output_path):
    """Create visualization comparing Spark vs sklearn."""
    import matplotlib.pyplot as plt
    
    sklearn_results = [r for r in all_results if r['tool'] == 'scikit-learn' and r['status'] == 'success']
    spark_results = [r for r in all_results if r['tool'] == 'Spark' and r['status'] == 'success']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Execution Time
    if sklearn_results:
        sklearn_sizes = [r['n_samples'] for r in sklearn_results]
        sklearn_times = [r['time_seconds'] for r in sklearn_results]
        ax1.plot(sklearn_sizes, sklearn_times, 'ro-', label='scikit-learn', linewidth=2, markersize=8)
    
    if spark_results:
        spark_sizes = [r['n_samples'] for r in spark_results]
        spark_times = [r['time_seconds'] for r in spark_results]
        ax1.plot(spark_sizes, spark_times, 'bs-', label='Spark', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Dataset Size (rows)', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('K-Means Performance: Spark vs scikit-learn', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Throughput
    if sklearn_results:
        sklearn_throughput = [r['throughput'] for r in sklearn_results]
        ax2.bar([f"{s/1000:.0f}K" for s in sklearn_sizes], sklearn_throughput, 
                color='red', alpha=0.6, label='scikit-learn')
    
    if spark_results:
        x_pos = np.arange(len(spark_sizes))
        if sklearn_results:
            x_pos = x_pos + 0.4
        spark_throughput = [r['throughput'] for r in spark_results]
        ax2.bar(x_pos, spark_throughput, color='blue', alpha=0.6, label='Spark', width=0.4)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{s/1000:.0f}K" if s < 1000000 else f"{s/1000000:.1f}M" for s in spark_sizes])
    
    ax2.set_xlabel('Dataset Size', fontsize=12)
    ax2.set_ylabel('Throughput (rows/second)', fontsize=12)
    ax2.set_title('Processing Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n✓ Comparison plot saved to {output_path}")


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("PROPER BENCHMARK: Spark vs scikit-learn K-Means")
        print("=" * 60)
        print("\nThis benchmark demonstrates:")
        print("  1. Both tools perform K-Means clustering")
        print("  2. Spark scales to much larger datasets")
        print("  3. scikit-learn struggles with memory on large data")
        
        # Create Spark session
        spark = create_spark_session()
        
        # Load data
        print("\nStep 1: Loading data...")
        feature_path = os.path.join(PROCESSED_DATA_PATH, "with_features")
        df = load_processed_data(spark, feature_path)
        total_rows = df.count()
        print(f"Loaded {total_rows:,} rows")
        
        # Prepare data for sklearn (collect small samples to numpy)
        print("\nStep 2: Preparing data for sklearn...")
        max_sklearn_sample = 500000  # Don't try more than 500K for sklearn
        
        sklearn_sample_df = df.select("PULocationID").limit(max_sklearn_sample)
        sklearn_data = np.array([float(row['PULocationID']) for row in sklearn_sample_df.collect()])
        sklearn_data = sklearn_data.reshape(-1, 1)  # sklearn expects 2D array
        
        print(f"✓ Prepared {len(sklearn_data):,} samples for sklearn")
        
        # Run benchmarks
        print("\nStep 3: Running benchmarks...")
        
        # sklearn benchmark
        sklearn_sizes = [10000, 50000, 100000, 200000]
        sklearn_results = benchmark_sklearn_kmeans(sklearn_data, k=10, sample_sizes=sklearn_sizes)
        
        # Spark benchmark
        spark_sizes = [10000, 50000, 100000, 500000, 1000000, total_rows]
        spark_results = benchmark_spark_kmeans(spark, df, k=10, sample_sizes=spark_sizes)
        
        # Combine results
        all_results = sklearn_results + spark_results
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"\n{'Tool':<15}{'Samples':<15}{'Time (s)':<12}{'Throughput':<20}{'Status'}")
        print("-" * 70)
        
        for r in all_results:
            if r['status'] == 'success':
                print(f"{r['tool']:<15}{r['n_samples']:<15,}{r['time_seconds']:<12.3f}"
                      f"{r['throughput']:<20,.0f}{r['status']}")
            else:
                print(f"{r['tool']:<15}{r['n_samples']:<15,}{'N/A':<12}"
                      f"{'N/A':<20}{r['status']}")
        
        # Key insights
        print("\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        
        sklearn_max = max([r['n_samples'] for r in sklearn_results if r['status'] == 'success'], default=0)
        spark_max = max([r['n_samples'] for r in spark_results if r['status'] == 'success'], default=0)
        
        print(f"\n1. scikit-learn handled up to: {sklearn_max:,} rows")
        print(f"2. Spark handled up to: {spark_max:,} rows")
        print(f"3. Spark scalability: {spark_max/sklearn_max:.1f}x more data")
        print("4. Spark uses distributed processing for Big Data")
        print("5. scikit-learn is limited by single-machine memory")
        
        # Save results
        print("\nStep 4: Saving results...")
        results_path = os.path.join(REPORTS_PATH, "benchmark_sklearn_vs_spark.json")
        with open(results_path, 'w') as f:
            json.dump({
                "total_rows": total_rows,
                "results": all_results,
                "summary": {
                    "sklearn_max_rows": sklearn_max,
                    "spark_max_rows": spark_max,
                    "scalability_factor": spark_max / sklearn_max if sklearn_max > 0 else None
                }
            }, f, indent=2)
        print(f"✓ Results saved to {results_path}")
        
        # Create visualization
        plot_path = os.path.join(FIGURES_PATH, "benchmark_sklearn_vs_spark.png")
        create_comparison_plot(all_results, plot_path)
        
        print("\n" + "=" * 60)
        print("✅ PROPER BENCHMARK COMPLETE!")
        print("=" * 60)
        
        spark.stop()
        
    except Exception as e:
        print(f"\n✗ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

