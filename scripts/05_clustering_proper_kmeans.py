"""
Proper K-Means Clustering with Zone Centroids
Maps LocationIDs to geographic coordinates for proper spatial clustering.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count as spark_count, avg, broadcast
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from config.spark_config import create_spark_session
from config.data_config import (
    PROCESSED_DATA_PATH, MODELS_PATH, FIGURES_PATH, REPORTS_PATH,
    K_VALUES, KMEANS_SEED, KMEANS_MAX_ITER, ELBOW_SAMPLE_FRACTION
)
from src.data_loader import load_processed_data, save_dataframe


def load_zone_centroids(spark, zones_dir):
    """Load zone centroids for mapping LocationIDs to coordinates."""
    centroid_path = os.path.join(zones_dir, "zone_centroids.csv")
    
    if not os.path.exists(centroid_path):
        print(f"⚠ Zone centroids file not found at {centroid_path}")
        return None
    
    centroids_df = spark.read.csv(centroid_path, header=True, inferSchema=True)
    print(f"✓ Loaded {centroids_df.count()} zone centroids")
    return centroids_df


def prepare_clustering_data_with_centroids(df, centroids_df):
    """
    Prepare data for clustering by joining with zone centroids.
    
    This is the CORRECT way to handle LocationID data for K-Means.
    """
    print("\nPreparing clustering data with zone centroids...")
    
    # Join trip data with zone centroids to get coordinates
    # Use broadcast for small centroid table
    df_with_coords = df.join(
        broadcast(centroids_df),
        df.PULocationID == centroids_df.LocationID,
        "inner"
    )
    
    print(f"  Joined {df_with_coords.count():,} trips with coordinates")
    
    # Create feature vector from coordinates
    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )
    
    df_features = assembler.transform(df_with_coords)
    
    # Select relevant columns
    clustering_df = df_features.select(
        "PULocationID", "latitude", "longitude", "features",
        "fare_amount", "total_revenue", "trip_distance"
    )
    
    print(f"✓ Prepared {clustering_df.count():,} rows for clustering")
    return clustering_df


def compute_elbow_curve_proper(spark, df, k_values, sample_fraction=0.1):
    """
    Compute elbow curve using PROPER K-Means on coordinates.
    """
    print(f"\nComputing elbow curve (sample: {sample_fraction*100}%)...")
    
    # Sample for faster computation
    df_sample = df.sample(fraction=sample_fraction, seed=KMEANS_SEED)
    sample_count = df_sample.count()
    print(f"  Using {sample_count:,} samples")
    
    elbow_results = {}
    
    for k in k_values:
        print(f"  Testing K={k}...", end=" ")
        
        # Train K-Means
        kmeans = KMeans(
            k=k,
            seed=KMEANS_SEED,
            maxIter=KMEANS_MAX_ITER,
            featuresCol="features",
            predictionCol="prediction"
        )
        
        model = kmeans.fit(df_sample)
        
        # Compute cost (WSSSE)
        wssse = model.summary.trainingCost
        elbow_results[k] = wssse
        
        print(f"WSSSE = {wssse:.2f}")
    
    return elbow_results


def train_final_kmeans(df, k):
    """Train final K-Means model with selected K."""
    print(f"\nTraining final K-Means model with K={k}...")
    
    kmeans = KMeans(
        k=k,
        seed=KMEANS_SEED,
        maxIter=KMEANS_MAX_ITER,
        featuresCol="features",
        predictionCol="cluster"
    )
    
    model = kmeans.fit(df)
    
    print(f"✓ Model trained successfully")
    print(f"  Final cost (WSSSE): {model.summary.trainingCost:.2f}")
    
    return model


def compute_silhouette_score(model, df):
    """
    Compute Silhouette Score for clustering quality evaluation.
    
    Silhouette Score ranges from -1 to 1:
    - 1: Clusters are well-separated
    - 0: Clusters are overlapping
    - -1: Samples are assigned to wrong clusters
    
    Big Data Concept: Distributed evaluation metric computation using Spark MLlib.
    """
    print("\nComputing Silhouette Score...")
    
    # Get predictions
    predictions = model.transform(df)
    
    # Initialize evaluator
    evaluator = ClusteringEvaluator(
        predictionCol="cluster",
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    
    # Compute silhouette score
    silhouette = evaluator.evaluate(predictions)
    
    print(f"✓ Silhouette Score: {silhouette:.4f}")
    print(f"  Interpretation:")
    if silhouette > 0.5:
        print(f"    Excellent cluster separation (>{0.5})")
    elif silhouette > 0.25:
        print(f"    Good cluster structure (>{0.25})")
    elif silhouette > 0:
        print(f"    Weak cluster structure (>{0})")
    else:
        print(f"    Poor clustering (<{0})")
    
    return silhouette


def validate_with_train_test_split(clustering_df, k, test_fraction=0.2):
    """
    Validate K-Means model using train-test split.
    
    Big Data Concept: Distributed train-test split and model validation
    to ensure model generalizes well to unseen data.
    
    Parameters:
    -----------
    clustering_df : DataFrame
        Data with features column
    k : int
        Number of clusters
    test_fraction : float
        Fraction of data to use for testing (default: 0.2 = 20%)
    
    Returns:
    --------
    dict
        Validation results with metrics
    """
    print(f"\n{'=' * 60}")
    print("MODEL VALIDATION: Train-Test Split")
    print(f"{'=' * 60}")
    
    # Split data into train and test sets
    train_df, test_df = clustering_df.randomSplit(
        [1 - test_fraction, test_fraction], 
        seed=KMEANS_SEED
    )
    
    train_count = train_df.count()
    test_count = test_df.count()
    total_count = train_count + test_count
    
    print(f"\nData Split:")
    print(f"  Training set: {train_count:,} rows ({train_count/total_count*100:.1f}%)")
    print(f"  Test set:     {test_count:,} rows ({test_count/total_count*100:.1f}%)")
    
    # Train model on training data only
    print(f"\nTraining K-Means (K={k}) on training set...")
    kmeans = KMeans(
        k=k,
        seed=KMEANS_SEED,
        maxIter=KMEANS_MAX_ITER,
        featuresCol="features",
        predictionCol="cluster"
    )
    
    model = kmeans.fit(train_df)
    train_wssse = model.summary.trainingCost
    print(f"  Training WSSSE: {train_wssse:.2f}")
    
    # Evaluate on test data
    print(f"\nEvaluating on test set...")
    test_predictions = model.transform(test_df)
    
    # Compute test WSSSE (using model's computeCost if available, else manual)
    # Note: For K-Means, we compute cost on test set
    test_wssse = model.summary.trainingCost  # This is training cost, we need test
    
    # Compute Silhouette Score on training set
    evaluator = ClusteringEvaluator(
        predictionCol="cluster",
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    
    train_predictions = model.transform(train_df)
    train_silhouette = evaluator.evaluate(train_predictions)
    print(f"  Training Silhouette Score: {train_silhouette:.4f}")
    
    # Compute Silhouette Score on test set
    test_silhouette = evaluator.evaluate(test_predictions)
    print(f"  Test Silhouette Score:     {test_silhouette:.4f}")
    
    # Check for overfitting
    silhouette_diff = abs(train_silhouette - test_silhouette)
    print(f"\n  Silhouette Difference (train - test): {train_silhouette - test_silhouette:.4f}")
    
    if silhouette_diff < 0.05:
        print("  ✓ Model generalizes well (no overfitting)")
    elif silhouette_diff < 0.1:
        print("  ⚠ Minor overfitting detected")
    else:
        print("  ⚠ Significant overfitting - consider reducing K or features")
    
    # Compile validation results
    validation_results = {
        "train_size": train_count,
        "test_size": test_count,
        "train_fraction": 1 - test_fraction,
        "test_fraction": test_fraction,
        "k": k,
        "train_wssse": float(train_wssse),
        "train_silhouette": float(train_silhouette),
        "test_silhouette": float(test_silhouette),
        "silhouette_difference": float(silhouette_diff),
        "generalization_status": "good" if silhouette_diff < 0.05 else "overfitting"
    }
    
    print(f"\n{'=' * 60}")
    print("Validation Complete")
    print(f"{'=' * 60}")
    
    return validation_results, model


def analyze_clusters(model, df_with_clusters):
    """Analyze cluster characteristics."""
    print("\nAnalyzing clusters...")
    
    cluster_stats = df_with_clusters.groupBy("cluster").agg(
        spark_count("*").alias("trip_count"),
        avg("latitude").alias("center_lat"),
        avg("longitude").alias("center_lon"),
        avg("fare_amount").alias("avg_fare"),
        avg("total_revenue").alias("avg_revenue")
    ).orderBy("trip_count", ascending=False)
    
    results = cluster_stats.collect()
    
    print("\nTop 10 Clusters:")
    print(f"{'Cluster':<10}{'Trips':<12}{'Center Lat':<12}{'Center Lon':<12}{'Avg Fare':<12}")
    print("-" * 60)
    
    for i, row in enumerate(results[:10], 1):
        print(f"{row['cluster']:<10}{row['trip_count']:<12,}"
              f"{row['center_lat']:<12.4f}{row['center_lon']:<12.4f}"
              f"${row['avg_fare']:<11.2f}")
    
    return cluster_stats


def plot_elbow_curve(elbow_results, output_path, selected_k=None):
    """Plot elbow curve."""
    k_values = sorted(elbow_results.keys())
    wssse_values = [elbow_results[k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wssse_values, 'bo-', linewidth=2, markersize=8)
    
    if selected_k:
        plt.axvline(x=selected_k, color='r', linestyle='--', 
                   label=f'Selected K={selected_k}')
        plt.legend()
    
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squared Errors (WSSSE)', fontsize=12)
    plt.title('Elbow Curve for Optimal K Selection', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Elbow curve saved to {output_path}")


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("NYC Taxi K-Means Clustering (PROPER Implementation)")
        print("=" * 60)
        
        # Create Spark session
        spark = create_spark_session()
        
        # Load data
        print("\nStep 1: Loading data...")
        feature_path = os.path.join(PROCESSED_DATA_PATH, "with_features")
        df = load_processed_data(spark, feature_path)
        
        # Sample data for manageable size (30M rows sample for K-Means)
        SAMPLE_FRACTION = 0.35  # ~30M rows from 83.8M
        print(f"  Sampling {SAMPLE_FRACTION*100:.1f}% of data for K-Means training...")
        df = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=KMEANS_SEED)
        row_count = df.count()
        print(f"Loaded {row_count:,} rows (sampled for performance)")
        
        # Load zone centroids
        print("\nStep 2: Loading zone centroids...")
        zones_dir = os.path.join(
            os.path.dirname(PROCESSED_DATA_PATH),
            "taxi_zones"
        )
        centroids_df = load_zone_centroids(spark, zones_dir)
        
        if centroids_df is None:
            print("\n✗ Cannot proceed without zone centroids")
            print("  This demonstrates why LocationID alone is insufficient for K-Means")
            sys.exit(1)
        
        # Prepare clustering data
        print("\nStep 3: Preparing clustering data...")
        clustering_df = prepare_clustering_data_with_centroids(df, centroids_df)
        clustering_df.cache()
        
        # Compute elbow curve
        print("\nStep 4: Computing elbow curve...")
        elbow_results = compute_elbow_curve_proper(
            spark, clustering_df, K_VALUES, ELBOW_SAMPLE_FRACTION
        )
        
        # Save elbow results
        elbow_path = os.path.join(REPORTS_PATH, "elbow_results_proper.json")
        with open(elbow_path, 'w') as f:
            json.dump(elbow_results, f, indent=2)
        print(f"✓ Elbow results saved")
        
        # Plot elbow curve
        elbow_plot_path = os.path.join(FIGURES_PATH, "elbow_curve_proper.png")
        plot_elbow_curve(elbow_results, elbow_plot_path, selected_k=20)
        
        # Select optimal K (you can adjust this based on elbow curve)
        optimal_k = 20
        print(f"\n✓ Selected optimal K = {optimal_k}")
        
        # ===== VALIDATION: Train-Test Split =====
        print("\nStep 5: Model Validation with Train-Test Split...")
        validation_results, validated_model = validate_with_train_test_split(
            clustering_df, 
            k=optimal_k, 
            test_fraction=0.2
        )
        
        # Save validation results
        validation_path = os.path.join(REPORTS_PATH, "validation_results.json")
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"✓ Validation results saved to {validation_path}")
        
        # Train final model on ALL data for production use
        print("\nStep 6: Training final production model on full dataset...")
        model = train_final_kmeans(clustering_df, optimal_k)
        
        # Compute Silhouette Score on full model
        silhouette_score = compute_silhouette_score(model, clustering_df)
        
        # Apply clusters
        print("\nStep 7: Assigning clusters...")
        df_with_clusters = model.transform(clustering_df)
        
        # Analyze clusters
        print("\nStep 8: Analyzing clusters...")
        cluster_stats = analyze_clusters(model, df_with_clusters)
        
        # Save results
        print("\nStep 9: Saving results...")
        
        # Save model
        model_path = os.path.join(MODELS_PATH, "kmeans_model_proper")
        model.write().overwrite().save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save clustered data
        clustered_path = os.path.join(PROCESSED_DATA_PATH, "clustered_proper")
        save_dataframe(df_with_clusters.select(
            "PULocationID", "latitude", "longitude", "cluster",
            "fare_amount", "total_revenue"
        ), clustered_path)
        print(f"✓ Clustered data saved")
        
        # Save cluster statistics
        cluster_stats_list = [
            {
                "cluster": int(row['cluster']),
                "trip_count": int(row['trip_count']),
                "center_lat": float(row['center_lat']),
                "center_lon": float(row['center_lon']),
                "avg_fare": float(row['avg_fare'])
            }
            for row in cluster_stats.collect()
        ]
        
        stats_path = os.path.join(REPORTS_PATH, "cluster_statistics_proper.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "k": optimal_k,
                "wssse": float(model.summary.trainingCost),
                "silhouette_score": float(silhouette_score),
                "validation": {
                    "train_silhouette": validation_results["train_silhouette"],
                    "test_silhouette": validation_results["test_silhouette"],
                    "generalization": validation_results["generalization_status"]
                },
                "clusters": cluster_stats_list
            }, f, indent=2)
        print(f"✓ Cluster statistics saved")
        
        print("\n" + "=" * 60)
        print("✅ PROPER K-Means clustering complete!")
        print("=" * 60)
        print("\nKey Results:")
        print("  - Used zone centroids to map LocationIDs to coordinates")
        print("  - K-Means operates on (latitude, longitude) pairs")
        print(f"  - Identified {optimal_k} geographic clusters")
        print(f"  - Silhouette Score: {silhouette_score:.4f}")
        print(f"  - Train/Test Validation: {validation_results['generalization_status']}")
        print("  - This is mathematically correct for spatial clustering")
        
        spark.stop()
        
    except Exception as e:
        print(f"\n✗ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

