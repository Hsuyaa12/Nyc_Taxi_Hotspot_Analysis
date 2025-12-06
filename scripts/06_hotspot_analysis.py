"""
Hotspot Analysis Script
Analyzes clusters to identify most profitable hotspots and their peak times.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql.functions import col, count, avg, sum as spark_sum, max as spark_max, collect_list, row_number
from pyspark.sql import Window

from config.spark_config import create_spark_session
from config.data_config import PROCESSED_DATA_PATH, MODELS_PATH, REPORTS_PATH
from src.data_loader import load_processed_data
from src.clustering import load_model, assign_clusters, get_cluster_centroids


def compute_cluster_metrics(df_with_clusters):
    """
    Compute metrics for each cluster.
    
    Parameters:
    -----------
    df_with_clusters : DataFrame
        DataFrame with cluster_id column
    
    Returns:
    --------
    DataFrame
        DataFrame with cluster metrics
    """
    print("Computing cluster metrics...")
    
    cluster_metrics = df_with_clusters.groupBy("cluster_id").agg(
        count("*").alias("trip_count"),
        avg("fare_amount").alias("avg_fare"),
        avg("total_revenue").alias("avg_revenue"),
        spark_sum("total_revenue").alias("total_revenue"),
        avg("trip_duration_minutes").alias("avg_duration_minutes"),
        avg("pickup_hour").alias("avg_hour")
    ).orderBy("cluster_id")
    
    print("✓ Cluster metrics computed")
    return cluster_metrics


def get_peak_hour_per_cluster(df_with_clusters):
    """
    Get peak hour (mode) for each cluster.
    
    Parameters:
    -----------
    df_with_clusters : DataFrame
        DataFrame with cluster_id and pickup_hour columns
    
    Returns:
    --------
    dict
        Dictionary mapping cluster_id to peak hour
    """
    print("Computing peak hours per cluster...")
    
    # Count trips by cluster and hour
    hour_counts = df_with_clusters.groupBy("cluster_id", "pickup_hour") \
        .agg(count("*").alias("count")) \
        .orderBy("cluster_id", col("count").desc())
    
    # Get peak hour for each cluster (hour with most trips)
    window = Window.partitionBy("cluster_id").orderBy(col("count").desc())
    peak_hours_df = hour_counts.withColumn("rank", 
        row_number().over(window)) \
        .filter(col("rank") == 1) \
        .select("cluster_id", "pickup_hour")
    
    peak_hours = {}
    for row in peak_hours_df.collect():
        peak_hours[row["cluster_id"]] = row["pickup_hour"]
    
    print("✓ Peak hours computed")
    return peak_hours


def rank_hotspots(cluster_metrics_df, ranking_method="revenue"):
    """
    Rank clusters by different criteria.
    
    Parameters:
    -----------
    cluster_metrics_df : DataFrame
        DataFrame with cluster metrics
    ranking_method : str
        Ranking method: "revenue", "count", "avg_fare", "combined"
    
    Returns:
    --------
    DataFrame
        Ranked DataFrame
    """
    print(f"Ranking hotspots by {ranking_method}...")
    
    if ranking_method == "revenue":
        ranked = cluster_metrics_df.orderBy(col("total_revenue").desc())
    elif ranking_method == "count":
        ranked = cluster_metrics_df.orderBy(col("trip_count").desc())
    elif ranking_method == "avg_fare":
        ranked = cluster_metrics_df.orderBy(col("avg_fare").desc())
    elif ranking_method == "combined":
        # Combined score: normalized trip_count * normalized avg_fare
        ranked = cluster_metrics_df.withColumn(
            "profitability_score",
            (col("trip_count") / cluster_metrics_df.agg(spark_max("trip_count")).collect()[0][0]) *
            (col("avg_fare") / cluster_metrics_df.agg(spark_max("avg_fare")).collect()[0][0])
        ).orderBy(col("profitability_score").desc())
    else:
        ranked = cluster_metrics_df
    
    print("✓ Hotspots ranked")
    return ranked


def analyze_temporal_patterns(df_with_clusters, top_clusters, n_top=5):
    """
    Analyze temporal patterns for top clusters.
    
    Parameters:
    -----------
    df_with_clusters : DataFrame
        DataFrame with cluster_id and temporal features
    top_clusters : list
        List of top cluster IDs
    n_top : int
        Number of top clusters to analyze
    
    Returns:
    --------
    dict
        Dictionary with temporal patterns per cluster
    """
    print(f"Analyzing temporal patterns for top {n_top} clusters...")
    
    top_cluster_list = top_clusters[:n_top]
    
    temporal_patterns = {}
    
    for cluster_id in top_cluster_list:
        cluster_data = df_with_clusters.filter(col("cluster_id") == cluster_id)
        
        # Hourly pattern
        hourly = cluster_data.groupBy("pickup_hour") \
            .agg(count("*").alias("count")) \
            .orderBy("pickup_hour") \
            .collect()
        
        hourly_pattern = {row["pickup_hour"]: row["count"] for row in hourly}
        
        # Day of week pattern
        daily = cluster_data.groupBy("pickup_dayofweek") \
            .agg(count("*").alias("count")) \
            .orderBy("pickup_dayofweek") \
            .collect()
        
        daily_pattern = {row["pickup_dayofweek"]: row["count"] for row in daily}
        
        temporal_patterns[cluster_id] = {
            "hourly": hourly_pattern,
            "daily": daily_pattern
        }
    
    print("✓ Temporal patterns analyzed")
    return temporal_patterns


def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC Taxi Hotspot Analysis")
    print("=" * 60)
    
    spark = create_spark_session(app_name="NYC_Taxi_Hotspot_Analysis")
    
    try:
        # Load feature-engineered data
        print("\nStep 1: Loading feature-engineered data...")
        feature_path = os.path.join(PROCESSED_DATA_PATH, "with_features")
        df = load_processed_data(spark, feature_path)
        
        # Load model
        print("\nStep 2: Loading K-Means model...")
        model_path = os.path.join(MODELS_PATH, "kmeans_model")
        model = load_model(spark, model_path)
        
        # Prepare clustering data and assign clusters
        print("\nStep 3: Assigning clusters...")
        clustering_df = df.select("location_vector").filter(col("location_vector").isNotNull())
        clustering_df = clustering_df.withColumnRenamed("location_vector", "features")
        
        clustered_locations = assign_clusters(model, clustering_df)
        
        # Join cluster assignments back to full DataFrame
        # We need to match on location_vector
        df_with_features = df.withColumnRenamed("location_vector", "features")
        df_with_clusters = df_with_features.join(
            clustered_locations.select("features", "cluster_id"),
            on="features",
            how="left"
        )
        
        # Compute cluster metrics
        print("\nStep 4: Computing cluster metrics...")
        cluster_metrics = compute_cluster_metrics(df_with_clusters)
        
        # Get peak hours
        print("\nStep 5: Computing peak hours...")
        peak_hours = get_peak_hour_per_cluster(df_with_clusters)
        
        # Rank hotspots
        print("\nStep 6: Ranking hotspots...")
        ranked_by_revenue = rank_hotspots(cluster_metrics, "revenue")
        ranked_by_count = rank_hotspots(cluster_metrics, "count")
        ranked_combined = rank_hotspots(cluster_metrics, "combined")
        
        # Get top clusters
        top_clusters_revenue = [row["cluster_id"] for row in ranked_by_revenue.head(10)]
        
        # Analyze temporal patterns
        print("\nStep 7: Analyzing temporal patterns...")
        temporal_patterns = analyze_temporal_patterns(df_with_clusters, top_clusters_revenue, n_top=5)
        
        # Get cluster centroids
        centroids = get_cluster_centroids(model)
        
        # Compile results
        results = {
            "top_hotspots_by_revenue": [
                {
                    "cluster_id": row["cluster_id"],
                    "trip_count": row["trip_count"],
                    "total_revenue": float(row["total_revenue"]),
                    "avg_fare": float(row["avg_fare"]),
                    "avg_revenue": float(row["avg_revenue"]),
                    "peak_hour": peak_hours.get(row["cluster_id"], None),
                    "centroid": centroids[row["cluster_id"]] if row["cluster_id"] < len(centroids) else None
                }
                for row in ranked_by_revenue.head(10)
            ],
            "top_hotspots_by_trip_count": [
                {
                    "cluster_id": row["cluster_id"],
                    "trip_count": row["trip_count"],
                    "total_revenue": float(row["total_revenue"]),
                    "avg_fare": float(row["avg_fare"])
                }
                for row in ranked_by_count.head(10)
            ],
            "temporal_patterns": temporal_patterns
        }
        
        # Save results
        results_path = os.path.join(REPORTS_PATH, "hotspot_analysis.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {results_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Top 10 Hotspots by Revenue:")
        print("=" * 60)
        for i, hotspot in enumerate(results["top_hotspots_by_revenue"], 1):
            print(f"{i}. Cluster {hotspot['cluster_id']}: "
                  f"{hotspot['trip_count']:,} trips, "
                  f"${hotspot['total_revenue']:,.2f} total revenue, "
                  f"${hotspot['avg_fare']:.2f} avg fare, "
                  f"Peak hour: {hotspot['peak_hour']}")
        
    except Exception as e:
        print(f"\n✗ Error during hotspot analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

