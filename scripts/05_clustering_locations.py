"""
Location-Based Hotspot Analysis
Instead of K-Means clustering, analyze hotspots by aggregating trip data per LocationID.
This works better for data with LocationIDs but no GPS coordinates (2019+ data).
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql.functions import col, count as spark_count, avg, sum as spark_sum, desc
from config.spark_config import create_spark_session
from config.data_config import PROCESSED_DATA_PATH, REPORTS_PATH
from src.data_loader import load_processed_data, save_dataframe


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("NYC Taxi Location-Based Hotspot Analysis")
        print("=" * 60)
        
        # Create Spark session
        spark = create_spark_session()
        
        # Load feature-engineered data
        print("\nStep 1: Loading feature-engineered data...")
        feature_path = os.path.join(PROCESSED_DATA_PATH, "with_features")
        df = load_processed_data(spark, feature_path)
        print(f"Loaded {df.count():,} rows")
        
        # Aggregate by pickup location
        print("\nStep 2: Aggregating trips by pickup location...")
        location_stats = df.groupBy("PULocationID").agg(
            spark_count("*").alias("trip_count"),
            avg("fare_amount").alias("avg_fare"),
            spark_sum("total_revenue").alias("total_revenue"),
            avg("trip_distance").alias("avg_distance"),
            avg("trip_duration_minutes").alias("avg_duration"),
            spark_sum("is_high_value").alias("high_value_trips")
        ).orderBy(desc("trip_count"))
        
        print(f"✓ Aggregated data for {location_stats.count()} unique pickup locations")
        
        # Calculate rankings
        print("\nStep 3: Ranking hotspots...")
        
        # Add profit score (combination of volume and value)
        from pyspark.sql.functions import expr
        location_stats = location_stats.withColumn(
            "profit_score",
            expr("total_revenue * trip_count / 1000000")  # Normalize to millions
        )
        
        # Collect top locations
        top_by_trips = location_stats.orderBy(desc("trip_count")).limit(20).collect()
        top_by_revenue = location_stats.orderBy(desc("total_revenue")).limit(20).collect()
        top_by_profit = location_stats.orderBy(desc("profit_score")).limit(20).collect()
        
        # Print top hotspots
        print("\n" + "=" * 60)
        print("Top 10 Hotspots by Trip Count")
        print("=" * 60)
        print(f"{'Rank':<6}{'LocationID':<12}{'Trips':<12}{'Avg Fare':<12}{'Total Revenue':<15}")
        print("-" * 60)
        for i, row in enumerate(top_by_trips[:10], 1):
            print(f"{i:<6}{row['PULocationID']:<12}{row['trip_count']:<12,}"
                  f"${row['avg_fare']:<11.2f}${row['total_revenue']:<14,.2f}")
        
        print("\n" + "=" * 60)
        print("Top 10 Hotspots by Total Revenue")
        print("=" * 60)
        print(f"{'Rank':<6}{'LocationID':<12}{'Revenue':<15}{'Trips':<12}{'Avg Fare':<12}")
        print("-" * 60)
        for i, row in enumerate(top_by_revenue[:10], 1):
            print(f"{i:<6}{row['PULocationID']:<12}${row['total_revenue']:<14,.2f}"
                  f"{row['trip_count']:<12,}${row['avg_fare']:<11.2f}")
        
        # Analyze temporal patterns for top locations
        print("\n\nStep 4: Analyzing temporal patterns for top hotspots...")
        top_location_ids = [row['PULocationID'] for row in top_by_trips[:5]]
        
        temporal_patterns = {}
        for location_id in top_location_ids:
            # Hourly pattern
            hourly = df.filter(col("PULocationID") == location_id) \
                .groupBy("pickup_hour") \
                .agg(spark_count("*").alias("trips")) \
                .orderBy("pickup_hour") \
                .collect()
            
            # Day of week pattern
            daily = df.filter(col("PULocationID") == location_id) \
                .groupBy("pickup_dayofweek") \
                .agg(spark_count("*").alias("trips")) \
                .orderBy("pickup_dayofweek") \
                .collect()
            
            temporal_patterns[int(location_id)] = {
                "hourly": {int(row['pickup_hour']): int(row['trips']) for row in hourly},
                "daily": {int(row['pickup_dayofweek']): int(row['trips']) for row in daily}
            }
        
        print(f"✓ Analyzed temporal patterns for top {len(top_location_ids)} locations")
        
        # Save results
        print("\nStep 5: Saving results...")
        
        # Save location statistics
        hotspot_results = {
            "top_by_trips": [
                {
                    "rank": i + 1,
                    "location_id": int(row['PULocationID']),
                    "trip_count": int(row['trip_count']),
                    "avg_fare": float(row['avg_fare']),
                    "total_revenue": float(row['total_revenue']),
                    "avg_distance": float(row['avg_distance']),
                    "high_value_trips": int(row['high_value_trips'])
                }
                for i, row in enumerate(top_by_trips[:20])
            ],
            "top_by_revenue": [
                {
                    "rank": i + 1,
                    "location_id": int(row['PULocationID']),
                    "total_revenue": float(row['total_revenue']),
                    "trip_count": int(row['trip_count']),
                    "avg_fare": float(row['avg_fare'])
                }
                for i, row in enumerate(top_by_revenue[:20])
            ],
            "temporal_patterns": temporal_patterns
        }
        
        results_path = os.path.join(REPORTS_PATH, "hotspot_analysis.json")
        with open(results_path, 'w') as f:
            json.dump(hotspot_results, f, indent=2)
        print(f"✓ Results saved to {results_path}")
        
        # Save full location statistics as Parquet
        stats_path = os.path.join(PROCESSED_DATA_PATH, "location_statistics")
        save_dataframe(location_stats, stats_path)
        print(f"✓ Location statistics saved to {stats_path}")
        
        print("\n" + "=" * 60)
        print("Location-based hotspot analysis complete!")
        print("=" * 60)
        
        spark.stop()
        
    except Exception as e:
        print(f"\n✗ Error during hotspot analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

