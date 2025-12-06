"""
Feature Engineering Script
Main script for creating temporal, spatial, and revenue features.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from config.data_config import (
    PROCESSED_DATA_PATH, TAXI_ZONE_PATH, TAXI_ZONE_FILENAME
)
from src.data_loader import load_processed_data, load_taxi_zones, print_schema_info, save_dataframe
from src.feature_engineer import (
    add_temporal_features, add_revenue_features,
    prepare_spatial_features, join_with_zones
)


def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC Taxi Feature Engineering")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session(app_name="NYC_Taxi_Feature_Engineering")
    
    try:
        # Load cleaned data
        print("\nStep 1: Loading cleaned data...")
        df = load_processed_data(spark, PROCESSED_DATA_PATH)
        
        print(f"Loaded {df.count():,} rows")
        
        # Add temporal features
        print("\nStep 2: Adding temporal features...")
        df = add_temporal_features(df)
        
        # Add revenue features
        print("\nStep 3: Adding revenue features...")
        df = add_revenue_features(df)
        
        # Load taxi zones if available
        print("\nStep 4: Loading taxi zone lookup...")
        zones_path = os.path.join(TAXI_ZONE_PATH, TAXI_ZONE_FILENAME)
        
        if os.path.exists(zones_path):
            zones_df = load_taxi_zones(spark, zones_path)
            
            # Join with zones
            print("\nStep 5: Joining with taxi zones...")
            df = join_with_zones(df, zones_df)
        else:
            print("⚠ Taxi zone file not found, skipping zone join")
        
        # Prepare spatial features for clustering
        print("\nStep 6: Preparing spatial features...")
        df = prepare_spatial_features(df, use_coordinates=True)
        
        # Cache the DataFrame (will be used multiple times)
        print("\nStep 7: Caching DataFrame...")
        df.cache()
        
        # Print feature-engineered data info
        print("\nStep 8: Feature-engineered data inspection...")
        print_schema_info(df, num_rows=5)
        
        # Save feature-engineered data
        print("\nStep 9: Saving feature-engineered data...")
        feature_output_path = os.path.join(PROCESSED_DATA_PATH, "with_features")
        save_dataframe(df, feature_output_path, mode="overwrite", format="parquet")
        
        print("\n" + "=" * 60)
        print("Feature engineering complete!")
        print("=" * 60)
        print(f"\nNew features added:")
        print("  Temporal: pickup_hour, pickup_dayofweek, pickup_month, pickup_year,")
        print("            is_weekend, is_rush_hour, trip_duration_minutes")
        print("  Revenue: total_revenue, revenue_per_minute, is_high_value")
        print("  Spatial: location_vector (for clustering)")
        
    except Exception as e:
        print(f"\n✗ Error during feature engineering: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

