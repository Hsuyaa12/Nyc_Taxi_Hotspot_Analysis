"""
Data Cleaning Script
Main script for data preprocessing and cleaning pipeline.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from config.data_config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, REPORTS_PATH,
    NYC_BOUNDS, MAX_TRIP_DURATION_HOURS, MAX_FARE_AMOUNT
)
from src.data_loader import load_raw_data, print_schema_info, save_dataframe, get_dataframe_info
from src.data_cleaner import check_data_quality, clean_dataframe


def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC Taxi Data Cleaning Pipeline")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session(app_name="NYC_Taxi_Data_Cleaning")
    
    try:
        # Load raw data
        print("\nStep 1: Loading raw data...")
        raw_df = load_raw_data(spark, RAW_DATA_PATH)
        
        # Print initial schema and info
        print("\nStep 2: Initial data inspection...")
        print_schema_info(raw_df, num_rows=5)
        
        # Data quality checks
        print("\nStep 3: Data quality checks...")
        quality_metrics = check_data_quality(raw_df)
        
        # Save quality report
        quality_report_path = os.path.join(REPORTS_PATH, "data_quality_report.json")
        with open(quality_report_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        print(f"✓ Quality report saved to {quality_report_path}")
        
        # Clean data
        print("\nStep 4: Cleaning data...")
        cleaned_df = clean_dataframe(
            raw_df,
            bounds=NYC_BOUNDS,
            max_duration_hours=MAX_TRIP_DURATION_HOURS,
            max_fare=MAX_FARE_AMOUNT
        )
        
        # Print cleaned data info
        print("\nStep 5: Cleaned data inspection...")
        print_schema_info(cleaned_df, num_rows=5)
        
        # Save cleaned data
        print("\nStep 6: Saving cleaned data...")
        save_dataframe(cleaned_df, PROCESSED_DATA_PATH, mode="overwrite", format="parquet")
        
        # Save sample for documentation
        print("\nStep 7: Saving sample for documentation...")
        sample_df = cleaned_df.limit(10)
        sample_path = os.path.join(REPORTS_PATH, "cleaned_data_sample.csv")
        sample_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(sample_path.replace(".csv", ""))
        print(f"✓ Sample saved to {sample_path}")
        
        # Save summary statistics
        print("\nStep 8: Saving summary statistics...")
        summary = {
            "initial_row_count": raw_df.count(),
            "cleaned_row_count": cleaned_df.count(),
            "rows_removed": raw_df.count() - cleaned_df.count(),
            "removal_percentage": ((raw_df.count() - cleaned_df.count()) / raw_df.count() * 100) if raw_df.count() > 0 else 0,
            "quality_metrics": quality_metrics
        }
        
        summary_path = os.path.join(REPORTS_PATH, "cleaning_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Summary saved to {summary_path}")
        
        print("\n" + "=" * 60)
        print("Data cleaning complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during data cleaning: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

