"""
Data Cleaning Utilities
Functions for data quality checks and cleaning operations using PySpark.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, isnan, isnull, percentile_approx
from pyspark.sql.types import TimestampType
import json


def check_data_quality(df: DataFrame) -> dict:
    """
    Perform data quality checks on DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to check
    
    Returns:
    --------
    dict
        Dictionary with quality metrics
    """
    print("Performing data quality checks...")
    
    total_rows = df.count()
    
    # Count nulls per column
    null_counts = {}
    for column_name in df.columns:
        # Use isnan only for numeric types, not timestamps
        col_type = dict(df.dtypes)[column_name]
        if col_type in ['double', 'float', 'int', 'bigint', 'long']:
            null_count = df.filter(col(column_name).isNull() | isnan(col(column_name))).count()
        else:
            null_count = df.filter(col(column_name).isNull()).count()
        null_counts[column_name] = {
            "count": null_count,
            "percentage": (null_count / total_rows * 100) if total_rows > 0 else 0
        }
    
    # Basic statistics for numeric columns only
    numeric_stats = {}
    numeric_types = ['int', 'bigint', 'long', 'double', 'float', 'decimal']
    dtypes_dict = dict(df.dtypes)
    
    for column_name in df.columns:
        col_type = dtypes_dict[column_name].lower()
        # Only process numeric columns
        if any(num_type in col_type for num_type in numeric_types):
            try:
                stats = df.select(
                    percentile_approx(col(column_name), [0.25, 0.5, 0.75], 10000).alias("percentiles")
                ).collect()[0]["percentiles"]
                
                numeric_stats[column_name] = {
                    "min": df.select(col(column_name)).agg({column_name: "min"}).collect()[0][0],
                    "max": df.select(col(column_name)).agg({column_name: "max"}).collect()[0][0],
                    "mean": df.select(col(column_name)).agg({column_name: "avg"}).collect()[0][0],
                    "percentiles": stats if stats else None
                }
            except Exception as e:
                # Skip columns that cause errors
                print(f"  Warning: Could not compute stats for {column_name}: {str(e)[:100]}")
    
    quality_metrics = {
        "total_rows": total_rows,
        "null_counts": null_counts,
        "numeric_stats": numeric_stats
    }
    
    return quality_metrics


def filter_invalid_coordinates(df: DataFrame, lat_col: str = "pickup_latitude",
                              lon_col: str = "pickup_longitude",
                              bounds: dict = None) -> DataFrame:
    """
    Filter rows where coordinates are outside NYC bounds.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    lat_col : str
        Latitude column name (default: "pickup_latitude")
    lon_col : str
        Longitude column name (default: "pickup_longitude")
    bounds : dict
        Dictionary with lat_min, lat_max, lon_min, lon_max
    
    Returns:
    --------
    DataFrame
        Filtered DataFrame
    """
    if bounds is None:
        bounds = {
            "latitude_min": 40.4,
            "latitude_max": 40.9,
            "longitude_min": -74.3,
            "longitude_max": -73.7
        }
    
    initial_count = df.count()
    
    # Filter coordinates within bounds and not null
    filtered_df = df.filter(
        col(lat_col).isNotNull() &
        col(lon_col).isNotNull() &
        col(lat_col).between(bounds["latitude_min"], bounds["latitude_max"]) &
        col(lon_col).between(bounds["longitude_min"], bounds["longitude_max"]) &
        (col(lat_col) != 0) &
        (col(lon_col) != 0)
    )
    
    filtered_count = filtered_df.count()
    removed = initial_count - filtered_count
    
    print(f"  Filtered invalid coordinates: Removed {removed:,} rows "
          f"({removed/initial_count*100:.2f}%)")
    
    return filtered_df


def filter_invalid_times(df: DataFrame, pickup_col: str = "tpep_pickup_datetime",
                         dropoff_col: str = "tpep_dropoff_datetime",
                         max_duration_hours: float = 24) -> DataFrame:
    """
    Filter rows with invalid timestamps or durations.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    pickup_col : str
        Pickup datetime column name
    dropoff_col : str
        Dropoff datetime column name
    max_duration_hours : float
        Maximum trip duration in hours (default: 24)
    
    Returns:
    --------
    DataFrame
        Filtered DataFrame
    """
    from pyspark.sql.functions import unix_timestamp
    
    initial_count = df.count()
    
    # Filter: not null, pickup < dropoff, duration < max
    filtered_df = df.filter(
        col(pickup_col).isNotNull() &
        col(dropoff_col).isNotNull() &
        (col(pickup_col) < col(dropoff_col)) &
        ((unix_timestamp(col(dropoff_col)) - unix_timestamp(col(pickup_col))) / 3600 <= max_duration_hours)
    )
    
    filtered_count = filtered_df.count()
    removed = initial_count - filtered_count
    
    print(f"  Filtered invalid times: Removed {removed:,} rows "
          f"({removed/initial_count*100:.2f}%)")
    
    return filtered_df


def filter_invalid_values(df: DataFrame, min_passengers: int = 0,
                          min_distance: float = 0, min_fare: float = 0,
                          max_fare: float = 500) -> DataFrame:
    """
    Filter rows with invalid numeric values.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    min_passengers : int
        Minimum passenger count (default: 0, allows package deliveries)
    min_distance : float
        Minimum trip distance (default: 0)
    min_fare : float
        Minimum fare amount (default: 0)
    max_fare : float
        Maximum fare amount (default: 500)
    
    Returns:
    --------
    DataFrame
        Filtered DataFrame
    """
    initial_count = df.count()
    
    # Build filter conditions
    conditions = []
    
    # Passenger count - allow 0 (package delivery), but filter negative
    if "passenger_count" in df.columns:
        conditions.append(
            col("passenger_count").isNotNull() &
            (col("passenger_count") >= 0)  # Allow 0 for package deliveries
        )
    
    # Trip distance and fare - allow 0 distance IF fare > 0 (cancellation/wait fee)
    # Otherwise require positive distance
    if "trip_distance" in df.columns and "fare_amount" in df.columns:
        conditions.append(
            col("trip_distance").isNotNull() &
            col("fare_amount").isNotNull() &
            (
                # Case 1: Normal trip (distance > 0 and fare > 0)
                ((col("trip_distance") > 0) & (col("fare_amount") > 0)) |
                # Case 2: Cancellation/wait fee (distance = 0 but fare > 0)
                ((col("trip_distance") == 0) & (col("fare_amount") > 0))
            ) &
            (col("trip_distance") >= 0)  # No negative distances
        )
    elif "fare_amount" in df.columns:
        # If only fare_amount exists (no trip_distance column)
        conditions.append(
            col("fare_amount").isNotNull() &
            (col("fare_amount") > 0) &  # Must be positive
            (col("fare_amount") >= min_fare) &
            (col("fare_amount") <= max_fare)
        )
    elif "trip_distance" in df.columns:
        # If only trip_distance exists (no fare_amount column)
        conditions.append(
            col("trip_distance").isNotNull() &
            (col("trip_distance") >= 0)  # Allow 0, no negative
        )
    
    # Apply all conditions by combining them with AND
    if conditions:
        from functools import reduce
        from operator import and_
        combined_condition = reduce(and_, conditions)
        filtered_df = df.filter(combined_condition)
    else:
        filtered_df = df
    
    filtered_count = filtered_df.count()
    removed = initial_count - filtered_count
    
    print(f"  Filtered invalid values: Removed {removed:,} rows "
          f"({removed/initial_count*100:.2f}%)")
    
    return filtered_df


def remove_duplicates(df: DataFrame, key_columns: list = None) -> DataFrame:
    """
    Remove duplicate records based on key columns.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    key_columns : list
        List of column names to use for deduplication.
        If None, uses all columns.
    
    Returns:
    --------
    DataFrame
        DataFrame with duplicates removed
    """
    initial_count = df.count()
    
    if key_columns is None:
        # Use all columns
        filtered_df = df.dropDuplicates()
    else:
        # Use specified key columns
        filtered_df = df.dropDuplicates(subset=key_columns)
    
    filtered_count = filtered_df.count()
    removed = initial_count - filtered_count
    
    if removed > 0:
        print(f"  Removed duplicates: Removed {removed:,} rows "
              f"({removed/initial_count*100:.2f}%)")
    
    return filtered_df


def clean_dataframe(df: DataFrame, bounds: dict = None,
                   max_duration_hours: float = 24,
                   max_fare: float = 500,
                   key_columns: list = None) -> DataFrame:
    """
    Master function that applies all cleaning steps in sequence.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    bounds : dict
        NYC coordinate bounds
    max_duration_hours : float
        Maximum trip duration
    max_fare : float
        Maximum fare amount
    key_columns : list
        Columns for deduplication
    
    Returns:
    --------
    DataFrame
        Cleaned DataFrame
    """
    initial_count = df.count()
    print(f"\nStarting data cleaning...")
    print(f"Initial row count: {initial_count:,}")
    
    # Apply cleaning steps
    cleaned_df = df
    
    # 1. Filter invalid coordinates
    if "pickup_latitude" in cleaned_df.columns and "pickup_longitude" in cleaned_df.columns:
        cleaned_df = filter_invalid_coordinates(cleaned_df, bounds=bounds)
    
    # 2. Filter invalid times
    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in cleaned_df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in cleaned_df.columns else "dropoff_datetime"
    
    if pickup_col in cleaned_df.columns and dropoff_col in cleaned_df.columns:
        cleaned_df = filter_invalid_times(cleaned_df, pickup_col, dropoff_col, max_duration_hours)
    
    # 3. Filter invalid values
    cleaned_df = filter_invalid_values(cleaned_df, max_fare=max_fare)
    
    # 4. Remove duplicates - ONLY if ALL columns are identical
    # This prevents false positives from multiple trips at same time/location
    print("  Removing exact duplicates (all columns must match)...")
    cleaned_df = remove_duplicates(cleaned_df, key_columns=None)  # Use all columns
    
    final_count = cleaned_df.count()
    total_removed = initial_count - final_count
    
    print(f"\nCleaning complete!")
    print(f"Final row count: {final_count:,}")
    print(f"Total rows removed: {total_removed:,} ({total_removed/initial_count*100:.2f}%)")
    
    return cleaned_df

