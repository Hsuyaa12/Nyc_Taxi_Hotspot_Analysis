"""
Feature Engineering Utilities
Functions for creating temporal, spatial, and revenue features.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, dayofweek, month, year, when, unix_timestamp,
    percentile_approx
)
from pyspark.ml.feature import VectorAssembler


def add_temporal_features(df: DataFrame, 
                         pickup_col: str = "tpep_pickup_datetime") -> DataFrame:
    """
    Add temporal features from pickup datetime.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    pickup_col : str
        Name of pickup datetime column
    
    Returns:
    --------
    DataFrame
        DataFrame with temporal features added
    """
    print("Adding temporal features...")
    
    # Extract temporal components
    df_with_temporal = df \
        .withColumn("pickup_hour", hour(col(pickup_col))) \
        .withColumn("pickup_dayofweek", dayofweek(col(pickup_col))) \
        .withColumn("pickup_month", month(col(pickup_col))) \
        .withColumn("pickup_year", year(col(pickup_col))) \
        .withColumn("is_weekend", 
                   when(dayofweek(col(pickup_col)).isin([1, 7]), 1).otherwise(0)) \
        .withColumn("is_rush_hour",
                   when(
                       ((hour(col(pickup_col)).between(7, 9)) | 
                        (hour(col(pickup_col)).between(17, 19))) &
                       (dayofweek(col(pickup_col)).between(2, 6)),
                       1
                   ).otherwise(0))
    
    # Calculate trip duration in minutes
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"
    
    if dropoff_col in df.columns:
        df_with_temporal = df_with_temporal.withColumn(
            "trip_duration_minutes",
            (unix_timestamp(col(dropoff_col)) - unix_timestamp(col(pickup_col))) / 60
        )
    
    print("✓ Temporal features added")
    return df_with_temporal


def add_revenue_features(df: DataFrame) -> DataFrame:
    """
    Add revenue and profitability features.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    
    Returns:
    --------
    DataFrame
        DataFrame with revenue features added
    """
    print("Adding revenue features...")
    
    # Calculate total revenue (fare + tip)
    df_with_revenue = df
    
    if "fare_amount" in df.columns and "tip_amount" in df.columns:
        df_with_revenue = df_with_revenue.withColumn(
            "total_revenue",
            col("fare_amount") + col("tip_amount")
        )
    elif "fare_amount" in df.columns:
        df_with_revenue = df_with_revenue.withColumn(
            "total_revenue",
            col("fare_amount")
        )
    
    # Calculate revenue per minute
    if "total_revenue" in df_with_revenue.columns and "trip_duration_minutes" in df_with_revenue.columns:
        df_with_revenue = df_with_revenue.withColumn(
            "revenue_per_minute",
            when(col("trip_duration_minutes") > 0,
                 col("total_revenue") / col("trip_duration_minutes"))
            .otherwise(0)
        )
    
    # Calculate high-value trip flag (75th percentile)
    if "total_revenue" in df_with_revenue.columns:
        # Calculate 75th percentile
        percentile_value = df_with_revenue.select(
            percentile_approx("total_revenue", 0.75, 10000).alias("p75")
        ).collect()[0]["p75"]
        
        if percentile_value:
            df_with_revenue = df_with_revenue.withColumn(
                "is_high_value",
                when(col("total_revenue") >= percentile_value, 1).otherwise(0)
            )
            print(f"  High-value threshold: ${percentile_value:.2f}")
    
    print("✓ Revenue features added")
    return df_with_revenue


def prepare_spatial_features(df: DataFrame, 
                            use_coordinates: bool = True,
                            lat_col: str = "pickup_latitude",
                            lon_col: str = "pickup_longitude") -> DataFrame:
    """
    Prepare spatial features for clustering.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    use_coordinates : bool
        If True, use lat/lon directly; if False, use zone centroids
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    
    Returns:
    --------
    DataFrame
        DataFrame with location_vector column added
    """
    print("Preparing spatial features...")
    
    if use_coordinates and lat_col in df.columns and lon_col in df.columns:
        # Use coordinates directly
        # Filter to valid coordinates
        df_spatial = df.filter(
            col(lat_col).isNotNull() &
            col(lon_col).isNotNull() &
            (col(lat_col) != 0) &
            (col(lon_col) != 0)
        )
        
        # Create feature vector using VectorAssembler
        assembler = VectorAssembler(
            inputCols=[lat_col, lon_col],
            outputCol="location_vector"
        )
        
        df_with_vector = assembler.transform(df_spatial)
        
        print("✓ Spatial features prepared (using coordinates)")
        return df_with_vector
    else:
        print("⚠ Coordinates not available, spatial features not created")
        return df


def join_with_zones(df: DataFrame, zones_df: DataFrame,
                    location_id_col: str = "PULocationID") -> DataFrame:
    """
    Join trip data with taxi zone lookup to add borough/zone information.
    
    Parameters:
    -----------
    df : DataFrame
        Trip DataFrame
    zones_df : DataFrame
        Taxi zone lookup DataFrame
    location_id_col : str
        Location ID column name in trip DataFrame
    
    Returns:
    --------
    DataFrame
        Enriched DataFrame with zone information
    """
    print("Joining with taxi zones...")
    
    # Broadcast zones_df since it's small
    from pyspark.sql.functions import broadcast
    
    # Determine zone ID column name in zones_df
    zone_id_col = "LocationID" if "LocationID" in zones_df.columns else "locationid"
    
    if location_id_col in df.columns and zone_id_col in zones_df.columns:
        df_enriched = df.join(
            broadcast(zones_df),
            df[location_id_col] == zones_df[zone_id_col],
            "left"
        )
        print("✓ Joined with taxi zones")
        return df_enriched
    else:
        print("⚠ Zone columns not found, skipping join")
        return df


def create_feature_vector(df: DataFrame, feature_columns: list,
                          output_col: str = "features") -> DataFrame:
    """
    Create feature vector from multiple columns using VectorAssembler.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    feature_columns : list
        List of column names to combine into vector
    output_col : str
        Name of output vector column (default: "features")
    
    Returns:
    --------
    DataFrame
        DataFrame with feature vector column
    """
    # Filter to columns that exist
    existing_cols = [col for col in feature_columns if col in df.columns]
    
    if not existing_cols:
        raise ValueError(f"None of the specified columns exist: {feature_columns}")
    
    assembler = VectorAssembler(
        inputCols=existing_cols,
        outputCol=output_col
    )
    
    df_with_features = assembler.transform(df)
    
    return df_with_features

