"""
Data Loader Utilities
Functions for loading data into Spark DataFrames.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
import os
from pathlib import Path


def load_raw_data(spark: SparkSession, data_path: str, 
                  file_pattern: str = "*.parquet") -> DataFrame:
    """
    Load raw Parquet files into a single Spark DataFrame.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    data_path : str
        Directory path containing Parquet files
    file_pattern : str
        File pattern to match (default: "*.parquet")
    
    Returns:
    --------
    DataFrame
        Unified DataFrame containing all data from matching files
    """
    # Construct full path pattern
    full_pattern = os.path.join(data_path, file_pattern)
    
    print(f"Loading Parquet files from {data_path}...")
    print(f"Pattern: {file_pattern}")
    
    # Read all matching Parquet files
    # Spark automatically handles multiple files
    df = spark.read.parquet(data_path)
    
    # Get file count for logging
    file_count = len([f for f in os.listdir(data_path) 
                     if f.endswith('.parquet')])
    
    print(f"Loaded {file_count} Parquet file(s)")
    
    return df


def load_taxi_zones(spark: SparkSession, zones_path: str) -> DataFrame:
    """
    Load taxi zone lookup CSV into DataFrame.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    zones_path : str
        Path to taxi zone CSV file
    
    Returns:
    --------
    DataFrame
        DataFrame containing taxi zone information
    """
    if not os.path.exists(zones_path):
        raise FileNotFoundError(f"Taxi zone file not found: {zones_path}")
    
    print(f"Loading taxi zone lookup from {zones_path}...")
    
    # Read CSV with header
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(zones_path)
    
    print(f"Loaded {df.count()} taxi zones")
    
    return df


def print_schema_info(df: DataFrame, num_rows: int = 5):
    """
    Print DataFrame schema and sample rows.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to inspect
    num_rows : int
        Number of sample rows to show (default: 5)
    """
    print("\n" + "=" * 60)
    print("DataFrame Schema:")
    print("=" * 60)
    df.printSchema()
    
    print(f"\nTotal rows: {df.count():,}")
    
    print(f"\nSample rows (first {num_rows}):")
    print("=" * 60)
    df.show(num_rows, truncate=False)
    
    print("=" * 60 + "\n")


def get_dataframe_info(df: DataFrame) -> dict:
    """
    Get summary information about a DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame
    
    Returns:
    --------
    dict
        Dictionary with DataFrame information
    """
    info = {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": df.columns,
        "schema": df.schema.json()
    }
    
    return info


def load_processed_data(spark: SparkSession, processed_path: str) -> DataFrame:
    """
    Load processed/cleaned Parquet data.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    processed_path : str
        Path to processed Parquet files
    
    Returns:
    --------
    DataFrame
        Processed DataFrame
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data path not found: {processed_path}")
    
    print(f"Loading processed data from {processed_path}...")
    
    df = spark.read.parquet(processed_path)
    
    print(f"Loaded {df.count():,} rows")
    
    return df


def save_dataframe(df: DataFrame, output_path: str, 
                   mode: str = "overwrite", format: str = "parquet"):
    """
    Save DataFrame to disk.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to save
    output_path : str
        Output path
    mode : str
        Write mode: "overwrite", "append", "ignore", "error" (default: "overwrite")
    format : str
        Output format: "parquet", "csv", "json" (default: "parquet")
    """
    print(f"Saving DataFrame to {output_path}...")
    
    writer = df.write.mode(mode)
    
    if format == "parquet":
        writer.parquet(output_path)
    elif format == "csv":
        writer.option("header", "true").csv(output_path)
    elif format == "json":
        writer.json(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✓ DataFrame saved successfully")

