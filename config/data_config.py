"""
Data Configuration
Centralizes all data paths, URLs, and parameters for the NYC Taxi analysis.
"""

import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NYC TLC Data URLs
NYC_TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.csv"

# Alternative S3 URLs (if using AWS CLI)
NYC_TLC_S3_BUCKET = "s3://nyc-tlc/trip data"
TAXI_ZONE_S3_URL = "s3://nyc-tlc/misc/taxi_zones.csv"

# Data Configuration
DATA_YEAR = 2019  # Year to download (2019 has good data volume >2GB)
DATA_MONTHS = list(range(1, 13))  # All 12 months for full year analysis

# File naming patterns
YELLOW_TAXI_PATTERN = "yellow_tripdata_{year}-{month:02d}.parquet"
TAXI_ZONE_FILENAME = "taxi_zones.csv"

# Directory Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "samples")
TAXI_ZONE_PATH = os.path.join(BASE_DIR, "data", "taxi_zones")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
FIGURES_PATH = os.path.join(BASE_DIR, "results", "figures")
MODELS_PATH = os.path.join(BASE_DIR, "results", "models")
REPORTS_PATH = os.path.join(BASE_DIR, "results", "reports")

# NYC Geographic Bounds
NYC_BOUNDS = {
    "latitude_min": 40.4,
    "latitude_max": 40.9,
    "longitude_min": -74.3,
    "longitude_max": -73.7
}

# Clustering Parameters
K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]  # K values to test for elbow method
OPTIMAL_K = None  # Will be set after elbow method analysis (typically 20-40)
KMEANS_SEED = 42  # Random seed for reproducibility
KMEANS_MAX_ITER = 20  # Maximum iterations for K-Means
KMEANS_TOL = 1e-4  # Convergence tolerance

# Sampling Parameters
SAMPLE_FRACTION = 0.01  # 1% sample for local development
ELBOW_SAMPLE_FRACTION = 0.1  # 10% sample for elbow method (faster computation)

# Data Quality Thresholds
MAX_TRIP_DURATION_HOURS = 24  # Remove trips longer than 24 hours
MAX_FARE_AMOUNT = 500  # Remove trips with fare > $500
MIN_PASSENGERS = 1  # Minimum passenger count
MIN_DISTANCE = 0  # Minimum trip distance (0 means no minimum, but filter negative)
MIN_FARE = 0  # Minimum fare (0 means no minimum, but filter negative)

# Feature Engineering
HIGH_VALUE_PERCENTILE = 75  # Percentile for high-value trip flag

# Benchmarking Parameters
BENCHMARK_SAMPLE_SIZES = [10000, 50000, 100000, 500000, 1000000]  # Sample sizes for benchmarking

# Create directories if they don't exist
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLE_DATA_PATH, 
             TAXI_ZONE_PATH, RESULTS_PATH, FIGURES_PATH, MODELS_PATH, REPORTS_PATH]:
    os.makedirs(path, exist_ok=True)

