# NYC Taxi Trip Hotspot Identification Using Big Data Analytics

## Project Overview

This project develops an end-to-end Big Data analytics pipeline to identify the most profitable pickup locations and optimal times for New York City taxi drivers. By processing 84.6 million Yellow Taxi trip records from the NYC Taxi and Limousine Commission (TLC), the system uses K-Means clustering to discover geographic hotspots and temporal patterns that maximize driver earnings.

### Key Results

- **Data Processed**: 81.2 million trip records (100% of cleaned dataset)
- **Clusters Identified**: 20 geographic hotspots
- **Silhouette Score**: 0.7100 (excellent cluster separation)
- **Processing Time**: 6.3 minutes using Dask
- **Data Retention**: 99.15% after cleaning

### Primary Findings

| Hotspot | Location | Trip Count | Average Fare |
|---------|----------|------------|--------------|
| Cluster 0 | Midtown Manhattan | 13.6 million | $11.19 |
| Cluster 1 | JFK Airport | 3.2 million | $40.32 |
| Cluster 2 | LaGuardia Airport | 2.2 million | $31.50 |

**Actionable Insight**: Taxi drivers should position in Midtown Manhattan during weekday evening rush hours (6-9 PM) for maximum trip volume, or target airport runs for 3-4x higher fares.

---

## Project Structure

```
nyc-taxi-hotspot-analysis/
|
|-- config/                          # Configuration files
|   |-- __init__.py
|   |-- data_config.py               # Data paths and parameters
|   |-- spark_config.py              # Spark session configuration
|
|-- data/                            # Data directory
|   |-- raw/                         # Downloaded Parquet files (84.6M records)
|   |-- processed/                   # Cleaned and feature-engineered data
|   |-- taxi_zones/                  # Zone centroids and lookup tables
|
|-- deployment/                      # Streaming infrastructure
|   |-- docker-compose.yml           # Kafka and Zookeeper containers
|   |-- RUN_STREAMING_DEMO.sh        # Streaming demo script
|   |-- README.md                    # Deployment instructions
|
|-- docs/                            # Documentation
|   |-- EXECUTIVE_SUMMARY.md         # Project summary
|
|-- results/                         # Output files
|   |-- figures/                     # Visualizations (PNG files)
|   |   |-- elbow_curve_dask.png
|   |   |-- cluster_distribution_dask.png
|   |   |-- temporal_patterns_hourly.png
|   |   |-- benchmark_sklearn_vs_spark.png
|   |   |-- (10+ additional figures)
|   |
|   |-- models/                      # Trained K-Means model
|   |   |-- kmeans_model_proper/
|   |
|   |-- reports/                     # JSON statistics and metrics
|       |-- cluster_statistics_dask_full.json
|       |-- validation_results_dask.json
|       |-- benchmark_results.json
|       |-- elbow_results_dask.json
|
|-- scripts/                         # Pipeline execution scripts
|   |-- 01_data_acquisition.py       # Download NYC TLC data
|   |-- 02_data_cleaning.py          # Data quality and cleaning
|   |-- 03_feature_engineering.py    # Temporal and spatial features
|   |-- 05_clustering_locations.py   # Location-based analysis
|   |-- 05_clustering_proper_kmeans.py  # Spark K-Means implementation
|   |-- 06_hotspot_analysis.py       # Hotspot statistics
|   |-- 07_benchmark_sklearn_vs_spark.py  # Scalability comparison
|   |-- 08_visualizations_simple.py  # Generate charts
|   |-- 09_clustering_dask_full.py   # Dask K-Means (100% data)
|   |-- kafka_producer.py            # Streaming data producer
|   |-- streaming_consumer.py        # Real-time prediction consumer
|
|-- src/                             # Utility modules
|   |-- __init__.py
|   |-- clustering.py                # Clustering utilities
|   |-- data_cleaner.py              # Data cleaning functions
|   |-- data_loader.py               # Data loading utilities
|   |-- feature_engineer.py          # Feature engineering functions
|   |-- visualizations.py            # Plotting utilities
|
|-- main.py                          # Pipeline orchestrator
|-- requirements.txt                 # Python dependencies
|-- FINAL_PROJECT_REPORT.docx        # Final project report
|-- README.md                        # This file
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Distributed Computing | Apache Spark | 4.0.1 |
| Out-of-Core Processing | Dask | 2023.1.0+ |
| Machine Learning | Spark MLlib, Dask-ML, scikit-learn | - |
| Streaming | Apache Kafka | 3.6.0 |
| Containerization | Docker Compose | - |
| Programming Language | Python | 3.11+ |
| Visualization | Matplotlib, Seaborn | - |
| Data Format | Apache Parquet | - |

---

## Prerequisites

Before running the pipeline, ensure the following are installed:

1. **Python 3.11 or higher**
2. **Java 17 or higher** (required for Apache Spark)
3. **Docker** (optional, required only for streaming demo)

---

## Installation

### Step 1: Clone or Navigate to the Project Directory

```bash
cd /path/to/nyc-taxi-hotspot-analysis
```

### Step 2: Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The pipeline supports three execution modes:

### Option 1: Dask Mode (Recommended)

Processes 100% of the dataset using out-of-core computation. No sampling required.

```bash
python main.py --mode dask
```

**Expected Runtime**: Approximately 15-20 minutes (including data download)

### Option 2: Spark Mode

Uses Apache Spark with 35% data sampling for K-Means clustering.

```bash
export _JAVA_OPTIONS="-Djava.security.manager=allow"
python main.py --mode spark
```

**Expected Runtime**: Approximately 45 minutes

### Option 3: Both Modes

Runs both Dask and Spark implementations for comparison.

```bash
export _JAVA_OPTIONS="-Djava.security.manager=allow"
python main.py --mode both
```

---

## Pipeline Stages

The pipeline executes the following stages in sequence:

| Stage | Script | Description | Duration |
|-------|--------|-------------|----------|
| 1 | 01_data_acquisition.py | Downloads 12 months of NYC taxi data | 10 min |
| 2 | 02_data_cleaning.py | Removes nulls, outliers, duplicates | 5 min |
| 3 | 03_feature_engineering.py | Creates temporal and spatial features | 3 min |
| 4 | 05_clustering_locations.py | Location-based hotspot analysis | 2 min |
| 5 | 09_clustering_dask_full.py | K-Means clustering (Dask, 100% data) | 6 min |
| 6 | 07_benchmark_sklearn_vs_spark.py | Scalability comparison | 3 min |
| 7 | 08_visualizations_simple.py | Generates visualizations | 1 min |

---

## Running the Streaming Demo (Optional)

The project includes a Lambda Architecture implementation with Apache Kafka for real-time predictions.

### Step 1: Start Kafka Infrastructure

```bash
cd deployment
docker-compose up -d
```

### Step 2: Run the Kafka Producer

In one terminal:

```bash
python scripts/kafka_producer.py --max-records 1000
```

### Step 3: Run the Streaming Consumer

In a second terminal:

```bash
python scripts/streaming_consumer.py
```

### Step 4: Stop Kafka

```bash
cd deployment
docker-compose down
```

---

## Output Files

After pipeline execution, the following outputs are generated:

### Visualizations (results/figures/)

- elbow_curve_dask.png - K selection using Elbow Method
- cluster_distribution_dask.png - Trip distribution by cluster
- temporal_patterns_hourly.png - Hourly demand patterns
- temporal_patterns_daily.png - Daily demand patterns
- benchmark_sklearn_vs_spark.png - Scalability comparison

### Reports (results/reports/)

- cluster_statistics_dask_full.json - Cluster statistics (trip counts, fares)
- validation_results_dask.json - Silhouette score and validation metrics
- elbow_results_dask.json - Inertia values for different K values
- benchmark_results.json - Spark vs scikit-learn comparison

### Models (results/models/)

- kmeans_model_proper/ - Trained K-Means model (Spark format)

---

## Data Sources

| Source | URL | Format |
|--------|-----|--------|
| NYC TLC Trip Records | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page | Parquet |
| Taxi Zone Lookup | NYC Open Data | CSV |

The dataset includes 84.6 million Yellow Taxi trip records from January to December 2019.

---

## Methodology

### Data Preprocessing

1. **Null Removal**: Dropped rows with null values in critical columns (pickup datetime, location ID, fare amount)
2. **Outlier Filtering**: Removed negative fares, extreme values (fare > $500), trips longer than 24 hours
3. **Deduplication**: Removed exact duplicate rows
4. **Retention Rate**: 99.15% of original data retained

### Feature Engineering

- **Temporal Features**: pickup_hour, pickup_dayofweek, is_weekend, is_rush_hour
- **Spatial Features**: latitude and longitude derived from taxi zone centroids

### Machine Learning

- **Algorithm**: K-Means Clustering
- **Optimal K**: 20 (determined via Elbow Method)
- **Validation**: Train-test split with Silhouette Score evaluation
- **Implementation**: Dask MiniBatch K-Means for 100% data processing

---

## Benchmarking Results

| Dataset Size | scikit-learn | Spark | Dask |
|--------------|--------------|-------|------|
| 10K rows | 0.42s | 3.21s | 0.3s |
| 1M rows | 35.24s | 8.73s | 2.1s |
| 10M rows | Memory Error | 47.31s | 12.4s |
| 81.2M rows | N/A | 480s | 378s |

**Conclusion**: Distributed processing (Spark/Dask) is required for datasets exceeding 10 million rows.

---

## Authors

- Ayush Bhandari - Data Engineering and ML Pipeline
- Jaljala Shrestha Lama - Streaming and Infrastructure
- Aryan Kafle - Analysis and Documentation

---

## License

This project uses publicly available data from the NYC Taxi and Limousine Commission under the NYC Open Data Terms of Use.

---

## References

1. Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. Communications of the ACM, 59(11), 56-65.
2. Rocklin, M. (2015). Dask: Parallel computation with blocked algorithms and task scheduling. Proceedings of the 14th Python in Science Conference.
3. Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129-137.
4. NYC Taxi and Limousine Commission. (2024). TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
