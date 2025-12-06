"""
Spark Structured Streaming Consumer - Real-Time Hotspot Prediction
Big Data Concepts:
- Lambda Architecture: Combines batch (historical K-Means) with speed layer (streaming)
- Structured Streaming: Micro-batch processing with exactly-once semantics
- Stream-Static Join: Joining streaming data with static reference data (zone centroids)
- Model Serving: Using pre-trained ML model for real-time inference
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, IntegerType
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler

from config.data_config import MODELS_PATH


def create_streaming_spark_session():
    """
    Create Spark session with Kafka dependencies.
    
    Big Data Concept: Spark Structured Streaming enables scalable stream processing
    with the same DataFrame API used for batch processing.
    """
    spark = SparkSession.builder \
        .appName("TaxiHotspotStreamingPredictor") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0,org.apache.kafka:kafka-clients:3.4.1") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
        .config("spark.sql.streaming.schemaInference", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark Streaming session created with Kafka support")
    return spark


def define_trip_schema():
    """
    Define schema for incoming taxi trip JSON data.
    
    Big Data Concept: Schema-on-read allows parsing of semi-structured JSON data
    into structured DataFrames for SQL operations.
    """
    return StructType([
        StructField("VendorID", LongType(), True),
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("passenger_count", DoubleType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("RatecodeID", DoubleType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("PULocationID", LongType(), True),
        StructField("DOLocationID", LongType(), True),
        StructField("payment_type", LongType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("congestion_surcharge", DoubleType(), True),
        StructField("airport_fee", IntegerType(), True)
    ])


def load_zone_centroids_static(spark, zones_dir):
    """
    Load zone centroids as static DataFrame for stream-static join.
    
    Big Data Concept: Stream-Static Join - Enrich streaming data with static
    reference data (zone coordinates) for real-time feature engineering.
    """
    centroid_path = os.path.join(zones_dir, "zone_centroids.csv")
    
    if not os.path.exists(centroid_path):
        print(f"⚠ Zone centroids not found at {centroid_path}")
        return None
    
    centroids_df = spark.read.csv(centroid_path, header=True, inferSchema=True)
    print(f"✓ Loaded {centroids_df.count()} zone centroids (static reference data)")
    return centroids_df


def load_kmeans_model(spark, model_path):
    """
    Load pre-trained K-Means model for real-time prediction.
    
    Big Data Concept: Model Serving - Deploying batch-trained models
    for real-time inference on streaming data (Lambda Architecture).
    """
    if not os.path.exists(model_path):
        print(f"⚠ K-Means model not found at {model_path}")
        print("  Run the batch clustering first: python scripts/05_clustering_proper_kmeans.py")
        return None
    
    model = KMeansModel.load(model_path)
    print(f"✓ Loaded pre-trained K-Means model (K={model.getK()})")
    return model


def process_stream(spark, kafka_servers='localhost:9092', topic='taxi-trips'):
    """
    Main streaming processing logic.
    
    Big Data Concepts:
    - Micro-batch Processing: Spark Streaming processes data in small batches
    - Watermarking: Handles late-arriving data (not used here but important for production)
    - Stateful Aggregations: Maintains running counts across micro-batches
    """
    print(f"\n{'=' * 70}")
    print("Starting Taxi Hotspot Real-Time Predictor")
    print(f"{'=' * 70}")
    print(f"  Kafka server: {kafka_servers}")
    print(f"  Topic: {topic}")
    print(f"  Processing mode: Micro-batch (Structured Streaming)")
    print(f"{'=' * 70}\n")
    
    # Load static reference data (zone centroids)
    zones_dir = os.path.join(
        Path(__file__).parent.parent,
        "data", "taxi_zones"
    )
    centroids_df = load_zone_centroids_static(spark, zones_dir)
    
    if centroids_df is None:
        print("✗ Cannot proceed without zone centroids")
        sys.exit(1)
    
    # Load pre-trained K-Means model
    model_path = os.path.join(MODELS_PATH, "kmeans_model_proper")
    model = load_kmeans_model(spark, model_path)
    
    if model is None:
        print("\n⚠ K-Means model not available - running without clustering")
        print("  Stream will show aggregations by LocationID instead")
        use_model = False
    else:
        use_model = True
    
    # Define schema for incoming JSON data
    trip_schema = define_trip_schema()
    
    # Read from Kafka stream
    print("\n📡 Connecting to Kafka stream...")
    df_stream = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", topic) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    print("✓ Connected to Kafka stream")
    
    # Parse JSON value
    print("  Parsing JSON messages...")
    df_parsed = df_stream.select(
        from_json(col("value").cast("string"), trip_schema).alias("data")
    ).select("data.*")
    
    if use_model:
        print("\n🤖 Streaming with K-Means Prediction:")
        print("  1. Join with zone centroids to get coordinates")
        print("  2. Apply pre-trained K-Means model")
        print("  3. Aggregate predictions per cluster")
        
        # Stream-Static Join: Add coordinates to streaming data
        df_with_coords = df_parsed.join(
            centroids_df,
            df_parsed.PULocationID == centroids_df.LocationID,
            "left"  # Left join to keep all streaming records
        )
        
        # Create feature vector for model prediction
        assembler = VectorAssembler(
            inputCols=["latitude", "longitude"],
            outputCol="features",
            handleInvalid="skip"  # Skip records without coordinates
        )
        
        df_features = assembler.transform(df_with_coords)
        
        # Apply K-Means model to predict cluster
        df_predictions = model.transform(df_features)
        
        # Aggregate by cluster
        df_aggregated = df_predictions \
            .groupBy("cluster") \
            .count() \
            .orderBy("count", ascending=False)
        
        output_mode = "complete"  # Show all clusters each batch
        
    else:
        print("\n📊 Streaming without model (Location aggregation):")
        print("  Aggregating trips by pickup location")
        
        # Simple aggregation by location
        df_aggregated = df_parsed \
            .groupBy("PULocationID") \
            .count() \
            .orderBy("count", ascending=False) \
            .limit(10)  # Top 10 locations
        
        output_mode = "complete"
    
    # Write stream to console for monitoring
    print(f"\n{'=' * 70}")
    print("🎬 Starting Stream Processing...")
    print(f"{'=' * 70}")
    print("  Output: Console (live updates)")
    print("  Press Ctrl+C to stop")
    print(f"{'=' * 70}\n")
    
    query = df_aggregated \
        .writeStream \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", "false") \
        .option("numRows", 20) \
        .trigger(processingTime='5 seconds') \
        .start()
    
    # Wait for termination
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n\n⚠ Stream stopped by user")
        query.stop()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark Streaming Consumer - Real-Time Hotspot Predictor')
    parser.add_argument('--kafka-server', default='localhost:9092', help='Kafka bootstrap server')
    parser.add_argument('--topic', default='taxi-trips', help='Kafka topic name')
    
    args = parser.parse_args()
    
    try:
        # Create Spark session with Kafka support
        spark = create_streaming_spark_session()
        
        # Process stream
        process_stream(spark, args.kafka_server, args.topic)
        
    except KeyboardInterrupt:
        print("\n✓ Consumer stopped by user")
    except Exception as e:
        print(f"\n✗ Consumer error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

