"""
Kafka Producer - Simulates Real-Time Taxi Trip Stream
Big Data Concepts:
- Event Streaming: Simulates real-time data ingestion
- Serialization: Converts Spark DataFrame rows to JSON for network transport
- Backpressure: Controlled rate limiting to simulate realistic stream velocity
"""

import os
import sys
import json
import time
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from config.data_config import RAW_DATA_PATH


def create_producer(bootstrap_servers='localhost:9092'):
    """
    Create Kafka producer with proper serialization.
    
    Big Data Concept: Producer handles serialization and network buffering
    for efficient batch transmission to Kafka brokers.
    """
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        # Performance tuning
        batch_size=16384,  # Batch messages for efficiency
        linger_ms=10,  # Wait up to 10ms to batch more messages
        compression_type='gzip',  # Compress data for network efficiency
        acks='all',  # Ensure durability (wait for all replicas)
        retries=3
    )
    print(f"✓ Kafka producer connected to {bootstrap_servers}")
    return producer


def convert_row_to_json(row):
    """
    Convert Spark Row to JSON-serializable dictionary.
    
    Big Data Concept: Schema transformation for cross-system compatibility.
    Handles Spark-specific types (timestamps) for JSON serialization.
    """
    row_dict = row.asDict()
    
    # Convert timestamp objects to ISO format strings
    for key, value in row_dict.items():
        if value is None:
            continue
        # Handle timestamp types
        if hasattr(value, 'isoformat'):
            row_dict[key] = value.isoformat()
        # Handle other non-serializable types
        elif isinstance(value, (int, float, str, bool)):
            row_dict[key] = value
        else:
            row_dict[key] = str(value)
    
    return row_dict


def simulate_taxi_stream(producer, data_path, topic_name='taxi-trips', 
                         rate_limit=100, max_records=None):
    """
    Simulate real-time taxi trip stream from historical data.
    
    Parameters:
    -----------
    producer : KafkaProducer
        Kafka producer instance
    data_path : str
        Path to Parquet file(s)
    topic_name : str
        Kafka topic name
    rate_limit : int
        Trips per second to simulate (default: 100)
    max_records : int
        Maximum records to send (None = all)
    
    Big Data Concepts:
    - Data Velocity: Simulates streaming data arrival rate
    - Event Time: Each trip has a timestamp representing when it occurred
    - Streaming Source: Historical data replayed as if arriving in real-time
    """
    print(f"\n{'=' * 70}")
    print(f"Starting Taxi Trip Stream Simulator")
    print(f"{'=' * 70}")
    print(f"  Topic: {topic_name}")
    print(f"  Rate: {rate_limit} trips/second")
    print(f"  Max records: {max_records or 'unlimited'}")
    print(f"{'=' * 70}\n")
    
    # Create Spark session to read Parquet
    spark = create_spark_session(app_name="TaxiStreamProducer")
    
    # Find and load first available Parquet file
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    
    if not parquet_files:
        print(f"✗ No Parquet files found in {data_path}")
        sys.exit(1)
    
    data_file = parquet_files[0]
    print(f"Loading data from: {os.path.basename(data_file)}")
    
    df = spark.read.parquet(data_file)
    total_rows = df.count()
    print(f"  Total records available: {total_rows:,}")
    
    if max_records:
        df = df.limit(max_records)
        print(f"  Limiting to: {max_records:,} records")
    
    # Calculate sleep time between messages
    sleep_time = 1.0 / rate_limit if rate_limit > 0 else 0.01
    
    # Stream data
    sent_count = 0
    error_count = 0
    start_time = time.time()
    
    try:
        print(f"\n🚀 Starting stream... (Press Ctrl+C to stop)\n")
        
        # Iterate through DataFrame rows
        for row in df.toLocalIterator():
            try:
                # Convert row to JSON-serializable dict
                trip_data = convert_row_to_json(row)
                
                # Send to Kafka
                future = producer.send(topic_name, value=trip_data)
                
                # Optional: Wait for confirmation (comment out for max speed)
                # future.get(timeout=10)
                
                sent_count += 1
                
                # Progress update every 1000 records
                if sent_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = sent_count / elapsed if elapsed > 0 else 0
                    print(f"  Sent {sent_count:,} trips | "
                          f"Rate: {rate:.1f} trips/sec | "
                          f"Errors: {error_count}")
                
                # Rate limiting to simulate realistic stream
                time.sleep(sleep_time)
                
            except KafkaError as e:
                error_count += 1
                if error_count <= 5:  # Show first few errors
                    print(f"  ⚠ Kafka error: {e}")
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"  ⚠ Error processing row: {e}")
        
        # Flush remaining messages
        producer.flush()
        
        # Final statistics
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Stream Complete!")
        print(f"{'=' * 70}")
        print(f"  Total sent: {sent_count:,} trips")
        print(f"  Errors: {error_count}")
        print(f"  Duration: {elapsed:.2f} seconds")
        print(f"  Average rate: {sent_count/elapsed:.1f} trips/sec")
        print(f"{'=' * 70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Stream interrupted by user")
        print(f"  Sent {sent_count:,} trips before interruption")
        producer.flush()
    
    finally:
        producer.close()
        spark.stop()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Producer - Taxi Trip Stream Simulator')
    parser.add_argument('--topic', default='taxi-trips', help='Kafka topic name')
    parser.add_argument('--rate', type=int, default=100, help='Trips per second')
    parser.add_argument('--max-records', type=int, default=None, help='Max records to send')
    parser.add_argument('--kafka-server', default='localhost:9092', help='Kafka bootstrap server')
    
    args = parser.parse_args()
    
    try:
        # Create producer
        producer = create_producer(args.kafka_server)
        
        # Start simulation
        simulate_taxi_stream(
            producer=producer,
            data_path=RAW_DATA_PATH,
            topic_name=args.topic,
            rate_limit=args.rate,
            max_records=args.max_records
        )
        
    except KeyboardInterrupt:
        print("\n✓ Producer stopped by user")
    except Exception as e:
        print(f"\n✗ Producer error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

