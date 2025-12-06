# Deployment & Infrastructure

This folder contains all infrastructure and deployment-related files for the streaming architecture.

## Files

- **`docker-compose.yml`**: Docker Compose configuration for Apache Kafka and Zookeeper
  - Sets up Kafka broker on port 9092
  - Sets up Zookeeper on port 2181
  - Creates network for container communication

- **`RUN_STREAMING_DEMO.sh`**: Automated script to start the streaming consumer demo
  - Checks Kafka status
  - Verifies topic and message count
  - Starts the PySpark Structured Streaming consumer

## Usage

### Start Kafka Infrastructure
```bash
cd deployment
docker-compose up -d
```

### Stop Kafka Infrastructure
```bash
cd deployment
docker-compose down
```

### Run Streaming Demo
```bash
cd deployment
./RUN_STREAMING_DEMO.sh
```

## Requirements

- Docker and Docker Compose must be installed
- Kafka producer should be running in a separate terminal:
  ```bash
  python scripts/kafka_producer.py --max-records 10000 --rate 100
  ```

