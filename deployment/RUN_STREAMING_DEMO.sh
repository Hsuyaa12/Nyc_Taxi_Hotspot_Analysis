#!/bin/bash
# Quick Streaming Demo Script
# Run from deployment/ directory

# Get the project root directory (parent of deployment/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "NYC Taxi Streaming Demo"
echo "=========================================="
echo ""

# Check if Kafka is running
echo "1. Checking Kafka status..."
cd deployment
docker-compose ps | grep kafka
if [ $? -ne 0 ]; then
    echo "ERROR: Kafka not running. Start with: cd deployment && docker-compose up -d"
    exit 1
fi
cd ..
echo "✓ Kafka is running"
echo ""

# Check if topic has messages
echo "2. Checking Kafka topic..."
MSG_COUNT=$(docker exec taxi-kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic taxi-trips 2>/dev/null | awk -F ":" '{sum += $3} END {print sum}')
echo "   Messages in topic: $MSG_COUNT"
echo ""

# Check if producer is running
echo "3. Checking producer..."
if ps aux | grep -q "[k]afka_producer.py"; then
    echo "✓ Producer is running"
else
    echo "⚠  Producer not running. Start in another terminal:"
    echo "   python scripts/kafka_producer.py --max-records 10000 --rate 100"
fi
echo ""

echo "=========================================="
echo "Starting Streaming Consumer..."
echo "=========================================="
echo "Press Ctrl+C to stop"
echo ""

# Run the consumer
python scripts/streaming_consumer.py

