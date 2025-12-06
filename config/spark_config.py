"""
Spark Session Configuration
Creates and configures Spark session with optimal settings for Big Data processing.
"""

from pyspark.sql import SparkSession
import os

# Fix for Java 17+ Security Manager deprecation issue
# This is required for Java 17, 21, 23+ compatibility with PySpark
os.environ['SPARK_SUBMIT_OPTS'] = '--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED -Djava.security.manager=allow'

# Also set for driver extra options
JAVA_17_OPTIONS = '--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED -Djava.security.manager=allow'


def create_spark_session(app_name="NYC_Taxi_Analysis", master="local[*]", 
                        driver_memory="8g", executor_memory="4g", 
                        shuffle_partitions=200):
    """
    Create and configure Spark session with optimal settings.
    
    Parameters:
    -----------
    app_name : str
        Name of the Spark application
    master : str
        Spark master URL (default: "local[*]" for local mode with all cores)
    driver_memory : str
        Driver memory allocation (default: "8g" - increased for large datasets)
    executor_memory : str
        Executor memory allocation per executor (default: "4g")
    shuffle_partitions : int
        Number of partitions for shuffle operations (default: 200)
    
    Returns:
    --------
    SparkSession
        Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.driver.extraJavaOptions", JAVA_17_OPTIONS) \
        .config("spark.executor.extraJavaOptions", JAVA_17_OPTIONS) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.autoBroadcastJoinThreshold", "50m") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    # Set log level to ERROR to suppress memory warnings
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark


def get_spark_session():
    """
    Get existing Spark session or create a new one.
    
    Returns:
    --------
    SparkSession
        Active Spark session
    """
    try:
        return SparkSession.getActiveSession()
    except:
        return create_spark_session()

