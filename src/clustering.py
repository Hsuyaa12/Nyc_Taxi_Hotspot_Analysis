"""
Clustering Utilities
Functions for K-Means clustering and model evaluation.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import json


def prepare_clustering_data(df: DataFrame, 
                           feature_col: str = "location_vector",
                           sample_fraction: float = None) -> DataFrame:
    """
    Prepare data for clustering by selecting feature column and optionally sampling.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with feature vector
    feature_col : str
        Name of feature vector column (default: "location_vector")
    sample_fraction : float
        Fraction to sample (None = no sampling)
    
    Returns:
    --------
    DataFrame
        DataFrame ready for clustering
    """
    if feature_col not in df.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in DataFrame")
    
    # Select only rows with valid feature vectors
    clustering_df = df.filter(col(feature_col).isNotNull())
    
    # Optionally sample
    if sample_fraction and sample_fraction < 1.0:
        clustering_df = clustering_df.sample(
            fraction=sample_fraction,
            seed=42
        )
        print(f"Sampled {sample_fraction*100}% of data for clustering")
    
    return clustering_df.select(feature_col).withColumnRenamed(feature_col, "features")


def compute_elbow_curve(df: DataFrame, k_values: list,
                       sample_fraction: float = 0.1,
                       max_iter: int = 20,
                       seed: int = 42) -> dict:
    """
    Compute WSSSE for different K values to find optimal K (elbow method).
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with features column
    k_values : list
        List of K values to test
    sample_fraction : float
        Fraction of data to use (default: 0.1 = 10%)
    max_iter : int
        Maximum iterations for K-Means
    seed : int
        Random seed
    
    Returns:
    --------
    dict
        Dictionary mapping K values to WSSSE scores
    """
    print(f"Computing elbow curve for K values: {k_values}")
    print(f"Using {sample_fraction*100}% sample of data")
    
    # Sample data for faster computation
    if sample_fraction < 1.0:
        sample_df = df.sample(fraction=sample_fraction, seed=seed)
    else:
        sample_df = df
    
    elbow_results = {}
    
    for k in k_values:
        print(f"  Testing K={k}...")
        
        # Train K-Means model
        kmeans = KMeans(
            k=k,
            seed=seed,
            maxIter=max_iter,
            featuresCol="features"
        )
        
        model = kmeans.fit(sample_df)
        
        # Compute WSSSE (Within Set Sum of Squared Errors)
        wssse = model.computeCost(sample_df)
        elbow_results[k] = float(wssse)
        
        print(f"    K={k}: WSSSE = {wssse:.2f}")
    
    return elbow_results


def train_kmeans_model(df: DataFrame, k: int,
                      seed: int = 42,
                      max_iter: int = 20,
                      tol: float = 1e-4) -> KMeans:
    """
    Train K-Means clustering model.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with features column
    k : int
        Number of clusters
    seed : int
        Random seed for reproducibility
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    KMeansModel
        Trained K-Means model
    """
    print(f"Training K-Means model with K={k}...")
    
    kmeans = KMeans(
        k=k,
        seed=seed,
        maxIter=max_iter,
        tol=tol,
        featuresCol="features"
    )
    
    model = kmeans.fit(df)
    
    # Compute WSSSE
    wssse = model.computeCost(df)
    print(f"✓ Model trained. WSSSE = {wssse:.2f}")
    
    return model


def assign_clusters(model: KMeans, df: DataFrame) -> DataFrame:
    """
    Assign cluster IDs to data points using trained model.
    
    Parameters:
    -----------
    model : KMeansModel
        Trained K-Means model
    df : DataFrame
        DataFrame with features column
    
    Returns:
    --------
    DataFrame
        DataFrame with cluster_id column added
    """
    print("Assigning clusters to data points...")
    
    predictions = model.transform(df)
    
    # Rename prediction column to cluster_id
    predictions = predictions.withColumnRenamed("prediction", "cluster_id")
    
    print("✓ Clusters assigned")
    
    return predictions


def get_cluster_statistics(df_with_clusters: DataFrame) -> dict:
    """
    Compute statistics for each cluster.
    
    Parameters:
    -----------
    df_with_clusters : DataFrame
        DataFrame with cluster_id column
    
    Returns:
    --------
    dict
        Dictionary with cluster statistics
    """
    from pyspark.sql.functions import count, avg
    
    print("Computing cluster statistics...")
    
    # Count trips per cluster
    cluster_counts = df_with_clusters.groupBy("cluster_id") \
        .agg(count("*").alias("trip_count")) \
        .orderBy("cluster_id") \
        .collect()
    
    cluster_stats = {}
    for row in cluster_counts:
        cluster_id = row["cluster_id"]
        trip_count = row["trip_count"]
        cluster_stats[cluster_id] = {
            "trip_count": trip_count
        }
    
    print(f"✓ Statistics computed for {len(cluster_stats)} clusters")
    
    return cluster_stats


def get_cluster_centroids(model: KMeans) -> list:
    """
    Get cluster centroids (coordinates) from trained model.
    
    Parameters:
    -----------
    model : KMeansModel
        Trained K-Means model
    
    Returns:
    --------
    list
        List of [latitude, longitude] pairs for each cluster
    """
    centroids = model.clusterCenters()
    
    # Convert to list of [lat, lon] pairs
    centroid_list = []
    for centroid in centroids:
        if len(centroid) >= 2:
            centroid_list.append([float(centroid[0]), float(centroid[1])])
    
    return centroid_list


def save_model(model: KMeans, output_path: str):
    """
    Save trained K-Means model to disk.
    
    Parameters:
    -----------
    model : KMeansModel
        Trained model
    output_path : str
        Path to save model
    """
    print(f"Saving model to {output_path}...")
    model.write().overwrite().save(output_path)
    print("✓ Model saved")


def load_model(spark, model_path: str) -> KMeans:
    """
    Load saved K-Means model from disk.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    model_path : str
        Path to saved model
    
    Returns:
    --------
    KMeansModel
        Loaded model
    """
    from pyspark.ml.clustering import KMeansModel
    
    print(f"Loading model from {model_path}...")
    model = KMeansModel.load(model_path)
    print("✓ Model loaded")
    
    return model

