"""
Visualization Utilities
Functions for creating plots and maps.
"""

import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_hotspot_map(cluster_centroids, cluster_stats, output_path, 
                      top_n=10):
    """
    Create interactive map showing hotspot locations.
    
    Parameters:
    -----------
    cluster_centroids : list
        List of [lat, lon] pairs for cluster centers
    cluster_stats : dict
        Dictionary with cluster statistics (trip_count, total_revenue, etc.)
    output_path : str
        Path to save HTML map
    top_n : int
        Number of top hotspots to highlight
    """
    print(f"Creating hotspot map...")
    
    # Center map on NYC
    nyc_center = [40.7128, -74.0060]
    m = folium.Map(location=nyc_center, zoom_start=11)
    
    # Sort clusters by trip count or revenue
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: x[1].get("trip_count", 0),
        reverse=True
    )[:top_n]
    
    # Add markers for top hotspots
    for cluster_id, stats in sorted_clusters:
        if cluster_id < len(cluster_centroids):
            lat, lon = cluster_centroids[cluster_id]
            
            trip_count = stats.get("trip_count", 0)
            revenue = stats.get("total_revenue", 0)
            avg_fare = stats.get("avg_fare", 0)
            
            # Marker size based on trip count
            marker_size = min(20 + (trip_count / 1000), 50)
            
            # Color based on revenue
            if revenue > 1000000:
                color = "red"
            elif revenue > 500000:
                color = "orange"
            else:
                color = "blue"
            
            popup_text = f"""
            <b>Cluster {cluster_id}</b><br>
            Trips: {trip_count:,}<br>
            Revenue: ${revenue:,.2f}<br>
            Avg Fare: ${avg_fare:.2f}
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_size,
                popup=folium.Popup(popup_text, max_width=200),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    m.save(output_path)
    print(f"✓ Map saved to {output_path}")


def create_hotspot_bar_chart(ranked_clusters, output_path, metric="total_revenue"):
    """
    Create bar chart of top hotspots.
    
    Parameters:
    -----------
    ranked_clusters : list
        List of cluster dictionaries with metrics
    output_path : str
        Path to save plot
    metric : str
        Metric to plot (default: "total_revenue")
    """
    print(f"Creating hotspot bar chart...")
    
    clusters = ranked_clusters[:10]  # Top 10
    cluster_ids = [f"Cluster {c['cluster_id']}" for c in clusters]
    values = [c.get(metric, 0) for c in clusters]
    
    plt.figure(figsize=(12, 6))
    plt.barh(cluster_ids, values, color='steelblue')
    plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Hotspot', fontsize=12)
    plt.title(f'Top 10 Hotspots by {metric.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Bar chart saved to {output_path}")
    plt.close()


def create_temporal_pattern_plots(temporal_patterns, output_path):
    """
    Create temporal pattern plots for top hotspots.
    
    Parameters:
    -----------
    temporal_patterns : dict
        Dictionary mapping cluster_id to temporal patterns
    output_path : str
        Path to save plot
    """
    print(f"Creating temporal pattern plots...")
    
    n_clusters = len(temporal_patterns)
    fig, axes = plt.subplots(n_clusters, 2, figsize=(15, 4*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (cluster_id, patterns) in enumerate(temporal_patterns.items()):
        # Hourly pattern
        hourly = patterns.get("hourly", {})
        hours = sorted(hourly.keys())
        counts = [hourly.get(h, 0) for h in hours]
        
        axes[idx, 0].plot(hours, counts, 'o-', linewidth=2, markersize=6)
        axes[idx, 0].set_xlabel('Hour of Day', fontsize=10)
        axes[idx, 0].set_ylabel('Trip Count', fontsize=10)
        axes[idx, 0].set_title(f'Cluster {cluster_id} - Hourly Pattern', fontsize=11)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xticks(range(0, 24, 2))
        
        # Daily pattern
        daily = patterns.get("daily", {})
        days = sorted(daily.keys())
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        day_labels = [day_names[d-1] if 1 <= d <= 7 else str(d) for d in days]
        counts = [daily.get(d, 0) for d in days]
        
        axes[idx, 1].bar(day_labels, counts, color='steelblue')
        axes[idx, 1].set_xlabel('Day of Week', fontsize=10)
        axes[idx, 1].set_ylabel('Trip Count', fontsize=10)
        axes[idx, 1].set_title(f'Cluster {cluster_id} - Daily Pattern', fontsize=11)
        axes[idx, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Temporal pattern plots saved to {output_path}")
    plt.close()

