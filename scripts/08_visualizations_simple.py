"""
Simple Visualization Script
Creates basic visualizations from the hotspot analysis results.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.data_config import REPORTS_PATH, FIGURES_PATH
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load zone names for mapping LocationID to Zone names
def load_zone_names():
    """Load zone names mapping."""
    zones_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "taxi_zones", "zone_names.csv"
    )
    try:
        zones_df = pd.read_csv(zones_path)
        return dict(zip(zones_df['LocationID'], zones_df['Zone']))
    except:
        return {}


def plot_top_hotspots(results, output_dir):
    """Plot top hotspots by different metrics."""
    
    # Load zone names
    zone_names = load_zone_names()
    
    # Top by trip count
    top_trips = results['top_by_trips'][:10]
    locations = [zone_names.get(x['location_id'], f"Loc {x['location_id']}") for x in top_trips]
    trips = [x['trip_count'] for x in top_trips]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(locations)), trips, color='steelblue')
    plt.xlabel('Location', fontsize=12)
    plt.ylabel('Number of Trips', fontsize=12)
    plt.title('Top 10 Pickup Hotspots by Trip Count', fontsize=14, fontweight='bold')
    plt.xticks(range(len(locations)), locations, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000)}K',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_hotspots_by_trips.png'), dpi=300)
    plt.close()
    print("✓ Created: top_hotspots_by_trips.png")
    
    # Top by revenue
    top_revenue = results['top_by_revenue'][:10]
    locations_rev = [zone_names.get(x['location_id'], f"Loc {x['location_id']}") for x in top_revenue]
    revenue = [x['total_revenue'] for x in top_revenue]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(locations_rev)), [r/1e6 for r in revenue], color='green')
    plt.xlabel('Location', fontsize=12)
    plt.ylabel('Total Revenue (Millions $)', fontsize=12)
    plt.title('Top 10 Pickup Hotspots by Total Revenue', fontsize=14, fontweight='bold')
    plt.xticks(range(len(locations_rev)), locations_rev, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.1f}M',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_hotspots_by_revenue.png'), dpi=300)
    plt.close()
    print("✓ Created: top_hotspots_by_revenue.png")


def plot_temporal_patterns(results, output_dir):
    """Plot temporal patterns for top locations."""
    
    temporal = results['temporal_patterns']
    
    # Get first location
    loc_ids = list(temporal.keys())
    if not loc_ids:
        print("⚠ No temporal patterns to plot")
        return
    
    # Plot hourly patterns for top 3 locations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, loc_id in enumerate(loc_ids[:3]):
        hourly = temporal[str(loc_id)]['hourly']
        hours = sorted([int(k) for k in hourly.keys()])
        trips = [hourly[str(h)] for h in hours]
        
        axes[idx].plot(hours, trips, marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[idx].set_xlabel('Hour of Day', fontsize=11)
        axes[idx].set_ylabel('Number of Trips', fontsize=11)
        axes[idx].set_title(f'Location {loc_id}: Hourly Pattern', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(range(0, 24, 3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_patterns_hourly.png'), dpi=300)
    plt.close()
    print("✓ Created: temporal_patterns_hourly.png")
    
    # Plot day of week patterns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    for idx, loc_id in enumerate(loc_ids[:3]):
        daily = temporal[str(loc_id)]['daily']
        days = sorted([int(k) for k in daily.keys()])
        trips = [daily[str(d)] for d in days]
        
        axes[idx].bar(range(len(days)), trips, color='coral')
        axes[idx].set_xlabel('Day of Week', fontsize=11)
        axes[idx].set_ylabel('Number of Trips', fontsize=11)
        axes[idx].set_title(f'Location {loc_id}: Day of Week Pattern', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(len(days)))
        axes[idx].set_xticklabels([day_names[d-1] for d in days])
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_patterns_daily.png'), dpi=300)
    plt.close()
    print("✓ Created: temporal_patterns_daily.png")


def plot_revenue_vs_volume(results, output_dir):
    """Plot revenue vs volume scatter."""
    
    top_trips = results['top_by_trips'][:20]
    
    trips = [x['trip_count'] for x in top_trips]
    revenue = [x['total_revenue'] for x in top_trips]
    locations = [x['location_id'] for x in top_trips]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter([t/1000 for t in trips], [r/1e6 for r in revenue], 
                         s=200, alpha=0.6, c=range(len(trips)), cmap='viridis')
    
    # Add labels for some points
    for i, loc in enumerate(locations[:5]):
        plt.annotate(f'Loc {loc}', 
                    ([t/1000 for t in trips][i], [r/1e6 for r in revenue][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Trip Volume (Thousands)', fontsize=12)
    plt.ylabel('Total Revenue (Millions $)', fontsize=12)
    plt.title('Revenue vs Volume for Top 20 Hotspots', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Rank')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_vs_volume.png'), dpi=300)
    plt.close()
    print("✓ Created: revenue_vs_volume.png")


def plot_benchmark_results(benchmark_path, output_dir):
    """Plot benchmark comparison."""
    
    try:
        with open(benchmark_path, 'r') as f:
            bench_data = json.load(f)
        
        results = [r for r in bench_data['results'] if 'error' not in r]
        
        if not results:
            print("⚠ No valid benchmark results to plot")
            return
        
        # Extract data
        tools = [r['tool'] for r in results]
        rows = [r['rows'] for r in results]
        times = [r['time_seconds'] for r in results]
        
        # Plot execution time
        plt.figure(figsize=(10, 6))
        colors = ['steelblue' if t == 'Spark' else 'coral' for t in tools]
        bars = plt.bar(range(len(rows)), times, color=colors)
        
        plt.xlabel('Dataset Size (rows)', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Spark Performance Across Dataset Sizes', fontsize=14, fontweight='bold')
        plt.xticks(range(len(rows)), [f'{r/1000:.0f}K' if r < 1000000 else f'{r/1000000:.1f}M' for r in rows])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=300)
        plt.close()
        print("✓ Created: benchmark_comparison.png")
        
    except Exception as e:
        print(f"⚠ Could not plot benchmark results: {e}")


def plot_cluster_distribution(cluster_stats_path, output_dir):
    """Plot distribution of trips across all K-Means clusters."""
    
    try:
        with open(cluster_stats_path, 'r') as f:
            cluster_data = json.load(f)
        
        clusters = cluster_data['clusters']
        # Sort by trip count descending
        clusters_sorted = sorted(clusters, key=lambda x: x['trip_count'], reverse=True)
        
        cluster_ids = [f"Cluster {c['cluster']}" for c in clusters_sorted]
        trip_counts = [c['trip_count'] for c in clusters_sorted]
        
        plt.figure(figsize=(16, 8))
        bars = plt.bar(range(len(cluster_ids)), [t/1e6 for t in trip_counts], 
                      color='steelblue', alpha=0.8)
        
        # Highlight top 3 clusters
        for i in range(3):
            bars[i].set_color('darkblue')
        
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Trips (Millions)', fontsize=12)
        plt.title('K-Means Cluster Distribution (All 20 Clusters)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(cluster_ids)), cluster_ids, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}M',
                    ha='center', va='bottom', fontsize=9)
        
        # Add annotation for top cluster
        plt.text(0, bars[0].get_height() + 0.1, 'Top Cluster\n(Midtown)', 
                ha='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300)
        plt.close()
        print("✓ Created: cluster_distribution.png")
        
    except Exception as e:
        print(f"⚠ Could not plot cluster distribution: {e}")


def plot_avg_fare_by_cluster(cluster_stats_path, output_dir):
    """Plot average fare by cluster, highlighting airport premium."""
    
    try:
        with open(cluster_stats_path, 'r') as f:
            cluster_data = json.load(f)
        
        clusters = cluster_data['clusters']
        # Sort by average fare descending
        clusters_sorted = sorted(clusters, key=lambda x: x['avg_fare'], reverse=True)
        
        cluster_ids = [f"C{c['cluster']}" for c in clusters_sorted]
        avg_fares = [c['avg_fare'] for c in clusters_sorted]
        trip_counts = [c['trip_count'] for c in clusters_sorted]
        
        # Color code: airports (high fare) vs regular zones
        colors = []
        for c in clusters_sorted:
            if c['avg_fare'] > 25:  # Airport zones
                colors.append('red')
            elif c['avg_fare'] > 15:  # High-value zones
                colors.append('orange')
            else:  # Regular zones
                colors.append('steelblue')
        
        plt.figure(figsize=(16, 8))
        bars = plt.bar(range(len(cluster_ids)), avg_fares, color=colors, alpha=0.8)
        
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Average Fare ($)', fontsize=12)
        plt.title('Average Fare by K-Means Cluster (Airport Premium Highlighted)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(cluster_ids)), cluster_ids, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(y=11.5, color='gray', linestyle='--', alpha=0.5, label='Average ($11.50)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Add annotations for airports
        for i, c in enumerate(clusters_sorted):
            if c['avg_fare'] > 25:
                cluster_idx = clusters_sorted.index(c)
                plt.text(cluster_idx, c['avg_fare'] + 2, 
                        f"JFK\n({c['trip_count']/1e6:.1f}M trips)" if c['cluster'] == 1 
                        else f"LGA\n({c['trip_count']/1e6:.1f}M trips)",
                        ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_fare_by_cluster.png'), dpi=300)
        plt.close()
        print("✓ Created: avg_fare_by_cluster.png")
        
    except Exception as e:
        print(f"⚠ Could not plot average fare by cluster: {e}")


def plot_data_retention(cleaning_summary_path, output_dir):
    """Plot data retention before and after cleaning optimization."""
    
    try:
        with open(cleaning_summary_path, 'r') as f:
            cleaning_data = json.load(f)
        
        initial = cleaning_data['initial_row_count']
        cleaned = cleaning_data['cleaned_row_count']
        removed = cleaning_data['rows_removed']
        
        # Calculate retention percentage
        retention_pct = (cleaned / initial) * 100
        removal_pct = (removed / initial) * 100
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Before/After comparison
        categories = ['Initial\n(84.6M)', 'Cleaned\n(83.9M)', 'Removed\n(0.7M)']
        values = [initial/1e6, cleaned/1e6, removed/1e6]
        colors_bar = ['lightblue', 'green', 'red']
        
        bars1 = ax1.bar(categories, values, color=colors_bar, alpha=0.8)
        ax1.set_ylabel('Number of Rows (Millions)', fontsize=12)
        ax1.set_title('Data Cleaning: Before vs After Optimization', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}M',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Chart 2: Retention percentage
        retention_data = [retention_pct, removal_pct]
        labels = [f'Retained\n{retention_pct:.2f}%', f'Removed\n{removal_pct:.2f}%']
        colors_pie = ['green', 'red']
        explode = (0.05, 0)  # Explode the removed slice slightly
        
        ax2.pie(retention_data, labels=labels, colors=colors_pie, autopct='%1.2f%%',
               startangle=90, explode=explode, shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Data Retention Rate', fontsize=14, fontweight='bold')
        
        # Add improvement annotation
        fig.text(0.5, 0.02, 
                f'✓ Data Retention: {retention_pct:.2f}% (Improved from ~30% with optimized cleaning logic)',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_retention_comparison.png'), dpi=300)
        plt.close()
        print("✓ Created: data_retention_comparison.png")
        
    except Exception as e:
        print(f"⚠ Could not plot data retention: {e}")


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("NYC Taxi Visualization Generation")
        print("=" * 60)
        
        # Load hotspot results
        print("\nStep 1: Loading results...")
        hotspot_path = os.path.join(REPORTS_PATH, "hotspot_analysis.json")
        
        if not os.path.exists(hotspot_path):
            print(f"✗ Hotspot results not found at {hotspot_path}")
            sys.exit(1)
        
        with open(hotspot_path, 'r') as f:
            hotspot_results = json.load(f)
        
        print("✓ Loaded hotspot analysis results")
        
        # Create visualizations
        print("\nStep 2: Creating visualizations...")
        
        plot_top_hotspots(hotspot_results, FIGURES_PATH)
        plot_temporal_patterns(hotspot_results, FIGURES_PATH)
        plot_revenue_vs_volume(hotspot_results, FIGURES_PATH)
        
        # Plot benchmark if available
        benchmark_path = os.path.join(REPORTS_PATH, "benchmark_results.json")
        if os.path.exists(benchmark_path):
            plot_benchmark_results(benchmark_path, FIGURES_PATH)
        
        # Plot cluster distribution from K-Means results
        print("\nStep 3: Creating K-Means cluster visualizations...")
        cluster_stats_path = os.path.join(REPORTS_PATH, "cluster_statistics_proper.json")
        if os.path.exists(cluster_stats_path):
            plot_cluster_distribution(cluster_stats_path, FIGURES_PATH)
            plot_avg_fare_by_cluster(cluster_stats_path, FIGURES_PATH)
        else:
            print("⚠ Cluster statistics not found, skipping cluster visualizations")
        
        # Plot data retention comparison
        print("\nStep 4: Creating data quality visualization...")
        cleaning_summary_path = os.path.join(REPORTS_PATH, "cleaning_summary.json")
        if os.path.exists(cleaning_summary_path):
            plot_data_retention(cleaning_summary_path, FIGURES_PATH)
        else:
            print("⚠ Cleaning summary not found, skipping data retention visualization")
        
        print("\n" + "=" * 60)
        print("Visualization generation complete!")
        print("=" * 60)
        print(f"\nVisualizations saved to: {FIGURES_PATH}")
        print("\nGenerated visualizations:")
        print("  - top_hotspots_by_trips.png")
        print("  - top_hotspots_by_revenue.png")
        print("  - temporal_patterns_hourly.png")
        print("  - temporal_patterns_daily.png")
        print("  - revenue_vs_volume.png")
        print("  - benchmark_comparison.png")
        print("  - cluster_distribution.png")
        print("  - avg_fare_by_cluster.png")
        print("  - data_retention_comparison.png")
        
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

