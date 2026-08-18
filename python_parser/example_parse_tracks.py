"""
Example script demonstrating how to parse Dopplium Tracks files.

This script shows how to:
1. Load and parse a Tracks binary file
2. Filter tracks by various criteria
3. Extract track trajectories
4. Analyze track statistics
5. Visualize tracks (optional with matplotlib)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parse_dopplium_tracks import (
    parse_dopplium_tracks,
    filter_tracks_by_status,
    filter_tracks_by_id,
    filter_tracks_by_lifetime,
    filter_tracks_by_class,
    get_valid_coordinates,
    get_track_statistics,
    get_blob_statistics,
    get_track_lifecycle_stats,
    cartesian_to_spherical,
)


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example_parse_tracks.py <tracks_file.bin>")
        print("\nExample: python example_parse_tracks.py ../example_tracks.bin")
        return
    
    filename = sys.argv[1]
    
    if not Path(filename).exists():
        print(f"Error: File not found: {filename}")
        return
    
    print("=== Parsing Tracks File ===\n")
    
    # Parse the file
    tracks, headers = parse_dopplium_tracks(filename, verbose=True)
    
    if len(tracks) == 0:
        print("\nNo tracks found in file.")
        return
    
    print("\n=== Track Analysis ===\n")
    
    # Get overall statistics
    stats = get_track_statistics(tracks)
    print(f"Total track records: {stats['count']}")
    print(f"Unique tracks: {stats['unique_tracks']}")
    print(f"Frames: {stats['unique_frames']}")
    
    print(f"\nStatus distribution:")
    for status_name, count in stats['status_distribution'].items():
        print(f"  {status_name}: {count}")
    
    print(f"\nValid coordinates:")
    print(f"  Cartesian: {stats['valid_coordinates']['cartesian']}/{stats['count']}")
    print(f"  ENU: {stats['valid_coordinates']['enu']}/{stats['count']}")
    
    print(f"\nLifetime statistics:")
    print(f"  Min: {stats['lifetime']['min']:.2f} s")
    print(f"  Max: {stats['lifetime']['max']:.2f} s")
    print(f"  Mean: {stats['lifetime']['mean']:.2f} s")
    print(f"  Std: {stats['lifetime']['std']:.2f} s")
    
    print(f"\nGap count statistics:")
    print(f"  Min: {stats['gap_count']['min']}")
    print(f"  Max: {stats['gap_count']['max']}")
    print(f"  Mean: {stats['gap_count']['mean']:.2f}")
    
    # Blob statistics
    blob_stats = get_blob_statistics(tracks)
    print(f"\nBlob statistics:")
    print(f"  Detections per blob - mean: {blob_stats['detections_per_blob']['mean']:.1f}, "
          f"max: {blob_stats['detections_per_blob']['max']}")
    print(f"  Blob size (range) - mean: {blob_stats['blob_size_range']['mean']:.2f} m")
    print(f"  Blob size (azimuth) - mean: {blob_stats['blob_size_azimuth']['mean']:.2f} deg")
    print(f"  Blob size (elevation) - mean: {blob_stats['blob_size_elevation']['mean']:.2f} deg")
    print(f"  Blob size (doppler) - mean: {blob_stats['blob_size_doppler']['mean']:.2f} m/s")
    
    # Lifecycle statistics
    lifecycle_stats = get_track_lifecycle_stats(tracks)
    print(f"\nLifecycle statistics (per unique track):")
    print(f"  Mean lifetime: {lifecycle_stats['track_lifetime']['mean']:.2f} s")
    print(f"  Mean gaps per track: {lifecycle_stats['gaps_per_track']['mean']:.2f}")
    
    # Filter examples
    print("\n=== Filtering Examples ===\n")
    
    # Filter by status
    confirmed_tracks = filter_tracks_by_status(tracks, status=1)
    print(f"Confirmed tracks: {len(confirmed_tracks)}")
    
    tentative_tracks = filter_tracks_by_status(tracks, status=0)
    print(f"Tentative tracks: {len(tentative_tracks)}")
    
    coasting_tracks = filter_tracks_by_status(tracks, status=2)
    print(f"Coasting tracks: {len(coasting_tracks)}")
    
    # Filter by lifetime
    long_tracks = filter_tracks_by_lifetime(tracks, min_lifetime=2.0)
    print(f"Tracks with lifetime > 2s: {len(long_tracks)}")
    
    # Get tracks with valid Cartesian coordinates
    valid_cart = get_valid_coordinates(tracks, 'cartesian')
    print(f"Tracks with valid Cartesian: {len(valid_cart)}")
    
    valid_enu = get_valid_coordinates(tracks, 'enu')
    print(f"Tracks with valid ENU: {len(valid_enu)}")
    
    # Individual track trajectories
    print("\n=== Track Trajectories ===\n")
    
    unique_track_ids = np.unique(tracks['track_id'])
    for track_id in unique_track_ids[:5]:  # Show first 5 tracks
        track_traj = filter_tracks_by_id(tracks, track_id)
        print(f"\nTrack ID {track_id}:")
        print(f"  Lifetime: {track_traj[-1]['track_lifetime_seconds']:.2f} s")
        print(f"  Frames: {len(track_traj)}")
        print(f"  Status: {track_traj[-1]['status']} "
              f"({'tentative' if track_traj[-1]['status']==0 else 'confirmed' if track_traj[-1]['status']==1 else 'coasting' if track_traj[-1]['status']==2 else 'terminated'})")
        print(f"  Class: {track_traj[-1]['target_class_id']}")
        print(f"  Gaps: {track_traj[-1]['gap_count']}")
        
        # Show position evolution
        if track_traj[0]['cart_x'] != -1.0:
            print(f"  Position evolution (Cartesian):")
            for i, rec in enumerate(track_traj):
                # Convert to spherical for display
                r, az, el = cartesian_to_spherical(rec['cart_x'], rec['cart_y'], rec['cart_z'])
                print(f"    Frame {i}: ({rec['cart_x']:.1f}, {rec['cart_y']:.1f}, {rec['cart_z']:.1f}) m "
                      f"= ({r:.1f} m, {az:.1f}° az, {el:.1f}° el)")
        
        # Show velocity
        if len(track_traj) > 1 and track_traj[0]['cart_vx'] != -1.0:
            print(f"  Mean velocity: ({np.mean(track_traj['cart_vx']):.1f}, "
                  f"{np.mean(track_traj['cart_vy']):.1f}, {np.mean(track_traj['cart_vz']):.1f}) m/s")
    
    # Optional: Plotting
    try:
        import matplotlib.pyplot as plt
        plot_tracks(tracks)
    except ImportError:
        print("\n(matplotlib not available for plotting)")


def plot_tracks(tracks):
    """Plot track trajectories."""
    import matplotlib.pyplot as plt
    
    # Get valid Cartesian tracks
    valid_tracks = get_valid_coordinates(tracks, 'cartesian')
    
    if len(valid_tracks) == 0:
        print("No tracks with valid Cartesian coordinates for plotting")
        return
    
    unique_track_ids = np.unique(valid_tracks['track_id'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color map for different tracks
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_track_ids)))
    
    # Plot 1: X-Y (top view: x forward, y right)
    ax = axes[0, 0]
    for i, track_id in enumerate(unique_track_ids):
        track_traj = valid_tracks[valid_tracks['track_id'] == track_id]
        ax.plot(track_traj['cart_x'], track_traj['cart_y'], 'o-', 
                color=colors[i], label=f'Track {track_id}', markersize=4)
        # Mark start with larger circle
        ax.plot(track_traj[0]['cart_x'], track_traj[0]['cart_y'], 
                'o', color=colors[i], markersize=10, markerfacecolor='none', markeredgewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Track Trajectories - Top View (X-Y)')
    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    
    # Plot 2: X-Z (side view: x forward, z up)
    ax = axes[0, 1]
    for i, track_id in enumerate(unique_track_ids):
        track_traj = valid_tracks[valid_tracks['track_id'] == track_id]
        ax.plot(track_traj['cart_x'], track_traj['cart_z'], 'o-', 
                color=colors[i], label=f'Track {track_id}', markersize=4)
        ax.plot(track_traj[0]['cart_x'], track_traj[0]['cart_z'], 
                'o', color=colors[i], markersize=10, markerfacecolor='none', markeredgewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Track Trajectories - Side View (X-Z)')
    ax.grid(True)
    ax.legend()
    
    # Plot 3: Range-Azimuth (polar-like)
    ax = axes[1, 0]
    for i, track_id in enumerate(unique_track_ids):
        track_traj = valid_tracks[valid_tracks['track_id'] == track_id]
        # Convert to spherical
        ranges = []
        azimuths = []
        for rec in track_traj:
            r, az, el = cartesian_to_spherical(rec['cart_x'], rec['cart_y'], rec['cart_z'])
            ranges.append(r)
            azimuths.append(az)
        ax.plot(azimuths, ranges, 'o-', color=colors[i], label=f'Track {track_id}', markersize=4)
        ax.plot(azimuths[0], ranges[0], 'o', color=colors[i], markersize=10, 
                markerfacecolor='none', markeredgewidth=2)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Track Trajectories - Range vs Azimuth')
    ax.grid(True)
    ax.legend()
    
    # Plot 4: Lifetime histogram
    ax = axes[1, 1]
    # Get max lifetime per track
    lifetimes = []
    for track_id in unique_track_ids:
        track_traj = tracks[tracks['track_id'] == track_id]
        lifetimes.append(np.max(track_traj['track_lifetime_seconds']))
    
    ax.hist(lifetimes, bins=10, edgecolor='black')
    ax.set_xlabel('Track Lifetime (s)')
    ax.set_ylabel('Count')
    ax.set_title('Track Lifetime Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tracks_visualization.png', dpi=150)
    print("\n[OK] Saved visualization to tracks_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()

