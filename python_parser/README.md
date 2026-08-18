# Python Parsers for Dopplium Binary Formats

Binary data parsers for Dopplium radar data formats, implemented in Python using `struct` and `numpy`.

## Overview

This package provides six binary parsers for different radar data formats:

| Parser | Format | Message Type | Data Shape | Use Case |
|--------|--------|--------------|------------|----------|
| **parse_dopplium_raw** | RawData/ADCData | 3 (v2) / 1 (v3) | [samples, chirpsPerTx, channels, frames] | Raw ADC data from radar |
| **parse_dopplium_rdch** | RDCMaps (RDCh) | 2 | [range, doppler, channels, cpis] | Range-Doppler-Channel processed data |
| **parse_dopplium_radarcube** | RadarCube | 3 | [range, doppler, azimuth, elevation, cpis] | Full 4D radar cubes with angles |
| **parse_dopplium_detections** | Detections | 4 | Structured array of detections | Processed target detections |
| **parse_dopplium_blobs** | Blobs | 5 | Structured array of blobs | Clustered detections with centroids/spreads |
| **parse_dopplium_tracks** | Tracks | 6 | Structured array of tracks | Tracked targets with state estimation |

All parsers support automatic format detection via the main `parse_dopplium()` dispatcher.
ADC, RDCh, and RadarCube parsers also support `start_frame` for zero-based
offset reads. Combined with a limit, the returned window is
`[start_frame, start_frame + max_frames)` for ADC or
`[start_frame, start_frame + max_cpis)` for RDCh/RadarCube.

## Installation

```bash
# From the repo root
pip install -e .

# Or install dependencies only
pip install numpy matplotlib
```

## Quick Start

### Automatic Format Detection (Recommended)

The easiest way to parse any Dopplium file:

```python
from python_parser import parse_dopplium

# Automatically detects format and routes to appropriate parser
data, headers = parse_dopplium("file.bin", verbose=True)

# Check what format was parsed
print(f"Message type: {headers['file'].message_type}")
print(f"Data shape: {data.shape}")
```

### Verbosity

Every parse function takes a `verbose` int from 0 (nothing) to 4 (everything, the default):

| Level | Output |
|-------|--------|
| 0 | Nothing (informational output only — see note below) |
| 1 | Final summary (shape/record counts) |
| 2 | + Dispatcher routing message, header/config summary, size estimates |
| 3 | + Throttled per-record/per-batch/per-CPI progress |
| 4 (default) | Same as 3 (reserved for future finer-grained progress) |

Anomaly warnings (payload-size mismatches) and EOF/struct-error notices always print, regardless of `verbose` — they are not gated by this scale.

```python
data, headers = parse_dopplium("file.bin", verbose=0)  # parses silently (warnings still show if any)
```

### Direct Parser Usage

For when you know the format ahead of time:

```python
from python_parser import (
    parse_dopplium_raw,
    parse_dopplium_rdch,
    parse_dopplium_radarcube,
    parse_dopplium_detections,
    parse_dopplium_blobs,
    parse_dopplium_tracks
)

# Parse specific formats
raw_data, headers = parse_dopplium_raw("raw.bin")
rdch_data, headers = parse_dopplium_rdch("rdch.bin")
cube_data, headers = parse_dopplium_radarcube("radarcube.bin")
detections, headers = parse_dopplium_detections("detections.bin")
blobs, headers = parse_dopplium_blobs("blobs.bin")
tracks, headers = parse_dopplium_tracks("tracks.bin")
```

## Parser Details

### 1. parse_dopplium_raw - Raw ADC Data

For unprocessed radar ADC samples.

```python
from python_parser import parse_dopplium_raw

# Parse raw ADC data
data, headers = parse_dopplium_raw(
    "raw_data.bin",
    start_frame=100,        # Start from zero-based frame index 100
    max_frames=100,          # Limit number of frames (None = all)
    cast='float32',          # Output data type
    return_complex=True,     # Return complex data (I+jQ)
    verbose=True
)

# Data shape: [samples, chirpsPerTx, channels, frames]
print(f"Shape: {data.shape}")
print(f"Samples per chirp: {data.shape[0]}")
print(f"Chirps per TX: {data.shape[1]}")
print(f"Channels: {data.shape[2]}")
print(f"Frames: {data.shape[3]}")

# Access headers
body_header = headers['body']
print(f"Sample rate: {body_header.sample_rate_ksps} kHz")
print(f"Chirp bandwidth: {body_header.bandwidth_ghz} GHz")
```

**Key Features:**
- Raw I+Q samples from radar ADC
- Configurable output type (float32, float64, int16)
- Complex or separate I/Q channels
- Frame-based organization

**Common Use Case:** Input to custom processing pipelines

### 2. parse_dopplium_rdch - Range-Doppler-Channel

For processed 3D radar data cubes.

```python
from python_parser import (
    parse_dopplium_rdch,
    get_range_axis,
    get_velocity_axis,
    get_processing_info
)

# Parse RDCh file
data, headers = parse_dopplium_rdch(
    "rdch.bin",
    start_frame=100,  # Start from zero-based CPI index 100
    max_cpis=50,    # Limit CPIs (None = all)
    verbose=True
)

# Data shape: [range_bins, doppler_bins, channels, cpis]
print(f"Shape: {data.shape}")

# Get axis information
range_axis = get_range_axis(headers)    # Range in meters
velocity_axis = get_velocity_axis(headers)  # Velocity in m/s

# Get processing parameters
proc_info = get_processing_info(headers)
print(f"FFT sizes: range={proc_info['nfft_range']}, doppler={proc_info['nfft_doppler']}")
print(f"Windows: range={proc_info['range_window']}, doppler={proc_info['doppler_window']}")
print(f"Data format: {proc_info['data_format']}")  # complex, amplitude, or power
print(f"Scale: {'dB' if proc_info['is_db_scale'] else 'linear'}")

# Process single CPI
cpi_data = data[:, :, :, 0]  # First CPI
print(f"CPI shape: {cpi_data.shape}")
```

**Key Features:**
- 3D data cubes (range × doppler × channels)
- Processing metadata (FFT sizes, windows, shifts)
- Physical and FFT resolutions
- Support for complex, amplitude, or power data
- Linear or dB scale

**Common Use Case:** Input to detection algorithms, visualization

### 3. parse_dopplium_radarcube - Full 4D Radar Cubes

For radar data with angle estimation (azimuth and elevation).

```python
from python_parser import (
    parse_dopplium_radarcube,
    get_range_axis,
    get_velocity_axis,
    get_azimuth_axis,
    get_elevation_axis,
    has_known_angles,
    uses_fft_for_angles
)

# Parse RadarCube file
data, headers = parse_dopplium_radarcube(
    "radarcube.bin",
    start_frame=100,
    max_cpis=50,
    verbose=True
)

# Data shape: [range, doppler, azimuth, elevation, cpis]
print(f"Shape: {data.shape}")

# Get all axes
range_axis = get_range_axis(headers)          # meters
velocity_axis = get_velocity_axis(headers)    # m/s
azimuth_axis = get_azimuth_axis(headers)      # degrees (FFT-aware, may be nonlinear)
elevation_axis = get_elevation_axis(headers)  # degrees

# Check angle estimation method
if uses_fft_for_angles(headers):
    print("Using FFT-based angle estimation")
else:
    algo = headers['body'].angle_estimation_algorithm
    algo_map = {1: 'CAPON', 2: 'MUSIC', 3: 'other'}
    print(f"Using {algo_map.get(algo, 'unknown')} angle estimation")

# Incoherent integration metadata (1 means no incoherent CPI summing)
print(f"Incoherent CPI integration: {headers['body'].cpis_incoherently_integrated}")

# Access single CPI
cpi_data = data[:, :, :, :, 0]  # First CPI: [range, doppler, az, el]
```

Axis mapping note:
- For FFT angle estimation (`angle_estimation_algorithm=FFT` and `nfft_azimuth>0`),
  `get_azimuth_axis()` returns nonlinear spacing in degrees.
- For full-span metadata (`azimuth_min_deg=-90`, `azimuth_max_deg=+90`), it uses
  ULA half-lambda FFT-bin mapping (`sin(theta_k)=2k/N`).
- For non-FFT angle algorithms, azimuth/elevation helpers use linear metadata mapping.

**Key Features:**
- 4D data cubes (range × doppler × azimuth × elevation)
- Multiple angle estimation algorithms (FFT, CAPON, MUSIC)
- Full angular information
- All RDCh features plus angular processing

**Common Use Case:** Advanced detection with angle information, 3D visualization

### 4. parse_dopplium_detections - Target Detections

For processed detection lists with full target parameters.

```python
from python_parser import (
    parse_dopplium_detections,
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics
)

# Parse detections file
detections, headers = parse_dopplium_detections(
    "detections.bin",
    max_batches=None,  # Read all batches
    verbose=True
)

# Detections is a numpy structured array
print(f"Total detections: {len(detections)}")
print(f"Fields: {list(detections.dtype.names)}")

# Access detection fields
ranges = detections['range']              # meters
velocities = detections['velocity']       # m/s
azimuths = detections['azimuth']         # degrees
elevations = detections['elevation']     # degrees
amplitudes = detections['amplitude']     # detection strength
monopulse_az = detections['monopulse_ratio_az']  # angle quality
monopulse_el = detections['monopulse_ratio_el']

# Cell indices (link back to radar cube)
range_cells = detections['range_cell']
doppler_cells = detections['doppler_cell']
azimuth_cells = detections['azimuth_cell']
elevation_cells = detections['elevation_cell']

# Filter detections
close_targets = filter_detections_by_range(detections, 0.0, 50.0)
moving_targets = filter_detections_by_velocity(detections, 1.0, 100.0)
strong_targets = filter_detections_by_amplitude(detections, 20.0)

# Combined filtering
close_strong_moving = filter_detections_by_amplitude(
    filter_detections_by_velocity(
        filter_detections_by_range(detections, 10.0, 50.0),
        2.0, 50.0
    ),
    25.0
)

# Get statistics
stats = get_detection_statistics(detections)
print(f"Range: {stats['range']['min']:.2f} to {stats['range']['max']:.2f} m")
print(f"Velocity: mean={stats['velocity']['mean']:.2f} m/s")
print(f"Total batches: {stats['batches']['unique_batches']}")

# Custom filtering with numpy
mask = (
    (detections['range'] > 10) & 
    (detections['range'] < 50) &
    (np.abs(detections['velocity']) > 1.0)
)
filtered = detections[mask]

# Group by batch
for batch_idx in np.unique(detections['batch_index']):
    batch_dets = detections[detections['batch_index'] == batch_idx]
    print(f"Batch {batch_idx}: {len(batch_dets)} detections")
```

**Detection Record Fields:**
- `range` (float64): Range in meters
- `velocity` (float64): Radial velocity in m/s
- `azimuth` (float64): Azimuth angle in degrees
- `elevation` (float64): Elevation angle in degrees
- `amplitude` (float64): Detection amplitude
- `monopulse_ratio_az` (float32): Azimuth monopulse ratio (angle quality)
- `monopulse_ratio_el` (float32): Elevation monopulse ratio
- `range_cell` (int16): Range bin index in processing grid
- `doppler_cell` (int16): Doppler bin index
- `azimuth_cell` (int16): Azimuth bin index
- `elevation_cell` (int16): Elevation bin index
- `batch_index` (uint32): Which batch this detection belongs to
- `sequence_number` (uint32): Sequence number of the batch

**Key Features:**
- Structured numpy arrays for efficient access
- Built-in filtering functions
- Statistics calculation
- Batch organization (detections grouped by time)
- Cell indices link back to radar cube
- Empty batches supported

**Common Use Case:** Target tracking, visualization, data export

### 5. parse_dopplium_blobs - Clustered Detections

For clustered detection data (blobs) with centroid positions and spatial spreads.

```python
from python_parser import (
    parse_dopplium_blobs,
    filter_blobs_by_range,
    filter_blobs_by_velocity,
    filter_blobs_by_amplitude,
    filter_blobs_by_size,
    filter_blobs_by_detection_count,
    get_blob_statistics
)

# Parse blobs file
blobs, headers = parse_dopplium_blobs(
    "blobs.bin",
    max_batches=None,  # Read all batches
    verbose=True
)

# Blobs is a numpy structured array
print(f"Total blobs: {len(blobs)}")
print(f"Fields: {list(blobs.dtype.names)}")

# Access blob fields - Centroids
ranges = blobs['range_centroid']          # meters
velocities = blobs['velocity_centroid']   # m/s
azimuths = blobs['azimuth_centroid']      # degrees
elevations = blobs['elevation_centroid']  # degrees

# Access blob fields - Spreads
range_spread = blobs['range_spread']      # meters
velocity_spread = blobs['velocity_spread'] # m/s
azimuth_spread = blobs['azimuth_spread']  # degrees
elevation_spread = blobs['elevation_spread'] # degrees

# Amplitude and identity
amplitudes = blobs['amplitude']
blob_ids = blobs['blob_id']
num_detections = blobs['num_detections']

# Cell indices (link back to radar cube)
range_cells = blobs['range_cell']
doppler_cells = blobs['doppler_cell']

# Filter blobs
close_blobs = filter_blobs_by_range(blobs, 0.0, 50.0)
moving_blobs = filter_blobs_by_velocity(blobs, -50.0, -1.0)
strong_blobs = filter_blobs_by_amplitude(blobs, 30.0)

# Filter by size (spread)
small_blobs = filter_blobs_by_size(blobs, max_range_spread=2.0)
large_clusters = filter_blobs_by_size(blobs, min_range_spread=5.0)

# Filter by detection count
dense_blobs = filter_blobs_by_detection_count(blobs, min_detections=10)

# Get statistics
stats = get_blob_statistics(blobs)
print(f"Range: {stats['range_centroid']['min']:.2f} to {stats['range_centroid']['max']:.2f} m")
print(f"Velocity: mean={stats['velocity_centroid']['mean']:.2f} m/s")
print(f"Total detections in blobs: {stats['num_detections']['total']}")
print(f"Batches: {stats['batches']['unique_batches']}")

# Group by batch
for batch_idx in np.unique(blobs['batch_index']):
    batch_blobs = blobs[blobs['batch_index'] == batch_idx]
    print(f"Batch {batch_idx}: {len(batch_blobs)} blobs")
```

**Blob Record Fields:**
- `range_centroid` (float32): Centroid range in meters
- `velocity_centroid` (float32): Centroid velocity in m/s
- `azimuth_centroid` (float32): Centroid azimuth in degrees
- `elevation_centroid` (float32): Centroid elevation in degrees
- `range_spread` (float32): Range extent in meters
- `velocity_spread` (float32): Velocity extent in m/s
- `azimuth_spread` (float32): Azimuth extent in degrees
- `elevation_spread` (float32): Elevation extent in degrees
- `amplitude` (float32): Aggregate amplitude
- `range_cell` (int16): Range bin index
- `doppler_cell` (int16): Doppler bin index
- `azimuth_cell` (int16): Azimuth bin index
- `elevation_cell` (int16): Elevation bin index
- `blob_id` (uint32): Unique blob identifier within batch
- `num_detections` (uint16): Number of detections in this blob
- `batch_index` (uint32): Which batch this blob belongs to
- `sequence_number` (uint32): Sequence number of the batch

**Key Features:**
- Structured numpy arrays for efficient access
- Built-in filtering functions (range, velocity, amplitude, size, detection count)
- Statistics calculation
- Batch organization (blobs grouped by time)
- Cell indices link back to radar cube
- Processing pipeline: Detections → Clustering → **Blobs** → Tracking → Tracks

**Common Use Case:** Input to tracking algorithms, cluster analysis, visualization

## Example Scripts

Each parser has example code:

```bash
python example_parse.py               # Raw/RDCh/RadarCube examples
python example_parse_detections.py    # Detections with visualization
python example_parse_blobs.py         # Blobs with filtering and statistics
python example_parse_tracks.py        # Tracks with trajectory analysis
```

## Common Features

All parsers share these capabilities:

| Feature | Description |
|---------|-------------|
| **Automatic Dispatch** | Use `parse_dopplium()` to auto-detect and parse any format |
| **Header Access** | Full access to file, body, and CPI/batch headers |
| **Verbose Mode** | Optional detailed output for debugging |
| **Partial Reading** | Limit number of CPIs/frames/batches read |
| **Offset Tensor Reading** | For ADC, RDCh, and RadarCube, `start_frame` reads from a zero-based frame/CPI offset |
| **Endianness Support** | Handles both little-endian and big-endian files |
| **Version Support** | Supports Version 2, Version 3, Version 4, and Version 5 formats |
| **Error Handling** | Comprehensive validation and error messages |

## Accessing Headers

All parsers return headers in a consistent dictionary structure:

```python
data, headers = parse_dopplium_*("file.bin")

# File header (common to all formats)
file_header = headers['file']
print(f"Version: {file_header.version}")
print(f"Message type: {file_header.message_type}")
print(f"Node ID: {file_header.node_id}")
print(f"Total CPIs/frames written: {file_header.total_frames_written}")
print(f"Endianness: {'little' if file_header.endianness == 1 else 'big'}")

# Body header (format-specific)
body_header = headers['body']
# Fields vary by format

# CPI/Frame/Batch headers (list of per-CPI metadata)
for i, cpi_header in enumerate(headers['cpi'][:5]):  # or 'frame', 'batch'
    print(f"CPI {i}: timestamp={cpi_header.cpi_timestamp_utc_ticks}")
```

## Message Type Reference

| Message Type | Version 2 | Version 3 | Version 4 | Version 5 | Parser Function |
|--------------|-----------|-----------|-----------|-----------|-----------------|
| 0 | Unknown | Unknown | Unknown | Unknown | Not supported |
| 1 | Detections | ADCData | ADCData | ADCData | `parse_dopplium_raw` |
| 2 | Tracks | RDCMaps (RDCh) | RDCMaps (RDCh) | RDCMaps (RDCh) | `parse_dopplium_rdch` |
| 3 | RawData/ADC | RadarCube | RadarCube | RadarCube | `parse_dopplium_raw` / `parse_dopplium_radarcube` |
| 4 | Aggregated | Detections | Detections | Detections | `parse_dopplium_detections` |
| 5 | - | Blobs | Blobs | Blobs | `parse_dopplium_blobs` |
| 6 | - | Tracks | Tracks | Tracks | `parse_dopplium_tracks` |

## Helper Functions

### For RDCh and RadarCube

```python
from python_parser import (
    get_range_axis,
    get_velocity_axis,
    get_azimuth_axis,        # RadarCube only
    get_elevation_axis,      # RadarCube only
    get_processing_info,
    has_known_angles,        # RadarCube only
    uses_fft_for_angles      # RadarCube only
)

# Extract axis vectors
range_m = get_range_axis(headers)
velocity_mps = get_velocity_axis(headers)

# RadarCube specific (azimuth/elevation helpers are FFT-aware)
azimuth_deg = get_azimuth_axis(headers)
elevation_deg = get_elevation_axis(headers)

# Check angle estimation
if has_known_angles(headers):
    if uses_fft_for_angles(headers):
        print("Using FFT-based angles")
    else:
        print("Using advanced angle estimation (CAPON/MUSIC)")

# Get all processing parameters
proc_info = get_processing_info(headers)
```

### For Detections

```python
from python_parser import (
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics
)

# Filter detections
filtered = filter_detections_by_range(detections, min_range, max_range)
filtered = filter_detections_by_velocity(detections, min_vel, max_vel)
filtered = filter_detections_by_amplitude(detections, min_amplitude)

# Get statistics
stats = get_detection_statistics(detections)
```

### For Blobs

```python
from python_parser import (
    filter_blobs_by_range,
    filter_blobs_by_velocity,
    filter_blobs_by_amplitude,
    filter_blobs_by_size,
    filter_blobs_by_detection_count,
    get_blob_statistics
)

# Filter blobs by position
filtered = filter_blobs_by_range(blobs, min_range, max_range)
filtered = filter_blobs_by_velocity(blobs, min_vel, max_vel)
filtered = filter_blobs_by_amplitude(blobs, min_amplitude)

# Filter by cluster size
filtered = filter_blobs_by_size(blobs, min_range_spread=1.0, max_range_spread=10.0)

# Filter by detection count
filtered = filter_blobs_by_detection_count(blobs, min_detections=5)

# Get statistics
stats = get_blob_statistics(blobs)
```

## Advanced Usage

### Working with Complex Data

```python
# RDCh/RadarCube often contain complex data
data, headers = parse_dopplium_rdch("file.bin")

# Convert to magnitude
magnitude = np.abs(data)

# Convert to dB
magnitude_db = 20 * np.log10(np.abs(data) + 1e-12)

# Extract phase
phase = np.angle(data)

# Separate real and imaginary
real_part = np.real(data)
imag_part = np.imag(data)
```

### Processing Single CPI

```python
# RDCh data
data, headers = parse_dopplium_rdch("file.bin")
# Shape: [range, doppler, channels, cpis]

# Process first CPI
cpi_0 = data[:, :, :, 0]  # [range, doppler, channels]

# Average across channels
cpi_avg = np.mean(cpi_0, axis=2)  # [range, doppler]

# Find peak
peak_idx = np.unravel_index(np.argmax(np.abs(cpi_avg)), cpi_avg.shape)
range_bin, doppler_bin = peak_idx

# Convert to physical units
range_axis = get_range_axis(headers)
velocity_axis = get_velocity_axis(headers)
peak_range = range_axis[range_bin]
peak_velocity = velocity_axis[doppler_bin]
```

### Memory-Efficient Partial Reading

```python
# Read 10 CPIs beginning at CPI 100 (useful for large files)
data, headers = parse_dopplium_rdch(
    "large_file.bin",
    start_frame=100,
    max_cpis=10,
    verbose=True
)

# Read 50 frames beginning at frame 200
raw_data, headers = parse_dopplium_raw(
    "raw_data.bin",
    start_frame=200,
    max_frames=50
)

# Read limited detections
detections, headers = parse_dopplium_detections(
    "detections.bin",
    max_batches=100
)

# Read limited blobs
blobs, headers = parse_dopplium_blobs(
    "blobs.bin",
    max_batches=100
)

# Read tracks with limited frames
tracks, headers = parse_dopplium_tracks(
    "tracks.bin",
    max_frames=50
)
```

## Tracks Parser

Parse tracking data with complete state estimation including position, velocity, and uncertainty.

### Key Features

- **Frame-centric organization**: Tracks organized by time frame, not track ID
- **Multiple coordinate systems**: Sensor Cartesian and ENU coordinates
- **Lifecycle tracking**: Birth time, lifetime, gap count for each track
- **Blob metadata**: Information about originating detection clusters
- **Uncertainty quantification**: Standard deviations for all state estimates
- **Invalid data handling**: `-1.0` sentinel for unavailable coordinates

### Basic Usage

```python
from python_parser import parse_dopplium_tracks

# Parse tracks file
tracks, headers = parse_dopplium_tracks("tracks.bin", verbose=True)

# tracks is a structured numpy array with these fields:
# - track_id, sequence_number, status, target_class_id
# - cart_x, cart_y, cart_z, cart_vx, cart_vy, cart_vz (+ std devs)
# - enu_east, enu_north, enu_up, enu_ve, enu_vn, enu_vu (+ std devs)
# - blob_size_range, blob_size_azimuth, blob_size_elevation, blob_size_doppler
# - amplitude_db, snr_db, confidence_score
# - track_lifetime_seconds, birth_timestamp_utc_ticks, gap_count

print(f"Total track records: {len(tracks)}")
print(f"Unique tracks: {len(np.unique(tracks['track_id']))}")
```

### Filtering Tracks

```python
from python_parser import (
    filter_tracks_by_status,
    filter_tracks_by_id,
    filter_tracks_by_lifetime,
    filter_tracks_by_class,
    get_valid_coordinates
)

# Filter by status
confirmed = filter_tracks_by_status(tracks, status=1)  # 0=tentative, 1=confirmed, 2=coasting
print(f"Confirmed tracks: {len(confirmed)}")

# Get specific track's trajectory
track_101 = filter_tracks_by_id(tracks, track_id=101)
print(f"Track 101 trajectory: {len(track_101)} frames")

# Filter by lifetime
long_tracks = filter_tracks_by_lifetime(tracks, min_lifetime=2.0)
print(f"Tracks > 2s: {len(long_tracks)}")

# Get tracks with valid coordinates
valid_cart = get_valid_coordinates(tracks, 'cartesian')
valid_enu = get_valid_coordinates(tracks, 'enu')
```

### Coordinate Conversions

```python
from python_parser import cartesian_to_spherical, spherical_to_cartesian

# Convert Cartesian to spherical (RAE)
for track in tracks:
    if track['cart_x'] != -1.0:  # Check if valid
        range_m, azimuth_deg, elevation_deg = cartesian_to_spherical(
            track['cart_x'], track['cart_y'], track['cart_z']
        )
        print(f"Track {track['track_id']}: {range_m:.1f}m at {azimuth_deg:.1f}° az")

# Convert back to Cartesian
x, y, z = spherical_to_cartesian(50.0, 10.0, 5.0)  # range, azimuth, elevation
```

### Track Statistics

```python
from python_parser import (
    get_track_statistics,
    get_blob_statistics,
    get_track_lifecycle_stats
)

# Overall statistics
stats = get_track_statistics(tracks)
print(f"Status distribution: {stats['status_distribution']}")
print(f"Mean lifetime: {stats['lifetime']['mean']:.2f} s")
print(f"Valid Cartesian: {stats['valid_coordinates']['cartesian']}")

# Blob statistics
blob_stats = get_blob_statistics(tracks)
print(f"Mean detections per blob: {blob_stats['detections_per_blob']['mean']:.1f}")

# Lifecycle statistics per unique track
lifecycle = get_track_lifecycle_stats(tracks)
print(f"Mean gaps per track: {lifecycle['gaps_per_track']['mean']:.2f}")
```

### Extract Track Trajectories

```python
# Get all unique track IDs
unique_ids = np.unique(tracks['track_id'])

# Extract and analyze each track
for tid in unique_ids:
    traj = filter_tracks_by_id(tracks, tid)
    
    # Sort by frame for temporal order
    traj = traj[np.argsort(traj['frame_index'])]
    
    # Analyze trajectory
    lifetime = traj[-1]['track_lifetime_seconds']
    frames = len(traj)
    gaps = traj[-1]['gap_count']
    
    print(f"Track {tid}: {lifetime:.2f}s, {frames} frames, {gaps} gaps")
    
    # Plot position over time
    if traj[0]['cart_x'] != -1.0:
        positions = np.array([(t['cart_x'], t['cart_y'], t['cart_z']) for t in traj])
        # ... plotting code ...
```

### Coordinate System Notes

**Sensor Cartesian System (left-handed):**
- x-axis: forward (0° azimuth, 0° elevation)
- y-axis: right (90° azimuth)
- z-axis: up (90° elevation)

Conversions therefore use `azimuth = atan2(y, x)` and `elevation = asin(z / range)`.

**Invalid Data:**
- Use value `-1.0` to indicate unavailable coordinates
- Check before using: `if track['cart_x'] != -1.0:`
- Precision (std dev) fields may also be `-1.0` or `0.0`

## Data Types

All parsers handle these numpy data types:

| Code | NumPy Type | Bytes | Common Use |
|------|------------|-------|------------|
| 0 | `np.complex64` | 8 | Complex radar data (default) |
| 1 | `np.complex128` | 16 | High-precision complex |
| 2 | `np.float32` | 4 | Real-valued (magnitude/power) |
| 3 | `np.float64` | 8 | High-precision real |
| 4 | `np.int16` | 2 | Integer data (rare) |
| 5 | `np.int32` | 4 | Integer data (rare) |

## Notes

- All files use **little-endian** byte order by default
- Data arrays stored in **Fortran-order** (column-major) with range varying fastest
- Timestamps stored as .NET ticks (100ns intervals since 0001-01-01)
- String fields (node_id) are ASCII with null-termination
- All parsers validate magic numbers and header consistency

## Best Practices

1. **Use automatic dispatch for flexibility**:
   ```python
   data, headers = parse_dopplium("unknown_format.bin")
   ```

2. **Check message type after parsing**:
   ```python
   msg_type = headers['file'].message_type
   if msg_type == 2:  # RDCh
       range_axis = get_range_axis(headers)
   ```

3. **Limit data for testing**:
   ```python
   # Test with small amount of data first
   data, headers = parse_dopplium("file.bin", max_cpis_or_frames=1)
   ```

4. **Use verbose mode for debugging**:
   ```python
   data, headers = parse_dopplium("file.bin", verbose=4)  # or verbose=2/3 for less detail
   ```

5. **Access metadata through helper functions**:
   ```python
   # Preferred
   range_axis = get_range_axis(headers)
   
   # Instead of manual calculation
   # range_axis = np.linspace(...)
   ```

## See Also

- **Writers**: `../dopplium_writer/python_writer/` - Write these binary formats
- **Main Dispatcher**: `parse_dopplium.py` - Automatic format detection
- **Format Specs**: See repository documentation for detailed binary format specifications

