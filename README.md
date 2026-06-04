# Dopplium Parser

Parsers for Dopplium radar data formats in Python and MATLAB.

## Installation

Python (editable install from this folder):

```bash
pip install -e .
```

MATLAB:
- Add `matlab_parser/` to your MATLAB path.

## Supported Message Types

### Parser versions 3, 4, and 5

- `1`: ADCData
- `2`: RDCMaps (RDCh)
- `3`: RadarCube
- `4`: Detections
- `5`: Blobs
- `6`: Tracks

### Parser version 2

- `3`: RawData/ADC supported
- `1` (Detections), `2` (Tracks), `4` (Aggregated) are not implemented in the Python dispatcher

## Python Usage

### Automatic format detection (recommended)

```python
from python_parser import parse_dopplium

data, headers = parse_dopplium("file.bin", verbose=True)
print(headers["file"].message_type)
```

For ADC, RDCh, and RadarCube files, pass `start_frame` to begin at a
zero-based frame/CPI offset. With a limit, `parse_dopplium(...,
start_frame=100, max_cpis_or_frames=50)` returns the half-open window
`[100, 150)`. `start_frame` is not supported for Detections, Blobs, or Tracks.

### Direct parser calls

```python
from python_parser import (
    parse_dopplium_raw,
    parse_dopplium_rdch,
    parse_dopplium_radarcube,
    parse_dopplium_detections,
    parse_dopplium_blobs,
    parse_dopplium_tracks,
)

raw, raw_headers = parse_dopplium_raw("raw.bin", start_frame=100, max_frames=50)
rdch, rdch_headers = parse_dopplium_rdch("rdch.bin", start_frame=100, max_cpis=50)
cube, cube_headers = parse_dopplium_radarcube("radarcube.bin", start_frame=100, max_cpis=50)
detections, det_headers = parse_dopplium_detections("detections.bin")
blobs, blob_headers = parse_dopplium_blobs("blobs.bin")
tracks, track_headers = parse_dopplium_tracks("tracks.bin")
```

## MATLAB Usage

Use the MATLAB dispatcher:

```matlab
[data, headers] = doppliumParser('file.bin');
```

For ADC, RDCMaps, and RadarCube files, MATLAB also accepts a zero-based
`startFrame` option:

```matlab
opts.startFrame = 100;
opts.maxFrames = 50;
[data, headers] = doppliumParser('file.bin', opts);
```

Or call format-specific functions in `matlab_parser/`:
- `parseADCData`
- `parseRDCMaps`
- `parseRadarCube`
- `parseDetections`
- `parseBlobs`
- `parseTracks`

## Data Shapes (Python)

- ADCData/RawData: `[samples, chirpsPerTx, channels, frames]`
- RDCh: `[range_bins, doppler_bins, channels, cpis]`
- RadarCube: `[range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]`
- Detections/Blobs/Tracks: numpy structured arrays

## Examples

- Python: `python_parser/example_parse.py`, `python_parser/example_parse_detections.py`, `python_parser/example_parse_tracks.py`
- MATLAB: `matlab_parser/exampleADCDataParse.m`, `matlab_parser/exampleRDCMapsDataParse.m`

For full Python parser details, see `python_parser/README.md`.
