# Dopplium Parser

Parsers for Dopplium radar data formats (MATLAB & Python).

## Installation

**MATLAB**: Add `matlab/` to your path.

**Python**: 
```bash
pip install numpy matplotlib
```

## Usage

**MATLAB**:
```matlab
[data, headers] = parseDoppliumRaw('file.bin');
```

**Python**:
```python
from parse_dopplium_raw import parse_dopplium_raw
data, headers = parse_dopplium_raw('file.bin')
```

## Data Format

Returns data shaped `[samples, chirpsPerTx, channels, frames]`:
- **Samples**: ADC samples per chirp
- **ChirpsPerTx**: Chirps per transmitter per frame  
- **Channels**: Receiver index (single TX) or TX-RX combinations (multi TX)
- **Frames**: Number of radar frames

## Options

- `maxFrames`/`max_frames`: Limit frames to read
- `cast`: Output type ('single'/'float32', 'double'/'float64', 'int16')
- `returnComplex`/`return_complex`: Return complex numbers for IQ data
- `verbose`: Show parsing info

## Examples

See `matlab/exampleParse.m` and `python/example_parse.py` for complete examples with 2D FFT processing and visualization.