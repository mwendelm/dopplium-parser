"""
Parser for Dopplium Detections binary format.

Supports:
- Version 3, message_type 4: Detections data

Reads detection data files written by DetectionsBinaryWriter.
Returns numpy structured array containing all detection fields and headers.

Format Notes:
- Body header: 64 bytes (magic "DETC")
- Payload header: 32 bytes per batch (magic "BTCH")
- Detection record: 56 bytes per detection
- Detection records contain range, velocity, azimuth, elevation, amplitude, 
  monopulse ratios, and grid cell indices
"""

from __future__ import annotations
import io
import struct
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

from ._verbose import _validate_verbose
from .parse_dopplium_header import FileHeader, parse_file_header


# ==============================
# Dataclasses for headers
# ==============================


@dataclass
class DetectionsBodyHeader:
    """Detections body header (64 bytes) containing configuration and metadata."""
    body_magic: str
    body_header_version: int
    body_header_size: int
    payload_header_size: int
    detection_record_size: int
    algorithm_id: int
    algorithm_version: int
    _reserved: bytes


@dataclass
class DetectionsBatchHeader:
    """Detections batch/payload header (32 bytes)."""
    payload_magic: str
    payload_header_size: int
    timestamp_utc_ticks: int
    reserved1: int
    detection_count: int
    sequence_number: int
    payload_size_bytes: int
    reserved2: int


# ==============================
# Public API
# ==============================

def parse_dopplium_detections(
    filename: str,
    *,
    max_batches: Optional[int] = None,
    verbose: int = 4,
    _file_header: Optional[FileHeader] = None,
    _endian_prefix: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium Detections file into a numpy structured array.
    
    Parameters:
    -----------
    filename : str
        Path to the Detections binary file
    max_batches : int, optional
        Maximum number of batches to read (None = all batches)
    verbose : int
        Verbosity level from 0 (nothing) to 4 (everything, default)
    _file_header : FileHeader, optional
        Pre-parsed file header (internal use)
    _endian_prefix : str, optional
        Endianness prefix (internal use)
    
    Returns:
    --------
    tuple : (detections, headers)
        detections : np.ndarray
            Structured array with fields:
            - range (float64): Range in meters
            - velocity (float64): Radial velocity in m/s
            - azimuth (float64): Azimuth angle in degrees
            - elevation (float64): Elevation angle in degrees
            - amplitude (float64): Detection amplitude
            - monopulse_ratio_az (float32): Azimuth monopulse ratio
            - monopulse_ratio_el (float32): Elevation monopulse ratio
            - range_cell (int16): Range bin index
            - doppler_cell (int16): Doppler bin index
            - azimuth_cell (int16): Azimuth bin index
            - elevation_cell (int16): Elevation bin index
            - batch_index (uint32): Which batch this detection came from
            - sequence_number (uint32): Sequence number of the batch
        headers : dict
            Dictionary containing 'file', 'body', and 'batch' headers
    """
    verbose = _validate_verbose(verbose)

    with open(filename, "rb") as f:
        # Use provided header or parse it
        if _file_header is None or _endian_prefix is None:
            FH, endian_prefix = parse_file_header(filename)
        else:
            FH = _file_header
            endian_prefix = _endian_prefix
            # Seek to start of file to read headers
            f.seek(0, io.SEEK_SET)
            # Skip the file header
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        if FH.message_type != 4:
            raise ValueError(f"This file is not Detections (message_type={FH.message_type}, expected 4).")
        
        # Position at body header if we parsed the file header ourselves
        if _file_header is None:
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        BH = _read_detections_body_header(f, endian_prefix)
        
        if verbose >= 2:
            _print_header_summary(FH, BH)
        
        # Determine number of batches from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        
        if verbose >= 2:
            print(f"\nBytes after headers: {bytes_after_headers}")
        
        # We'll read batch by batch since detection counts vary
        # Estimate number of batches (rough estimate)
        avg_batch_size = BH.payload_header_size + (10 * BH.detection_record_size)  # Assume ~10 detections per batch
        n_batches_estimate = max(1, bytes_after_headers // avg_batch_size)
        
        if verbose >= 2:
            print(f"Estimated batches in file: ~{n_batches_estimate}")
        
        n_batches = n_batches_estimate if max_batches is None else max_batches
        
        # Read batches
        f.seek(FH.file_header_size + BH.body_header_size, io.SEEK_SET)
        batch_headers = []
        all_detections = []
        
        batches_read = 0
        while True:
            # Check if we've read enough batches
            if max_batches is not None and batches_read >= max_batches:
                break
            
            try:
                # Check if we have enough bytes left
                current_pos = f.tell()
                if current_pos >= file_size:
                    break
                
                # Try to read batch header
                batch_header = _read_batch_header(f, endian_prefix)
                batch_headers.append(batch_header)
                
                if verbose >= 3 and (batches_read == 0 or (batches_read + 1) % 100 == 0):
                    print(f"  Reading batch {batches_read + 1}: "
                          f"seq={batch_header.sequence_number}, "
                          f"detections={batch_header.detection_count}, "
                          f"size={batch_header.payload_size_bytes} bytes")

                # Validate payload size
                expected_payload_size = batch_header.detection_count * BH.detection_record_size
                if batch_header.payload_size_bytes != expected_payload_size:
                    print(f"Warning: Batch {batches_read} payload size mismatch: "
                          f"expected={expected_payload_size}, got={batch_header.payload_size_bytes}")
                
                # Read detection records
                if batch_header.detection_count > 0:
                    detections = _read_detection_records(
                        f, 
                        endian_prefix, 
                        batch_header.detection_count,
                        batch_index=batches_read,
                        sequence_number=batch_header.sequence_number
                    )
                    all_detections.append(detections)
                
                batches_read += 1
                
            except EOFError:
                print(f"Reached end of file after reading {batches_read} batches.")
                break
            except struct.error as e:
                print(f"Struct error after reading {batches_read} batches: {e}")
                break
        
        # Combine all detections
        if all_detections:
            data = np.concatenate(all_detections)
        else:
            # Create empty array with correct dtype
            data = _create_detection_dtype(0, 0, 0)
        
        headers = {
            "file": FH,
            "body": BH,
            "batch": batch_headers,
        }
        
        if verbose >= 1:
            print(f"\nTotal batches read: {batches_read}")
            print(f"Total detections: {len(data)}")
            if len(data) > 0:
                print(f"Detection array shape: {data.shape}")
                print(f"Detection fields: {list(data.dtype.names)}")
        
        return data, headers


# ==============================
# Helpers
# ==============================

def _file_size(fh: io.BufferedReader) -> int:
    """Get file size without changing current position."""
    cur = fh.tell()
    fh.seek(0, io.SEEK_END)
    size = fh.tell()
    fh.seek(cur, io.SEEK_SET)
    return size


def _read_detections_body_header(f: io.BufferedReader, ep: str) -> DetectionsBodyHeader:
    """Read Detections body header (64 bytes)."""
    fmt = (
        f"{ep}"
        "4s"   # body_magic (DETC)
        "H"    # body_header_version
        "H"    # body_header_size
        "H"    # payload_header_size
        "H"    # detection_record_size
        "I"    # algorithm_id
        "I"    # algorithm_version
        "44s"  # reserved
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read body header. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    return DetectionsBodyHeader(
        body_magic=unpacked[0].decode("ascii"),
        body_header_version=unpacked[1],
        body_header_size=unpacked[2],
        payload_header_size=unpacked[3],
        detection_record_size=unpacked[4],
        algorithm_id=unpacked[5],
        algorithm_version=unpacked[6],
        _reserved=unpacked[7],
    )


def _read_batch_header(f: io.BufferedReader, ep: str) -> DetectionsBatchHeader:
    """Read Detections batch/payload header (32 bytes)."""
    fmt = f"{ep}4sHqHIIII"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read batch header.")
    
    (payload_magic_b, payload_header_size, timestamp_utc_ticks,
     reserved1, detection_count, sequence_number, payload_size_bytes, 
     reserved2) = struct.unpack(fmt, raw)
    
    payload_magic = payload_magic_b.decode("ascii")
    if payload_magic != "BTCH":
        raise ValueError(f"Invalid batch magic: expected 'BTCH', got '{payload_magic}'")
    
    return DetectionsBatchHeader(
        payload_magic=payload_magic,
        payload_header_size=payload_header_size,
        timestamp_utc_ticks=timestamp_utc_ticks,
        reserved1=reserved1,
        detection_count=detection_count,
        sequence_number=sequence_number,
        payload_size_bytes=payload_size_bytes,
        reserved2=reserved2,
    )


def _create_detection_dtype(batch_idx: int, seq_num: int, count: int) -> np.ndarray:
    """Create numpy array with detection dtype."""
    dtype = np.dtype([
        ('range', np.float64),
        ('velocity', np.float64),
        ('azimuth', np.float64),
        ('elevation', np.float64),
        ('amplitude', np.float64),
        ('monopulse_ratio_az', np.float32),
        ('monopulse_ratio_el', np.float32),
        ('range_cell', np.int16),
        ('doppler_cell', np.int16),
        ('azimuth_cell', np.int16),
        ('elevation_cell', np.int16),
        ('batch_index', np.uint32),
        ('sequence_number', np.uint32),
    ])
    return np.zeros(count, dtype=dtype)


def _read_detection_records(
    f: io.BufferedReader, 
    ep: str, 
    count: int,
    batch_index: int,
    sequence_number: int
) -> np.ndarray:
    """
    Read detection records from file.
    
    Each record is 56 bytes:
    - 5 doubles (8 bytes each) = 40 bytes: range, velocity, azimuth, elevation, amplitude
    - 2 floats (4 bytes each) = 8 bytes: monopulse ratios
    - 4 int16s (2 bytes each) = 8 bytes: cell indices
    """
    # Read all records at once
    fmt = f"{ep}" + "dddddffhhhh" * count
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read {count} detection records. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    # Create structured array
    detections = _create_detection_dtype(batch_index, sequence_number, count)
    
    # Fill in the data (each detection is 11 values)
    for i in range(count):
        idx = i * 11
        detections[i]['range'] = unpacked[idx + 0]
        detections[i]['velocity'] = unpacked[idx + 1]
        detections[i]['azimuth'] = unpacked[idx + 2]
        detections[i]['elevation'] = unpacked[idx + 3]
        detections[i]['amplitude'] = unpacked[idx + 4]
        detections[i]['monopulse_ratio_az'] = unpacked[idx + 5]
        detections[i]['monopulse_ratio_el'] = unpacked[idx + 6]
        detections[i]['range_cell'] = unpacked[idx + 7]
        detections[i]['doppler_cell'] = unpacked[idx + 8]
        detections[i]['azimuth_cell'] = unpacked[idx + 9]
        detections[i]['elevation_cell'] = unpacked[idx + 10]
        detections[i]['batch_index'] = batch_index
        detections[i]['sequence_number'] = sequence_number
    
    return detections


def _print_header_summary(FH: FileHeader, BH: DetectionsBodyHeader) -> None:
    """Print summary of file and body headers."""
    print("--- Dopplium Detections Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  "
          f"Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  "
          f"BatchHdr={BH.payload_header_size}  TotalBatchesWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    
    print("\n-- Detections Configuration --")
    print(f"Detection record size: {BH.detection_record_size} bytes")
    print(f"Algorithm ID: {BH.algorithm_id}")
    print(f"Algorithm version: {BH.algorithm_version}")
    print(f"Body header version: {BH.body_header_version}")


# ==============================
# Convenience functions
# ==============================

def filter_detections_by_range(
    detections: np.ndarray, 
    min_range: float, 
    max_range: float
) -> np.ndarray:
    """
    Filter detections by range.
    
    Parameters:
    -----------
    detections : np.ndarray
        Structured array from parse_dopplium_detections
    min_range : float
        Minimum range in meters (inclusive)
    max_range : float
        Maximum range in meters (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered detections
    """
    mask = (detections['range'] >= min_range) & (detections['range'] <= max_range)
    return detections[mask]


def filter_detections_by_velocity(
    detections: np.ndarray, 
    min_velocity: float, 
    max_velocity: float
) -> np.ndarray:
    """
    Filter detections by velocity.
    
    Parameters:
    -----------
    detections : np.ndarray
        Structured array from parse_dopplium_detections
    min_velocity : float
        Minimum velocity in m/s (inclusive)
    max_velocity : float
        Maximum velocity in m/s (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered detections
    """
    mask = (detections['velocity'] >= min_velocity) & (detections['velocity'] <= max_velocity)
    return detections[mask]


def filter_detections_by_amplitude(
    detections: np.ndarray, 
    min_amplitude: float
) -> np.ndarray:
    """
    Filter detections by amplitude threshold.
    
    Parameters:
    -----------
    detections : np.ndarray
        Structured array from parse_dopplium_detections
    min_amplitude : float
        Minimum amplitude threshold
    
    Returns:
    --------
    np.ndarray : Filtered detections
    """
    mask = detections['amplitude'] >= min_amplitude
    return detections[mask]


def get_detection_statistics(detections: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about detections.
    
    Parameters:
    -----------
    detections : np.ndarray
        Structured array from parse_dopplium_detections
    
    Returns:
    --------
    dict : Statistics including min, max, mean, std for each field
    """
    if len(detections) == 0:
        return {"count": 0}
    
    return {
        "count": len(detections),
        "range": {
            "min": float(np.min(detections['range'])),
            "max": float(np.max(detections['range'])),
            "mean": float(np.mean(detections['range'])),
            "std": float(np.std(detections['range'])),
        },
        "velocity": {
            "min": float(np.min(detections['velocity'])),
            "max": float(np.max(detections['velocity'])),
            "mean": float(np.mean(detections['velocity'])),
            "std": float(np.std(detections['velocity'])),
        },
        "amplitude": {
            "min": float(np.min(detections['amplitude'])),
            "max": float(np.max(detections['amplitude'])),
            "mean": float(np.mean(detections['amplitude'])),
            "std": float(np.std(detections['amplitude'])),
        },
        "azimuth": {
            "min": float(np.min(detections['azimuth'])),
            "max": float(np.max(detections['azimuth'])),
            "mean": float(np.mean(detections['azimuth'])),
            "std": float(np.std(detections['azimuth'])),
        },
        "elevation": {
            "min": float(np.min(detections['elevation'])),
            "max": float(np.max(detections['elevation'])),
            "mean": float(np.mean(detections['elevation'])),
            "std": float(np.std(detections['elevation'])),
        },
        "batches": {
            "unique_batches": len(np.unique(detections['batch_index'])),
            "unique_sequences": len(np.unique(detections['sequence_number'])),
        }
    }

