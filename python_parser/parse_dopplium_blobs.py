"""
Parser for Dopplium Blobs binary format.

Supports:
- Version 3, message_type 5: Blobs data (clustered detections)

Reads blob data files written by BlobsBinaryWriter.
Returns numpy structured array containing all blob fields and headers.

Format Notes:
- Body header: 64 bytes (magic "BLOB")
- Batch header: 30 bytes per batch (magic "BTCH")
- Blob record: 56 bytes per blob
- Blob records contain centroid position, spread, amplitude, 
  cell indices, and identification information
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
class BlobsBodyHeader:
    """Blobs body header (64 bytes) containing clustering algorithm metadata."""
    body_magic: str
    body_header_version: int
    body_header_size: int
    batch_header_size: int
    blob_record_size: int
    algorithm_id: int
    algorithm_version: int
    _reserved: bytes


@dataclass
class BlobsBatchHeader:
    """Blobs batch header (30 bytes) per-batch metadata."""
    batch_magic: str
    batch_header_size: int
    timestamp_utc_ticks: int
    reserved1: int
    blob_count: int
    sequence_number: int
    payload_size_bytes: int
    reserved2: int


# ==============================
# Public API
# ==============================

def parse_dopplium_blobs(
    filename: str,
    *,
    max_batches: Optional[int] = None,
    verbose: int = 4,
    _file_header: Optional[FileHeader] = None,
    _endian_prefix: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium Blobs file into a numpy structured array.
    
    Parameters:
    -----------
    filename : str
        Path to the Blobs binary file
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
    tuple : (blobs, headers)
        blobs : np.ndarray
            Structured array with fields:
            - range_centroid (float64): Centroid range in meters
            - velocity_centroid (float64): Centroid radial velocity in m/s
            - azimuth_centroid (float64): Centroid azimuth angle in degrees
            - elevation_centroid (float64): Centroid elevation angle in degrees
            - range_spread (float32): Range extent in meters
            - velocity_spread (float32): Velocity extent in m/s
            - azimuth_spread (float32): Azimuth extent in degrees
            - elevation_spread (float32): Elevation extent in degrees
            - amplitude (float64): Aggregate amplitude
            - range_cell (int16): Range bin index of centroid
            - doppler_cell (int16): Doppler bin index of centroid
            - azimuth_cell (int16): Azimuth bin index of centroid
            - elevation_cell (int16): Elevation bin index of centroid
            - blob_id (uint32): Unique blob identifier within batch
            - num_detections (uint16): Number of detections in this blob
            - batch_index (uint32): Which batch this blob came from
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
        
        if FH.message_type != 5:
            raise ValueError(f"This file is not Blobs (message_type={FH.message_type}, expected 5).")
        
        # Position at body header if we parsed the file header ourselves
        if _file_header is None:
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        BH = _read_blobs_body_header(f, endian_prefix)
        
        if verbose >= 2:
            _print_header_summary(FH, BH)
        
        # Determine number of batches from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        
        if verbose >= 2:
            print(f"\nBytes after headers: {bytes_after_headers}")
        
        # We'll read batch by batch since blob counts vary
        # Estimate number of batches (rough estimate)
        avg_batch_size = BH.batch_header_size + (5 * BH.blob_record_size)  # Assume ~5 blobs per batch
        n_batches_estimate = max(1, bytes_after_headers // avg_batch_size)
        
        if verbose >= 2:
            print(f"Estimated batches in file: ~{n_batches_estimate}")
        
        # Read batches
        f.seek(FH.file_header_size + BH.body_header_size, io.SEEK_SET)
        batch_headers = []
        all_blobs = []
        
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
                          f"blobs={batch_header.blob_count}, "
                          f"size={batch_header.payload_size_bytes} bytes")

                # Validate payload size
                expected_payload_size = batch_header.blob_count * BH.blob_record_size
                if batch_header.payload_size_bytes != expected_payload_size:
                    print(f"Warning: Batch {batches_read} payload size mismatch: "
                          f"expected={expected_payload_size}, got={batch_header.payload_size_bytes}")
                
                # Read blob records
                if batch_header.blob_count > 0:
                    blobs = _read_blob_records(
                        f, 
                        endian_prefix, 
                        batch_header.blob_count,
                        batch_index=batches_read,
                        sequence_number=batch_header.sequence_number
                    )
                    all_blobs.append(blobs)
                
                batches_read += 1
                
            except EOFError:
                print(f"Reached end of file after reading {batches_read} batches.")
                break
            except struct.error as e:
                print(f"Struct error after reading {batches_read} batches: {e}")
                break
        
        # Combine all blobs
        if all_blobs:
            data = np.concatenate(all_blobs)
        else:
            # Create empty array with correct dtype
            data = _create_blob_dtype(0, 0, 0)
        
        headers = {
            "file": FH,
            "body": BH,
            "batch": batch_headers,
        }
        
        if verbose >= 1:
            print(f"\nTotal batches read: {batches_read}")
            print(f"Total blobs: {len(data)}")
            if len(data) > 0:
                print(f"Blob array shape: {data.shape}")
                print(f"Blob fields: {list(data.dtype.names)}")
        
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


def _read_blobs_body_header(f: io.BufferedReader, ep: str) -> BlobsBodyHeader:
    """Read Blobs body header (64 bytes)."""
    fmt = (
        f"{ep}"
        "4s"   # body_magic (BLOB)
        "H"    # body_header_version
        "H"    # body_header_size
        "H"    # batch_header_size
        "H"    # blob_record_size
        "I"    # algorithm_id
        "I"    # algorithm_version
        "44s"  # reserved
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read body header. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    return BlobsBodyHeader(
        body_magic=unpacked[0].decode("ascii"),
        body_header_version=unpacked[1],
        body_header_size=unpacked[2],
        batch_header_size=unpacked[3],
        blob_record_size=unpacked[4],
        algorithm_id=unpacked[5],
        algorithm_version=unpacked[6],
        _reserved=unpacked[7],
    )


def _read_batch_header(f: io.BufferedReader, ep: str) -> BlobsBatchHeader:
    """Read Blobs batch header (30 bytes)."""
    fmt = f"{ep}4sHqHHIII"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read batch header.")
    
    (batch_magic_b, batch_header_size, timestamp_utc_ticks,
     reserved1, blob_count, sequence_number, payload_size_bytes, 
     reserved2) = struct.unpack(fmt, raw)
    
    batch_magic = batch_magic_b.decode("ascii")
    if batch_magic != "BTCH":
        raise ValueError(f"Invalid batch magic: expected 'BTCH', got '{batch_magic}'")
    
    return BlobsBatchHeader(
        batch_magic=batch_magic,
        batch_header_size=batch_header_size,
        timestamp_utc_ticks=timestamp_utc_ticks,
        reserved1=reserved1,
        blob_count=blob_count,
        sequence_number=sequence_number,
        payload_size_bytes=payload_size_bytes,
        reserved2=reserved2,
    )


def _create_blob_dtype(batch_idx: int, seq_num: int, count: int) -> np.ndarray:
    """Create numpy array with blob dtype."""
    dtype = np.dtype([
        # Centroids (16 bytes)
        ('range_centroid', np.float32),
        ('velocity_centroid', np.float32),
        ('azimuth_centroid', np.float32),
        ('elevation_centroid', np.float32),
        # Spreads (16 bytes)
        ('range_spread', np.float32),
        ('velocity_spread', np.float32),
        ('azimuth_spread', np.float32),
        ('elevation_spread', np.float32),
        # Amplitude (4 bytes)
        ('amplitude', np.float32),
        # Cell indices (8 bytes)
        ('range_cell', np.int16),
        ('doppler_cell', np.int16),
        ('azimuth_cell', np.int16),
        ('elevation_cell', np.int16),
        # Identity (8 bytes - uint32 + uint16 + padding)
        ('blob_id', np.uint32),
        ('num_detections', np.uint16),
        # Batch metadata
        ('batch_index', np.uint32),
        ('sequence_number', np.uint32),
    ])
    return np.zeros(count, dtype=dtype)


def _read_blob_records(
    f: io.BufferedReader, 
    ep: str, 
    count: int,
    batch_index: int,
    sequence_number: int
) -> np.ndarray:
    """
    Read blob records from file.
    
    Each record is 56 bytes:
    - 4 floats (4 bytes each) = 16 bytes: centroids (range, velocity, azimuth, elevation)
    - 4 floats (4 bytes each) = 16 bytes: spreads
    - 1 float (4 bytes) = 4 bytes: amplitude
    - 4 int16s (2 bytes each) = 8 bytes: cell indices
    - 1 uint32 (4 bytes) = 4 bytes: blob_id
    - 1 uint16 (2 bytes) = 2 bytes: num_detections
    - 2 bytes padding
    - 4 bytes reserved
    """
    # Read all records at once
    # Format: 9f (4 centroids + 4 spreads + 1 amplitude) + 4h (cells) + I (blob_id) + H (num_detections) + 2x (padding) + 4x (reserved)
    single_fmt = "fffffffffhhhhIH2x4x"
    fmt = f"{ep}" + single_fmt * count
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read {count} blob records. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    # Create structured array
    blobs = _create_blob_dtype(batch_index, sequence_number, count)
    
    # Each blob is 15 values (4 centroids + 4 spreads + 1 amplitude + 4 cells + 1 blob_id + 1 num_detections)
    values_per_blob = 4 + 4 + 1 + 4 + 1 + 1  # = 15 values
    
    for i in range(count):
        idx = i * values_per_blob
        # Centroids
        blobs[i]['range_centroid'] = unpacked[idx + 0]
        blobs[i]['velocity_centroid'] = unpacked[idx + 1]
        blobs[i]['azimuth_centroid'] = unpacked[idx + 2]
        blobs[i]['elevation_centroid'] = unpacked[idx + 3]
        # Spreads
        blobs[i]['range_spread'] = unpacked[idx + 4]
        blobs[i]['velocity_spread'] = unpacked[idx + 5]
        blobs[i]['azimuth_spread'] = unpacked[idx + 6]
        blobs[i]['elevation_spread'] = unpacked[idx + 7]
        # Amplitude
        blobs[i]['amplitude'] = unpacked[idx + 8]
        # Cell indices
        blobs[i]['range_cell'] = unpacked[idx + 9]
        blobs[i]['doppler_cell'] = unpacked[idx + 10]
        blobs[i]['azimuth_cell'] = unpacked[idx + 11]
        blobs[i]['elevation_cell'] = unpacked[idx + 12]
        # Identity
        blobs[i]['blob_id'] = unpacked[idx + 13]
        blobs[i]['num_detections'] = unpacked[idx + 14]
        # Batch metadata
        blobs[i]['batch_index'] = batch_index
        blobs[i]['sequence_number'] = sequence_number
    
    return blobs


def _print_header_summary(FH: FileHeader, BH: BlobsBodyHeader) -> None:
    """Print summary of file and body headers."""
    print("--- Dopplium Blobs Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  "
          f"Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  "
          f"BatchHdr={BH.batch_header_size}  TotalBatchesWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    
    print("\n-- Blobs Configuration --")
    print(f"Blob record size: {BH.blob_record_size} bytes")
    print(f"Algorithm ID: {BH.algorithm_id}")
    print(f"Algorithm version: {BH.algorithm_version}")
    print(f"Body header version: {BH.body_header_version}")


# ==============================
# Convenience functions
# ==============================

def filter_blobs_by_range(
    blobs: np.ndarray, 
    min_range: float, 
    max_range: float
) -> np.ndarray:
    """
    Filter blobs by centroid range.
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    min_range : float
        Minimum range in meters (inclusive)
    max_range : float
        Maximum range in meters (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered blobs
    """
    mask = (blobs['range_centroid'] >= min_range) & (blobs['range_centroid'] <= max_range)
    return blobs[mask]


def filter_blobs_by_velocity(
    blobs: np.ndarray, 
    min_velocity: float, 
    max_velocity: float
) -> np.ndarray:
    """
    Filter blobs by centroid velocity.
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    min_velocity : float
        Minimum velocity in m/s (inclusive)
    max_velocity : float
        Maximum velocity in m/s (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered blobs
    """
    mask = (blobs['velocity_centroid'] >= min_velocity) & (blobs['velocity_centroid'] <= max_velocity)
    return blobs[mask]


def filter_blobs_by_amplitude(
    blobs: np.ndarray, 
    min_amplitude: float
) -> np.ndarray:
    """
    Filter blobs by amplitude threshold.
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    min_amplitude : float
        Minimum amplitude threshold
    
    Returns:
    --------
    np.ndarray : Filtered blobs
    """
    mask = blobs['amplitude'] >= min_amplitude
    return blobs[mask]


def filter_blobs_by_size(
    blobs: np.ndarray,
    min_range_spread: float = 0.0,
    max_range_spread: float = np.inf,
    min_velocity_spread: float = 0.0,
    max_velocity_spread: float = np.inf
) -> np.ndarray:
    """
    Filter blobs by size (spread).
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    min_range_spread : float
        Minimum range spread in meters (inclusive)
    max_range_spread : float
        Maximum range spread in meters (inclusive)
    min_velocity_spread : float
        Minimum velocity spread in m/s (inclusive)
    max_velocity_spread : float
        Maximum velocity spread in m/s (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered blobs
    """
    mask = (
        (blobs['range_spread'] >= min_range_spread) & 
        (blobs['range_spread'] <= max_range_spread) &
        (blobs['velocity_spread'] >= min_velocity_spread) & 
        (blobs['velocity_spread'] <= max_velocity_spread)
    )
    return blobs[mask]


def filter_blobs_by_detection_count(
    blobs: np.ndarray,
    min_detections: int = 1,
    max_detections: int = 65535
) -> np.ndarray:
    """
    Filter blobs by number of detections.
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    min_detections : int
        Minimum number of detections (inclusive)
    max_detections : int
        Maximum number of detections (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered blobs
    """
    mask = (blobs['num_detections'] >= min_detections) & (blobs['num_detections'] <= max_detections)
    return blobs[mask]


def get_blob_statistics(blobs: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about blobs.
    
    Parameters:
    -----------
    blobs : np.ndarray
        Structured array from parse_dopplium_blobs
    
    Returns:
    --------
    dict : Statistics including min, max, mean, std for each field
    """
    if len(blobs) == 0:
        return {"count": 0}
    
    return {
        "count": len(blobs),
        "range_centroid": {
            "min": float(np.min(blobs['range_centroid'])),
            "max": float(np.max(blobs['range_centroid'])),
            "mean": float(np.mean(blobs['range_centroid'])),
            "std": float(np.std(blobs['range_centroid'])),
        },
        "velocity_centroid": {
            "min": float(np.min(blobs['velocity_centroid'])),
            "max": float(np.max(blobs['velocity_centroid'])),
            "mean": float(np.mean(blobs['velocity_centroid'])),
            "std": float(np.std(blobs['velocity_centroid'])),
        },
        "amplitude": {
            "min": float(np.min(blobs['amplitude'])),
            "max": float(np.max(blobs['amplitude'])),
            "mean": float(np.mean(blobs['amplitude'])),
            "std": float(np.std(blobs['amplitude'])),
        },
        "azimuth_centroid": {
            "min": float(np.min(blobs['azimuth_centroid'])),
            "max": float(np.max(blobs['azimuth_centroid'])),
            "mean": float(np.mean(blobs['azimuth_centroid'])),
            "std": float(np.std(blobs['azimuth_centroid'])),
        },
        "elevation_centroid": {
            "min": float(np.min(blobs['elevation_centroid'])),
            "max": float(np.max(blobs['elevation_centroid'])),
            "mean": float(np.mean(blobs['elevation_centroid'])),
            "std": float(np.std(blobs['elevation_centroid'])),
        },
        "range_spread": {
            "min": float(np.min(blobs['range_spread'])),
            "max": float(np.max(blobs['range_spread'])),
            "mean": float(np.mean(blobs['range_spread'])),
            "std": float(np.std(blobs['range_spread'])),
        },
        "velocity_spread": {
            "min": float(np.min(blobs['velocity_spread'])),
            "max": float(np.max(blobs['velocity_spread'])),
            "mean": float(np.mean(blobs['velocity_spread'])),
            "std": float(np.std(blobs['velocity_spread'])),
        },
        "num_detections": {
            "min": int(np.min(blobs['num_detections'])),
            "max": int(np.max(blobs['num_detections'])),
            "mean": float(np.mean(blobs['num_detections'])),
            "total": int(np.sum(blobs['num_detections'])),
        },
        "batches": {
            "unique_batches": len(np.unique(blobs['batch_index'])),
            "unique_sequences": len(np.unique(blobs['sequence_number'])),
        }
    }

