"""
Parser for Dopplium Tracks binary format.

Supports:
- Version 3, message_type 6: Tracks data

Reads track data files written by TracksBinaryWriter.
Returns numpy structured array containing all track fields and headers.

Format Notes:
- Body header: 96 bytes (magic "TRCK")
- Frame header: 30 bytes per frame (magic "FRME")
- Track record: 160 bytes per track
- Track records contain position, velocity, precision in two coordinate systems
  (sensor Cartesian and ENU), plus blob metadata and lifecycle information
- Sensor Cartesian is left-handed: x = forward (0° az, 0° el), y = right (90° az),
  z = up (90° el)
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
class TracksBodyHeader:
    """Tracks body header (96 bytes) containing tracking algorithm metadata."""
    body_magic: str
    body_header_version: int
    body_header_size: int
    frame_header_size: int
    track_record_size: int
    association_algorithm_id: int
    association_algorithm_version: int
    tracker_algorithm_id: int
    tracker_algorithm_version: int
    track_management_algorithm_id: int
    track_management_algorithm_version: int
    association_params: bytes
    tracker_params: bytes
    track_management_params: bytes


@dataclass
class TracksFrameHeader:
    """Tracks frame header (30 bytes) per-frame metadata."""
    frame_magic: str
    frame_header_size: int
    timestamp_utc_ticks: int
    num_tracks: int
    num_new_tracks: int
    num_terminated_tracks: int
    sequence_number: int
    payload_size_bytes: int
    flags: int


# ==============================
# Public API
# ==============================

def parse_dopplium_tracks(
    filename: str,
    *,
    max_frames: Optional[int] = None,
    verbose: int = 4,
    _file_header: Optional[FileHeader] = None,
    _endian_prefix: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium Tracks file into a numpy structured array.
    
    Parameters:
    -----------
    filename : str
        Path to the Tracks binary file
    max_frames : int, optional
        Maximum number of frames to read (None = all frames)
    verbose : int
        Verbosity level from 0 (nothing) to 4 (everything, default)
    _file_header : FileHeader, optional
        Pre-parsed file header (internal use)
    _endian_prefix : str, optional
        Endianness prefix (internal use)
    
    Returns:
    --------
    tuple : (tracks, headers)
        tracks : np.ndarray
            Structured array with fields for track state, coordinates, and metadata
        headers : dict
            Dictionary containing 'file', 'body', and 'frame' headers
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
        
        if FH.message_type != 6:
            raise ValueError(f"This file is not Tracks (message_type={FH.message_type}, expected 6).")
        
        # Position at body header if we parsed the file header ourselves
        if _file_header is None:
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        BH = _read_tracks_body_header(f, endian_prefix)
        
        if verbose >= 2:
            _print_header_summary(FH, BH)
        
        # Determine number of frames from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        
        if verbose >= 2:
            print(f"\nBytes after headers: {bytes_after_headers}")
        
        # We'll read frame by frame since track counts vary
        # Estimate number of frames
        avg_frame_size = BH.frame_header_size + (10 * BH.track_record_size)  # Assume ~10 tracks per frame
        n_frames_estimate = max(1, bytes_after_headers // avg_frame_size)
        
        if verbose >= 2:
            print(f"Estimated frames in file: ~{n_frames_estimate}")
        
        # Read frames
        f.seek(FH.file_header_size + BH.body_header_size, io.SEEK_SET)
        frame_headers = []
        all_tracks = []
        
        frames_read = 0
        while True:
            # Check if we've read enough frames
            if max_frames is not None and frames_read >= max_frames:
                break
            
            try:
                # Check if we have enough bytes left
                current_pos = f.tell()
                if current_pos >= file_size:
                    break
                
                # Try to read frame header
                frame_header = _read_frame_header(f, endian_prefix)
                frame_headers.append(frame_header)
                
                if verbose >= 3 and (frames_read == 0 or (frames_read + 1) % 100 == 0):
                    print(f"  Reading frame {frames_read + 1}: "
                          f"seq={frame_header.sequence_number}, "
                          f"tracks={frame_header.num_tracks}, "
                          f"size={frame_header.payload_size_bytes} bytes")

                # Validate payload size
                expected_payload_size = frame_header.num_tracks * BH.track_record_size
                if frame_header.payload_size_bytes != expected_payload_size:
                    print(f"Warning: Frame {frames_read} payload size mismatch: "
                          f"expected={expected_payload_size}, got={frame_header.payload_size_bytes}")
                
                # Read track records
                if frame_header.num_tracks > 0:
                    tracks = _read_track_records(
                        f, 
                        endian_prefix, 
                        frame_header.num_tracks,
                        frame_index=frames_read,
                        sequence_number=frame_header.sequence_number
                    )
                    all_tracks.append(tracks)
                
                frames_read += 1
                
            except EOFError:
                print(f"Reached end of file after reading {frames_read} frames.")
                break
            except struct.error as e:
                print(f"Struct error after reading {frames_read} frames: {e}")
                break
        
        # Combine all tracks
        if all_tracks:
            data = np.concatenate(all_tracks)
        else:
            # Create empty array with correct dtype
            data = _create_track_dtype(0, 0, 0)
        
        headers = {
            "file": FH,
            "body": BH,
            "frame": frame_headers,
        }
        
        if verbose >= 1:
            print(f"\nTotal frames read: {frames_read}")
            print(f"Total tracks: {len(data)}")
            if len(data) > 0:
                print(f"Track array shape: {data.shape}")
                print(f"Track fields: {list(data.dtype.names)}")
        
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


def _read_tracks_body_header(f: io.BufferedReader, ep: str) -> TracksBodyHeader:
    """Read Tracks body header (96 bytes)."""
    fmt = (
        f"{ep}"
        "4s"   # body_magic (TRCK)
        "H"    # body_header_version
        "H"    # body_header_size
        "H"    # frame_header_size
        "H"    # track_record_size
        "I"    # association_algorithm_id
        "I"    # association_algorithm_version
        "I"    # tracker_algorithm_id
        "I"    # tracker_algorithm_version
        "I"    # track_management_algorithm_id
        "I"    # track_management_algorithm_version
        "20s"  # association_params
        "20s"  # tracker_params
        "20s"  # track_management_params
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read body header. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    return TracksBodyHeader(
        body_magic=unpacked[0].decode("ascii"),
        body_header_version=unpacked[1],
        body_header_size=unpacked[2],
        frame_header_size=unpacked[3],
        track_record_size=unpacked[4],
        association_algorithm_id=unpacked[5],
        association_algorithm_version=unpacked[6],
        tracker_algorithm_id=unpacked[7],
        tracker_algorithm_version=unpacked[8],
        track_management_algorithm_id=unpacked[9],
        track_management_algorithm_version=unpacked[10],
        association_params=unpacked[11],
        tracker_params=unpacked[12],
        track_management_params=unpacked[13],
    )


def _read_frame_header(f: io.BufferedReader, ep: str) -> TracksFrameHeader:
    """Read Tracks frame header (30 bytes)."""
    fmt = f"{ep}4sHqHHHIIH"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read frame header.")
    
    (frame_magic_b, frame_header_size, timestamp_utc_ticks,
     num_tracks, num_new_tracks, num_terminated_tracks,
     sequence_number, payload_size_bytes, flags) = struct.unpack(fmt, raw)
    
    frame_magic = frame_magic_b.decode("ascii")
    if frame_magic != "FRME":
        raise ValueError(f"Invalid frame magic: expected 'FRME', got '{frame_magic}'")
    
    return TracksFrameHeader(
        frame_magic=frame_magic,
        frame_header_size=frame_header_size,
        timestamp_utc_ticks=timestamp_utc_ticks,
        num_tracks=num_tracks,
        num_new_tracks=num_new_tracks,
        num_terminated_tracks=num_terminated_tracks,
        sequence_number=sequence_number,
        payload_size_bytes=payload_size_bytes,
        flags=flags,
    )


def _create_track_dtype(frame_idx: int, seq_num: int, count: int) -> np.ndarray:
    """Create numpy array with track dtype."""
    dtype = np.dtype([
        # Identity & Status (28 bytes)
        ('track_id', np.uint32),
        ('sequence_number', np.uint32),
        ('status', np.uint8),
        ('associated_detection_count', np.uint8),
        ('frames_since_detection', np.uint16),
        ('target_class_id', np.uint16),
        ('track_lifetime_seconds', np.float32),
        ('birth_timestamp_utc_ticks', np.int64),
        ('gap_count', np.uint16),
        # Padding handled implicitly
        
        # Sensor Cartesian System (48 bytes)
        ('cart_x', np.float32),
        ('cart_x_std', np.float32),
        ('cart_y', np.float32),
        ('cart_y_std', np.float32),
        ('cart_z', np.float32),
        ('cart_z_std', np.float32),
        ('cart_vx', np.float32),
        ('cart_vx_std', np.float32),
        ('cart_vy', np.float32),
        ('cart_vy_std', np.float32),
        ('cart_vz', np.float32),
        ('cart_vz_std', np.float32),
        
        # ENU System (48 bytes)
        ('enu_east', np.float32),
        ('enu_east_std', np.float32),
        ('enu_north', np.float32),
        ('enu_north_std', np.float32),
        ('enu_up', np.float32),
        ('enu_up_std', np.float32),
        ('enu_ve', np.float32),
        ('enu_ve_std', np.float32),
        ('enu_vn', np.float32),
        ('enu_vn_std', np.float32),
        ('enu_vu', np.float32),
        ('enu_vu_std', np.float32),
        
        # Blob Information (20 bytes)
        ('blob_size_range', np.float32),
        ('blob_size_azimuth', np.float32),
        ('blob_size_elevation', np.float32),
        ('blob_size_doppler', np.float32),
        ('num_detections_in_blob', np.uint16),
        
        # Quality Metrics (12 bytes)
        ('amplitude_db', np.float32),
        ('snr_db', np.float32),
        ('confidence_score', np.float32),
        
        # Frame metadata
        ('frame_index', np.uint32),
    ])
    return np.zeros(count, dtype=dtype)


def _read_track_records(
    f: io.BufferedReader, 
    ep: str, 
    count: int,
    frame_index: int,
    sequence_number: int
) -> np.ndarray:
    """
    Read track records from file.
    
    Each record is 160 bytes total:
    - Identity & Status: 30 bytes (includes 2-byte padding)
    - Sensor Cartesian: 48 bytes (12 float32s)
    - ENU: 48 bytes (12 float32s)
    - Blob info: 20 bytes (includes 2-byte padding)
    - Quality: 12 bytes
    - Final padding: 2 bytes
    """
    # Format string for one track record
    # Identity & Status (30 bytes): IIBBHHfqH + 2 padding
    # Cartesian (48 bytes): 12 floats
    # ENU (48 bytes): 12 floats
    # Blob (20 bytes): 4 floats + 1 uint16 + 2 padding
    # Quality (12 bytes): 3 floats
    # Final record padding (2 bytes): 2x
    single_fmt = "IIBBHHfqH2x" + "f"*12 + "f"*12 + "ffffH2x" + "fff" + "2x"
    
    fmt = f"{ep}" + single_fmt * count
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read {count} track records. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    # Create structured array
    tracks = _create_track_dtype(frame_index, sequence_number, count)
    
    # Each track has 41 unpacked values (padding bytes are omitted by struct.unpack).
    values_per_track = 9 + 12 + 12 + 5 + 3
    
    for i in range(count):
        idx = i * values_per_track
        
        # Identity & Status
        tracks[i]['track_id'] = unpacked[idx + 0]
        tracks[i]['sequence_number'] = unpacked[idx + 1]
        tracks[i]['status'] = unpacked[idx + 2]
        tracks[i]['associated_detection_count'] = unpacked[idx + 3]
        tracks[i]['frames_since_detection'] = unpacked[idx + 4]
        tracks[i]['target_class_id'] = unpacked[idx + 5]
        tracks[i]['track_lifetime_seconds'] = unpacked[idx + 6]
        tracks[i]['birth_timestamp_utc_ticks'] = unpacked[idx + 7]
        tracks[i]['gap_count'] = unpacked[idx + 8]
        
        # Sensor Cartesian (12 values starting at idx+9)
        cart_start = idx + 9
        tracks[i]['cart_x'] = unpacked[cart_start + 0]
        tracks[i]['cart_x_std'] = unpacked[cart_start + 1]
        tracks[i]['cart_y'] = unpacked[cart_start + 2]
        tracks[i]['cart_y_std'] = unpacked[cart_start + 3]
        tracks[i]['cart_z'] = unpacked[cart_start + 4]
        tracks[i]['cart_z_std'] = unpacked[cart_start + 5]
        tracks[i]['cart_vx'] = unpacked[cart_start + 6]
        tracks[i]['cart_vx_std'] = unpacked[cart_start + 7]
        tracks[i]['cart_vy'] = unpacked[cart_start + 8]
        tracks[i]['cart_vy_std'] = unpacked[cart_start + 9]
        tracks[i]['cart_vz'] = unpacked[cart_start + 10]
        tracks[i]['cart_vz_std'] = unpacked[cart_start + 11]
        
        # ENU (12 values starting at idx+21)
        enu_start = idx + 21
        tracks[i]['enu_east'] = unpacked[enu_start + 0]
        tracks[i]['enu_east_std'] = unpacked[enu_start + 1]
        tracks[i]['enu_north'] = unpacked[enu_start + 2]
        tracks[i]['enu_north_std'] = unpacked[enu_start + 3]
        tracks[i]['enu_up'] = unpacked[enu_start + 4]
        tracks[i]['enu_up_std'] = unpacked[enu_start + 5]
        tracks[i]['enu_ve'] = unpacked[enu_start + 6]
        tracks[i]['enu_ve_std'] = unpacked[enu_start + 7]
        tracks[i]['enu_vn'] = unpacked[enu_start + 8]
        tracks[i]['enu_vn_std'] = unpacked[enu_start + 9]
        tracks[i]['enu_vu'] = unpacked[enu_start + 10]
        tracks[i]['enu_vu_std'] = unpacked[enu_start + 11]
        
        # Blob info (5 values starting at idx+33)
        blob_start = idx + 33
        tracks[i]['blob_size_range'] = unpacked[blob_start + 0]
        tracks[i]['blob_size_azimuth'] = unpacked[blob_start + 1]
        tracks[i]['blob_size_elevation'] = unpacked[blob_start + 2]
        tracks[i]['blob_size_doppler'] = unpacked[blob_start + 3]
        tracks[i]['num_detections_in_blob'] = unpacked[blob_start + 4]
        
        # Quality metrics (3 values starting at idx+38)
        quality_start = idx + 38
        tracks[i]['amplitude_db'] = unpacked[quality_start + 0]
        tracks[i]['snr_db'] = unpacked[quality_start + 1]
        tracks[i]['confidence_score'] = unpacked[quality_start + 2]
        
        # Frame metadata
        tracks[i]['frame_index'] = frame_index
    
    return tracks


def _print_header_summary(FH: FileHeader, BH: TracksBodyHeader) -> None:
    """Print summary of file and body headers."""
    print("--- Dopplium Tracks Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  "
          f"Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  "
          f"FrameHdr={BH.frame_header_size}  TotalFramesWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    
    print("\n-- Tracking Configuration --")
    print(f"Track record size: {BH.track_record_size} bytes")
    print(
        f"Association algorithm: id={BH.association_algorithm_id}, "
        f"version={BH.association_algorithm_version}"
    )
    print(
        f"Tracker algorithm: id={BH.tracker_algorithm_id}, "
        f"version={BH.tracker_algorithm_version}"
    )
    print(
        f"Track management algorithm: id={BH.track_management_algorithm_id}, "
        f"version={BH.track_management_algorithm_version}"
    )
    print(f"Association params length: {len(BH.association_params)} bytes")
    print(f"Tracker params length: {len(BH.tracker_params)} bytes")
    print(f"Track management params length: {len(BH.track_management_params)} bytes")
    print(f"Body header version: {BH.body_header_version}")


# ==============================
# Convenience functions
# ==============================

def filter_tracks_by_status(
    tracks: np.ndarray, 
    status: int
) -> np.ndarray:
    """
    Filter tracks by status.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    status : int
        Track status (0=tentative, 1=confirmed, 2=coasting, 3=terminated)
    
    Returns:
    --------
    np.ndarray : Filtered tracks
    """
    mask = tracks['status'] == status
    return tracks[mask]


def filter_tracks_by_class(
    tracks: np.ndarray, 
    target_class_id: int
) -> np.ndarray:
    """
    Filter tracks by target class.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    target_class_id : int
        Target class identifier
    
    Returns:
    --------
    np.ndarray : Filtered tracks
    """
    mask = tracks['target_class_id'] == target_class_id
    return tracks[mask]


def filter_tracks_by_id(
    tracks: np.ndarray, 
    track_id: int
) -> np.ndarray:
    """
    Filter tracks by track ID to get trajectory of a single track.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    track_id : int
        Unique track identifier
    
    Returns:
    --------
    np.ndarray : All records for the specified track
    """
    mask = tracks['track_id'] == track_id
    return tracks[mask]


def filter_tracks_by_lifetime(
    tracks: np.ndarray, 
    min_lifetime: float, 
    max_lifetime: float = np.inf
) -> np.ndarray:
    """
    Filter tracks by lifetime duration.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    min_lifetime : float
        Minimum lifetime in seconds (inclusive)
    max_lifetime : float
        Maximum lifetime in seconds (inclusive)
    
    Returns:
    --------
    np.ndarray : Filtered tracks
    """
    mask = (tracks['track_lifetime_seconds'] >= min_lifetime) & \
           (tracks['track_lifetime_seconds'] <= max_lifetime)
    return tracks[mask]


def get_valid_coordinates(
    tracks: np.ndarray, 
    coord_system: str = 'cartesian'
) -> np.ndarray:
    """
    Get tracks with valid (non -1.0) coordinates in specified system.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    coord_system : str
        'cartesian' or 'enu'
    
    Returns:
    --------
    np.ndarray : Tracks with valid coordinates
    """
    if coord_system.lower() == 'cartesian':
        mask = (tracks['cart_x'] != -1.0) & \
               (tracks['cart_y'] != -1.0) & \
               (tracks['cart_z'] != -1.0)
    elif coord_system.lower() == 'enu':
        mask = (tracks['enu_east'] != -1.0) & \
               (tracks['enu_north'] != -1.0) & \
               (tracks['enu_up'] != -1.0)
    else:
        raise ValueError(f"Unknown coordinate system: {coord_system}")
    
    return tracks[mask]


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert sensor Cartesian coordinates to spherical (RAE).
    
    Coordinate convention (left-handed):
    - x-axis: forward (0° azimuth, 0° elevation)
    - y-axis: right (90° azimuth)
    - z-axis: up (90° elevation)
    
    Parameters:
    -----------
    x, y, z : float
        Cartesian coordinates in meters
    
    Returns:
    --------
    tuple : (range, azimuth, elevation)
        range : float (meters)
        azimuth : float (degrees)
        elevation : float (degrees)
    """
    range_m = np.sqrt(x**2 + y**2 + z**2)
    azimuth_deg = np.arctan2(y, x) * 180.0 / np.pi
    elevation_deg = np.arcsin(z / range_m) * 180.0 / np.pi if range_m > 0 else 0.0
    
    return range_m, azimuth_deg, elevation_deg


def spherical_to_cartesian(range_m: float, azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """
    Convert spherical (RAE) coordinates to sensor Cartesian.
    
    Coordinate convention (left-handed):
    - x-axis: forward (0° azimuth, 0° elevation)
    - y-axis: right (90° azimuth)
    - z-axis: up (90° elevation)
    
    Parameters:
    -----------
    range_m : float
        Range in meters
    azimuth_deg : float
        Azimuth angle in degrees
    elevation_deg : float
        Elevation angle in degrees
    
    Returns:
    --------
    tuple : (x, y, z)
        Cartesian coordinates in meters
    """
    az_rad = azimuth_deg * np.pi / 180.0
    el_rad = elevation_deg * np.pi / 180.0
    
    x = range_m * np.cos(el_rad) * np.cos(az_rad)
    y = range_m * np.cos(el_rad) * np.sin(az_rad)
    z = range_m * np.sin(el_rad)
    
    return x, y, z


def get_track_statistics(tracks: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about tracks.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    
    Returns:
    --------
    dict : Statistics including min, max, mean, std for key fields
    """
    if len(tracks) == 0:
        return {"count": 0}
    
    # Get valid coordinate masks
    valid_cart = (tracks['cart_x'] != -1.0)
    valid_enu = (tracks['enu_east'] != -1.0)
    
    stats = {
        "count": len(tracks),
        "unique_tracks": len(np.unique(tracks['track_id'])),
        "unique_frames": len(np.unique(tracks['frame_index'])),
        "status_distribution": {
            "tentative": int(np.sum(tracks['status'] == 0)),
            "confirmed": int(np.sum(tracks['status'] == 1)),
            "coasting": int(np.sum(tracks['status'] == 2)),
            "terminated": int(np.sum(tracks['status'] == 3)),
        },
        "valid_coordinates": {
            "cartesian": int(np.sum(valid_cart)),
            "enu": int(np.sum(valid_enu)),
        },
        "lifetime": {
            "min": float(np.min(tracks['track_lifetime_seconds'])),
            "max": float(np.max(tracks['track_lifetime_seconds'])),
            "mean": float(np.mean(tracks['track_lifetime_seconds'])),
            "std": float(np.std(tracks['track_lifetime_seconds'])),
        },
        "gap_count": {
            "min": int(np.min(tracks['gap_count'])),
            "max": int(np.max(tracks['gap_count'])),
            "mean": float(np.mean(tracks['gap_count'])),
        },
    }
    
    # Add Cartesian stats if any valid
    if np.sum(valid_cart) > 0:
        cart_tracks = tracks[valid_cart]
        stats["cartesian"] = {
            "x": {"min": float(np.min(cart_tracks['cart_x'])), 
                  "max": float(np.max(cart_tracks['cart_x'])),
                  "mean": float(np.mean(cart_tracks['cart_x']))},
            "y": {"min": float(np.min(cart_tracks['cart_y'])),
                  "max": float(np.max(cart_tracks['cart_y'])),
                  "mean": float(np.mean(cart_tracks['cart_y']))},
            "z": {"min": float(np.min(cart_tracks['cart_z'])),
                  "max": float(np.max(cart_tracks['cart_z'])),
                  "mean": float(np.mean(cart_tracks['cart_z']))},
        }
    
    return stats


def get_blob_statistics(tracks: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about blob information in tracks.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    
    Returns:
    --------
    dict : Blob-related statistics
    """
    if len(tracks) == 0:
        return {"count": 0}
    
    return {
        "count": len(tracks),
        "detections_per_blob": {
            "min": int(np.min(tracks['num_detections_in_blob'])),
            "max": int(np.max(tracks['num_detections_in_blob'])),
            "mean": float(np.mean(tracks['num_detections_in_blob'])),
        },
        "blob_size_range": {
            "min": float(np.min(tracks['blob_size_range'])),
            "max": float(np.max(tracks['blob_size_range'])),
            "mean": float(np.mean(tracks['blob_size_range'])),
        },
        "blob_size_azimuth": {
            "min": float(np.min(tracks['blob_size_azimuth'])),
            "max": float(np.max(tracks['blob_size_azimuth'])),
            "mean": float(np.mean(tracks['blob_size_azimuth'])),
        },
        "blob_size_elevation": {
            "min": float(np.min(tracks['blob_size_elevation'])),
            "max": float(np.max(tracks['blob_size_elevation'])),
            "mean": float(np.mean(tracks['blob_size_elevation'])),
        },
        "blob_size_doppler": {
            "min": float(np.min(tracks['blob_size_doppler'])),
            "max": float(np.max(tracks['blob_size_doppler'])),
            "mean": float(np.mean(tracks['blob_size_doppler'])),
        },
    }


def get_track_lifecycle_stats(tracks: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about track lifecycle (lifetime, birth, gaps).
    
    Parameters:
    -----------
    tracks : np.ndarray
        Structured array from parse_dopplium_tracks
    
    Returns:
    --------
    dict : Lifecycle statistics
    """
    if len(tracks) == 0:
        return {"count": 0}
    
    unique_track_ids = np.unique(tracks['track_id'])
    
    # For each unique track, get its max lifetime (most recent record)
    lifetimes = []
    gaps = []
    for tid in unique_track_ids:
        track_data = tracks[tracks['track_id'] == tid]
        # Get the last record (highest lifetime)
        last_record = track_data[np.argmax(track_data['track_lifetime_seconds'])]
        lifetimes.append(last_record['track_lifetime_seconds'])
        gaps.append(last_record['gap_count'])
    
    return {
        "unique_tracks": len(unique_track_ids),
        "total_records": len(tracks),
        "track_lifetime": {
            "min": float(np.min(lifetimes)),
            "max": float(np.max(lifetimes)),
            "mean": float(np.mean(lifetimes)),
            "std": float(np.std(lifetimes)),
        },
        "gaps_per_track": {
            "min": int(np.min(gaps)),
            "max": int(np.max(gaps)),
            "mean": float(np.mean(gaps)),
        },
        "birth_timestamps": {
            "earliest": int(np.min(tracks['birth_timestamp_utc_ticks'])),
            "latest": int(np.max(tracks['birth_timestamp_utc_ticks'])),
        }
    }

