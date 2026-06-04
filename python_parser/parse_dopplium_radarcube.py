"""
Parser for Dopplium RadarCube (Range-Doppler-Azimuth-Elevation) binary format.

Supports:
- Version 3, message_type 3: RadarCube data

Reads processed radar data files written by RadarCubeBinaryWriter.
Returns numpy arrays shaped [range, doppler, azimuth, elevation, cpis] and all headers.

Format Notes:
- Data stored in Fortran-order (column-major) with range varying fastest
- Config version: 1 (first iteration of RadarCube format)
- Body header: 256 bytes total
- Includes angular processing algorithm information (FFT, CAPON, MUSIC)
"""

from __future__ import annotations
import io
import struct
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

from .parse_dopplium_header import FileHeader, parse_file_header


# ==============================
# Dataclasses for headers
# ==============================


@dataclass
class RadarCubeBodyHeader:
    """RadarCube body header (256 bytes) containing configuration and processing parameters."""
    config_magic: str
    config_version: int
    body_header_size: int
    cpi_header_size: int
    reserved1: int
    # Dimensions
    n_range_bins: int
    n_doppler_bins: int
    n_azimuth_bins: int
    n_elevation_bins: int
    # Range parameters
    range_min_m: float
    range_max_m: float
    range_resolution_m: float  # FFT bin spacing
    # Velocity parameters
    velocity_min_mps: float
    velocity_max_mps: float
    velocity_resolution_mps: float  # FFT bin spacing
    physical_velocity_resolution_mps: float  # Integration time-determined
    # Angular parameters
    # Note: Default values may be used when antenna configuration is stored elsewhere
    # Defaults: azimuth [-180, 180], elevation [-90, 90], resolution 0.0 if unknown
    azimuth_min_deg: float
    azimuth_max_deg: float
    azimuth_resolution_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    elevation_resolution_deg: float
    # Data type
    data_type: int  # 0=complex64, 1=complex128, 2=float32, 3=float64, 4=int16, 5=int32
    # Processing parameters
    # Note: nfft_azimuth and nfft_elevation may be 0 for non-FFT algorithms (CAPON, MUSIC, etc.)
    nfft_range: int
    nfft_doppler: int
    nfft_azimuth: int  # 0 = not applicable/non-FFT algorithm
    nfft_elevation: int  # 0 = not applicable/non-FFT algorithm
    range_window_type: int  # 0=none, 1=hann, 2=hamming, 3=blackman, 4=kaiser
    doppler_window_type: int
    azimuth_window_type: int
    elevation_window_type: int
    fftshift_range: int  # 0=no, 1=yes
    fftshift_doppler: int
    fftshift_azimuth: int
    fftshift_elevation: int
    range_half_spectrum: int
    coherent_integration_time_ms: float
    # Angular algorithm
    angle_estimation_algorithm: int  # 0=FFT, 1=CAPON, 2=MUSIC, 3=other
    # Additional resolution and scale parameters
    physical_range_resolution_m: float  # Bandwidth-determined
    is_db_scale: int  # 0=linear, 1=dB
    data_format: int  # 0=complex, 1=amplitude, 2=power
    cpis_incoherently_integrated: int  # Number of CPIs summed incoherently
    _reserved3: bytes


@dataclass
class CPIHeader:
    """RadarCube CPI header (22 bytes)."""
    cpi_magic: str
    cpi_header_size: int
    cpi_timestamp_utc_ticks: int
    cpi_number: int
    cpi_payload_size: int


# ==============================
# Public API
# ==============================

def parse_dopplium_radarcube(
    filename: str,
    *,
    max_cpis: Optional[int] = None,
    start_frame: Optional[int] = None,
    verbose: bool = True,
    _file_header: Optional[FileHeader] = None,
    _endian_prefix: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium RadarCube file into a numpy array shaped [range, doppler, azimuth, elevation, cpis].
    
    Parameters:
    -----------
    filename : str
        Path to the RadarCube binary file
    max_cpis : int, optional
        Maximum number of CPIs to read (None = all CPIs)
    start_frame : int, optional
        Zero-based CPI index to start reading from
    verbose : bool
        Print parsing information
    _file_header : FileHeader, optional
        Pre-parsed file header (internal use)
    _endian_prefix : str, optional
        Endianness prefix (internal use)
    
    Returns:
    --------
    tuple : (data, headers)
        data : np.ndarray
            Processed radar cube data [range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]
        headers : dict
            Dictionary containing 'file', 'body', and 'cpi' headers
    """
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
        
        if FH.message_type != 3:
            raise ValueError(f"This file is not RadarCube (message_type={FH.message_type}, expected 3).")
        
        # Position at body header if we parsed the file header ourselves
        if _file_header is None:
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        BH = _read_radarcube_body_header(f, endian_prefix)
        
        if verbose:
            _print_header_summary(FH, BH)
        
        # Extract dimensions
        n_range = int(BH.n_range_bins)
        n_doppler = int(BH.n_doppler_bins)
        n_azimuth = int(BH.n_azimuth_bins)
        n_elevation = int(BH.n_elevation_bins)
        
        # Map data type code to numpy dtype
        dtype = _map_data_type(BH.data_type)
        
        # Determine number of CPIs from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        cpi_unit = FH.frame_header_size + (n_range * n_doppler * n_azimuth * n_elevation * dtype.itemsize)
        
        # Calculate expected payload size per CPI
        expected_payload_size = n_range * n_doppler * n_azimuth * n_elevation * dtype.itemsize
        
        if verbose:
            print(f"\nExpected payload size per CPI: {expected_payload_size} bytes")
            print(f"Bytes after headers: {bytes_after_headers}")
        
        # Estimate number of CPIs
        n_cpis_estimate = max(0, bytes_after_headers // cpi_unit)
        
        if verbose:
            print(f"Estimated CPIs in file: {n_cpis_estimate}")
        
        start_frame = _coerce_optional_nonnegative_int(start_frame, "start_frame") or 0
        max_cpis = _coerce_optional_nonnegative_int(max_cpis, "max_cpis")
        n_cpis_available = max(0, n_cpis_estimate - start_frame)
        n_cpis = n_cpis_available if max_cpis is None else min(n_cpis_available, max_cpis)
        
        # Allocate output array
        data = np.zeros((n_range, n_doppler, n_azimuth, n_elevation, n_cpis), dtype=dtype)
        
        # Read CPIs
        f.seek(FH.file_header_size + BH.body_header_size + start_frame * cpi_unit, io.SEEK_SET)
        cpi_headers = []
        
        cpis_read = 0
        while cpis_read < n_cpis:
            cpi_index = start_frame + cpis_read
            try:
                # Check if we have enough bytes left
                current_pos = f.tell()
                if current_pos >= file_size:
                    break
                
                CH = _read_cpi_header(f, endian_prefix)
                cpi_headers.append(CH)
                
                if verbose and (cpis_read == 0 or (cpis_read + 1) % 10 == 0):
                    print(f"  Reading CPI {cpis_read + 1}/{n_cpis}: "
                          f"number={CH.cpi_number}, size={CH.cpi_payload_size} bytes")
                
                # Validate CPI payload size
                if CH.cpi_payload_size != expected_payload_size:
                    if verbose:
                        print(f"Warning: CPI {cpi_index} payload size mismatch: "
                              f"expected={expected_payload_size}, got={CH.cpi_payload_size}")
                
                # Read payload
                payload_bytes = f.read(CH.cpi_payload_size)
                if len(payload_bytes) != CH.cpi_payload_size:
                    raise EOFError(f"Unexpected EOF while reading CPI {cpi_index} payload.")
                
                # Reshape payload to [range, doppler, azimuth, elevation]
                # Note: Data is stored in Fortran-order (column-major) with range varying fastest
                cpi_data = np.frombuffer(payload_bytes, dtype=dtype)
                
                # Reshape from flat array to 4D using Fortran-order
                try:
                    cpi_data = cpi_data.reshape((n_range, n_doppler, n_azimuth, n_elevation), order='F')
                    data[:, :, :, :, cpis_read] = cpi_data
                except ValueError as e:
                    raise ValueError(f"CPI {cpi_index}: Cannot reshape payload. "
                                   f"Expected {n_range}x{n_doppler}x{n_azimuth}x{n_elevation} = "
                                   f"{n_range * n_doppler * n_azimuth * n_elevation} elements, "
                                   f"got {cpi_data.size}. Error: {e}")
                
                cpis_read += 1
                
            except EOFError:
                if verbose:
                    print(f"Reached end of file after reading {cpis_read} CPIs.")
                break
        
        # Trim data array if we read fewer CPIs than estimated
        if cpis_read < n_cpis:
            data = data[:, :, :, :, :cpis_read]
        
        headers = {
            "file": FH,
            "body": BH,
            "cpi": cpi_headers,
        }
        
        if verbose:
            print(f"\nParsed data shape: {tuple(data.shape)}  "
                  f"[range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]")
            print(f"Total CPIs read: {cpis_read}")
        
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


def _coerce_optional_nonnegative_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be >= 0")
    return coerced


def _read_radarcube_body_header(f: io.BufferedReader, ep: str) -> RadarCubeBodyHeader:
    """Read RadarCube body header (256 bytes)."""
    fmt = (
        f"{ep}"
        "4s"   # config_magic (RCUB)
        "H"    # config_version
        "H"    # body_header_size
        "H"    # cpi_header_size
        "H"    # reserved1
        "I"    # n_range_bins
        "I"    # n_doppler_bins
        "I"    # n_azimuth_bins
        "I"    # n_elevation_bins
        "f"    # range_min_m
        "f"    # range_max_m
        "f"    # range_resolution_m
        "f"    # velocity_min_mps
        "f"    # velocity_max_mps
        "f"    # velocity_resolution_mps
        "f"    # physical_velocity_resolution_mps
        "f"    # azimuth_min_deg
        "f"    # azimuth_max_deg
        "f"    # azimuth_resolution_deg
        "f"    # elevation_min_deg
        "f"    # elevation_max_deg
        "f"    # elevation_resolution_deg
        "B"    # data_type
        "I"    # nfft_range
        "I"    # nfft_doppler
        "I"    # nfft_azimuth
        "I"    # nfft_elevation
        "B"    # range_window_type
        "B"    # doppler_window_type
        "B"    # azimuth_window_type
        "B"    # elevation_window_type
        "B"    # fftshift_range
        "B"    # fftshift_doppler
        "B"    # fftshift_azimuth
        "B"    # fftshift_elevation
        "B"    # range_half_spectrum
        "f"    # coherent_integration_time_ms
        "B"    # angle_estimation_algorithm
        "f"    # physical_range_resolution_m
        "B"    # is_db_scale
        "B"    # data_format
        "H"    # cpis_incoherently_integrated
        "137s" # reserved3
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read body header. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    return RadarCubeBodyHeader(
        config_magic=unpacked[0].decode("ascii"),
        config_version=unpacked[1],
        body_header_size=unpacked[2],
        cpi_header_size=unpacked[3],
        reserved1=unpacked[4],
        n_range_bins=unpacked[5],
        n_doppler_bins=unpacked[6],
        n_azimuth_bins=unpacked[7],
        n_elevation_bins=unpacked[8],
        range_min_m=unpacked[9],
        range_max_m=unpacked[10],
        range_resolution_m=unpacked[11],
        velocity_min_mps=unpacked[12],
        velocity_max_mps=unpacked[13],
        velocity_resolution_mps=unpacked[14],
        physical_velocity_resolution_mps=unpacked[15],
        azimuth_min_deg=unpacked[16],
        azimuth_max_deg=unpacked[17],
        azimuth_resolution_deg=unpacked[18],
        elevation_min_deg=unpacked[19],
        elevation_max_deg=unpacked[20],
        elevation_resolution_deg=unpacked[21],
        data_type=unpacked[22],
        nfft_range=unpacked[23],
        nfft_doppler=unpacked[24],
        nfft_azimuth=unpacked[25],
        nfft_elevation=unpacked[26],
        range_window_type=unpacked[27],
        doppler_window_type=unpacked[28],
        azimuth_window_type=unpacked[29],
        elevation_window_type=unpacked[30],
        fftshift_range=unpacked[31],
        fftshift_doppler=unpacked[32],
        fftshift_azimuth=unpacked[33],
        fftshift_elevation=unpacked[34],
        range_half_spectrum=unpacked[35],
        coherent_integration_time_ms=unpacked[36],
        angle_estimation_algorithm=unpacked[37],
        physical_range_resolution_m=unpacked[38],
        is_db_scale=unpacked[39],
        data_format=unpacked[40],
        cpis_incoherently_integrated=unpacked[41],
        _reserved3=unpacked[42],
    )


def _read_cpi_header(f: io.BufferedReader, ep: str) -> CPIHeader:
    """Read RadarCube CPI header (22 bytes)."""
    fmt = f"{ep}4sHqII"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read CPI header.")
    
    (cpi_magic_b, cpi_header_size, cpi_timestamp_utc_ticks,
     cpi_number, cpi_payload_size) = struct.unpack(fmt, raw)
    
    cpi_magic = cpi_magic_b.decode("ascii")
    if cpi_magic != "CPII":
        raise ValueError(f"Invalid CPI magic: expected 'CPII', got '{cpi_magic}'")
    
    return CPIHeader(
        cpi_magic=cpi_magic,
        cpi_header_size=cpi_header_size,
        cpi_timestamp_utc_ticks=cpi_timestamp_utc_ticks,
        cpi_number=cpi_number,
        cpi_payload_size=cpi_payload_size,
    )


def _map_data_type(data_type_code: int) -> np.dtype:
    """Map data type code to numpy dtype."""
    type_map = {
        0: np.complex64,
        1: np.complex128,
        2: np.float32,
        3: np.float64,
        4: np.int16,
        5: np.int32
    }
    return np.dtype(type_map.get(data_type_code, np.complex64))


def _map_window_type(window_code: int) -> str:
    """Map window type code to string."""
    window_map = {
        0: 'none',
        1: 'hann',
        2: 'hamming',
        3: 'blackman',
        4: 'kaiser'
    }
    return window_map.get(window_code, 'unknown')


def _map_data_format(format_code: int) -> str:
    """Map data format code to string."""
    format_map = {
        0: 'complex',
        1: 'amplitude',
        2: 'power'
    }
    return format_map.get(format_code, 'unknown')


def _map_angle_algorithm(algorithm_code: int) -> str:
    """Map angle estimation algorithm code to string."""
    algorithm_map = {
        0: 'FFT',
        1: 'CAPON',
        2: 'MUSIC',
        3: 'other'
    }
    return algorithm_map.get(algorithm_code, 'unknown')


def _print_header_summary(FH: FileHeader, BH: RadarCubeBodyHeader) -> None:
    """Print summary of file and body headers."""
    print("--- Dopplium RadarCube Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  "
          f"Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  "
          f"CPIHdr={BH.cpi_header_size}  TotalCPIsWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    
    print("\n-- RadarCube Configuration --")
    dtype_str = ['complex64', 'complex128', 'float32', 'float64', 'int16', 'int32'][BH.data_type] \
                if BH.data_type < 6 else 'unknown'
    print(f"Dimensions: Range={BH.n_range_bins}, Doppler={BH.n_doppler_bins}, "
          f"Azimuth={BH.n_azimuth_bins}, Elevation={BH.n_elevation_bins}")
    print(f"Data type: {dtype_str}")
    print(f"Data format: {_map_data_format(BH.data_format)}")
    print(f"Scale: {'dB' if BH.is_db_scale else 'linear'}")
    print(f"Incoherent CPI integration: {BH.cpis_incoherently_integrated}")
    print(f"Storage order: Fortran (column-major, range varies fastest)")
    
    print("\n-- Range/Velocity Axes --")
    print(f"Range: {BH.range_min_m:.2f} to {BH.range_max_m:.2f} m")
    print(f"  FFT bin spacing: {BH.range_resolution_m:.4f} m")
    print(f"  Physical resolution: {BH.physical_range_resolution_m:.4f} m")
    print(f"Velocity: {BH.velocity_min_mps:.2f} to {BH.velocity_max_mps:.2f} m/s")
    print(f"  FFT bin spacing: {BH.velocity_resolution_mps:.4f} m/s")
    print(f"  Physical resolution: {BH.physical_velocity_resolution_mps:.4f} m/s")
    
    print("\n-- Angular Axes --")
    print(f"Azimuth: {BH.azimuth_min_deg:.2f} to {BH.azimuth_max_deg:.2f} deg")
    if BH.azimuth_resolution_deg > 0:
        print(f"  Resolution: {BH.azimuth_resolution_deg:.4f} deg")
    else:
        print(f"  Resolution: unknown/not specified")
    print(f"Elevation: {BH.elevation_min_deg:.2f} to {BH.elevation_max_deg:.2f} deg")
    if BH.elevation_resolution_deg > 0:
        print(f"  Resolution: {BH.elevation_resolution_deg:.4f} deg")
    else:
        print(f"  Resolution: unknown/not specified")
    print(f"Angle estimation algorithm: {_map_angle_algorithm(BH.angle_estimation_algorithm)}")
    
    print("\n-- Processing Parameters --")
    # Show FFT sizes with indication if not applicable
    azimuth_fft_str = str(BH.nfft_azimuth) if BH.nfft_azimuth > 0 else "N/A"
    elevation_fft_str = str(BH.nfft_elevation) if BH.nfft_elevation > 0 else "N/A"
    print(f"FFT sizes: Range={BH.nfft_range}, Doppler={BH.nfft_doppler}, "
          f"Azimuth={azimuth_fft_str}, Elevation={elevation_fft_str}")
    
    # Only show window types and shifts if FFT was used
    if BH.nfft_azimuth > 0 or BH.nfft_elevation > 0:
        print(f"Window types: Range={_map_window_type(BH.range_window_type)}, "
              f"Doppler={_map_window_type(BH.doppler_window_type)}, "
              f"Azimuth={_map_window_type(BH.azimuth_window_type)}, "
              f"Elevation={_map_window_type(BH.elevation_window_type)}")
        print(f"FFT shifts: Range={'Yes' if BH.fftshift_range else 'No'}, "
              f"Doppler={'Yes' if BH.fftshift_doppler else 'No'}, "
              f"Azimuth={'Yes' if BH.fftshift_azimuth else 'No'}, "
              f"Elevation={'Yes' if BH.fftshift_elevation else 'No'}")
    else:
        print(f"Window types: Range={_map_window_type(BH.range_window_type)}, "
              f"Doppler={_map_window_type(BH.doppler_window_type)}")
        print(f"FFT shifts: Range={'Yes' if BH.fftshift_range else 'No'}, "
              f"Doppler={'Yes' if BH.fftshift_doppler else 'No'}")
    print(f"Half spectrum: {'Yes' if BH.range_half_spectrum else 'No'}")
    print(f"Integration time: {BH.coherent_integration_time_ms:.3f} ms")


# ==============================
# Convenience functions
# ==============================

def get_range_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract range axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    np.ndarray : Range axis in meters
    """
    BH = headers['body']
    return np.linspace(BH.range_min_m, BH.range_max_m, BH.n_range_bins)


def get_velocity_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract velocity axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    np.ndarray : Velocity axis in m/s
    """
    BH = headers['body']
    return np.linspace(BH.velocity_min_mps, BH.velocity_max_mps, BH.n_doppler_bins)


def _build_linear_axis(axis_min: float, axis_resolution: float, n_bins: int) -> np.ndarray:
    """Build a linear axis using min + index * resolution."""
    if n_bins <= 0:
        return np.zeros((0,), dtype=np.float64)
    return float(axis_min) + np.arange(n_bins, dtype=np.float64) * float(axis_resolution)


def _build_angle_axis(
    angle_min_deg: float,
    angle_max_deg: float,
    angle_resolution_deg: float,
    n_bins: int,
    *,
    use_fft_mapping: bool,
    fftshift_enabled: bool,
) -> np.ndarray:
    """
    Build angle axis for bin-to-angle conversion.

    For FFT angle estimation, bins are interpreted in direction-cosine space
    (nonlinear in degrees). For full-span metadata [-90,+90], uses ULA half-lambda
    FFT-bin mapping: sin(theta_k) = 2k/N.
    """
    if n_bins <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n_bins == 1:
        return np.array([float(angle_min_deg)], dtype=np.float64)

    angle_min = float(angle_min_deg)
    angle_max = float(angle_max_deg)

    if use_fft_mapping and (-90.0 <= angle_min < angle_max <= 90.0):
        if np.isclose(angle_min, -90.0, atol=1e-3) and np.isclose(angle_max, 90.0, atol=1e-3):
            k = np.fft.fftfreq(n_bins) * float(n_bins)
            if fftshift_enabled:
                k = np.fft.fftshift(k)
            sin_theta = np.clip((2.0 * k) / float(n_bins), -1.0, 1.0)
            return np.rad2deg(np.arcsin(sin_theta)).astype(np.float64)

        sin_min = np.sin(np.deg2rad(angle_min))
        sin_max = np.sin(np.deg2rad(angle_max))
        direction_cosine_axis = np.linspace(sin_min, sin_max, n_bins, dtype=np.float64)
        angle_axis = np.rad2deg(np.arcsin(np.clip(direction_cosine_axis, -1.0, 1.0)))
        if not fftshift_enabled:
            angle_axis = np.fft.ifftshift(angle_axis)
        return angle_axis

    if float(angle_resolution_deg) != 0.0:
        return _build_linear_axis(angle_min, angle_resolution_deg, n_bins)
    if angle_max > angle_min:
        return np.linspace(angle_min, angle_max, n_bins, dtype=np.float64)
    return np.full((n_bins,), angle_min, dtype=np.float64)


def get_azimuth_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract azimuth axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    np.ndarray : Azimuth axis in degrees
    
    Note:
    -----
    For FFT angle estimation (angle_estimation_algorithm=FFT with nfft_azimuth>0),
    returns nonlinear angle spacing. For full-span metadata [-90,+90], uses ULA
    half-lambda FFT-bin mapping (sin(theta_k)=2k/N). For non-FFT algorithms, uses
    linear metadata mapping.
    """
    BH = headers['body']
    angle_algo = int(getattr(BH, 'angle_estimation_algorithm', -1))
    return _build_angle_axis(
        BH.azimuth_min_deg,
        BH.azimuth_max_deg,
        BH.azimuth_resolution_deg,
        BH.n_azimuth_bins,
        use_fft_mapping=(angle_algo == 0 and int(getattr(BH, 'nfft_azimuth', 0)) > 0),
        fftshift_enabled=bool(getattr(BH, 'fftshift_azimuth', True)),
    )


def get_elevation_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract elevation axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    np.ndarray : Elevation axis in degrees
    
    Note:
    -----
    For FFT angle estimation (angle_estimation_algorithm=FFT with nfft_elevation>0),
    returns nonlinear angle spacing. For non-FFT algorithms, uses linear metadata mapping.
    """
    BH = headers['body']
    angle_algo = int(getattr(BH, 'angle_estimation_algorithm', -1))
    return _build_angle_axis(
        BH.elevation_min_deg,
        BH.elevation_max_deg,
        BH.elevation_resolution_deg,
        BH.n_elevation_bins,
        use_fft_mapping=(angle_algo == 0 and int(getattr(BH, 'nfft_elevation', 0)) > 0),
        fftshift_enabled=bool(getattr(BH, 'fftshift_elevation', True)),
    )


def has_known_angles(headers: Dict[str, Any]) -> bool:
    """
    Check if angular ranges are from antenna configuration or are default/unknown values.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    bool : True if angles appear to be from actual antenna config (non-default values)
    """
    BH = headers['body']
    # Check if values differ from defaults [-180, 180] for azimuth and [-90, 90] for elevation
    is_default = (abs(BH.azimuth_min_deg - (-180.0)) < 0.01 and 
                  abs(BH.azimuth_max_deg - 180.0) < 0.01 and
                  abs(BH.elevation_min_deg - (-90.0)) < 0.01 and
                  abs(BH.elevation_max_deg - 90.0) < 0.01 and
                  abs(BH.azimuth_resolution_deg) < 0.001 and
                  abs(BH.elevation_resolution_deg) < 0.001)
    return not is_default


def uses_fft_for_angles(headers: Dict[str, Any]) -> bool:
    """
    Check if FFT was used for angle estimation (vs. CAPON, MUSIC, etc.).
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    bool : True if FFT algorithm was used (nfft values > 0)
    """
    BH = headers['body']
    return BH.nfft_azimuth > 0 or BH.nfft_elevation > 0


def get_processing_info(headers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract processing information from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_radarcube
    
    Returns:
    --------
    dict : Processing parameters
    """
    BH = headers['body']
    
    return {
        'nfft_range': BH.nfft_range,
        'nfft_doppler': BH.nfft_doppler,
        'nfft_azimuth': BH.nfft_azimuth,
        'nfft_elevation': BH.nfft_elevation,
        'range_window': _map_window_type(BH.range_window_type),
        'doppler_window': _map_window_type(BH.doppler_window_type),
        'azimuth_window': _map_window_type(BH.azimuth_window_type),
        'elevation_window': _map_window_type(BH.elevation_window_type),
        'fftshift_range': bool(BH.fftshift_range),
        'fftshift_doppler': bool(BH.fftshift_doppler),
        'fftshift_azimuth': bool(BH.fftshift_azimuth),
        'fftshift_elevation': bool(BH.fftshift_elevation),
        'range_half_spectrum': bool(BH.range_half_spectrum),
        'coherent_integration_time_ms': BH.coherent_integration_time_ms,
        'physical_velocity_resolution_mps': BH.physical_velocity_resolution_mps,
        'physical_range_resolution_m': BH.physical_range_resolution_m,
        'is_db_scale': bool(BH.is_db_scale),
        'data_format': _map_data_format(BH.data_format),
        'cpis_incoherently_integrated': BH.cpis_incoherently_integrated,
        'angle_estimation_algorithm': _map_angle_algorithm(BH.angle_estimation_algorithm),
        'uses_fft_for_angles': uses_fft_for_angles(headers),
        'has_known_angles': has_known_angles(headers),
    }

