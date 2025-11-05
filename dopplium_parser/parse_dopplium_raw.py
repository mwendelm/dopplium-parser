from __future__ import annotations
import io
import math
import struct
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np


# ==============================
# Dataclasses for headers
# ==============================

@dataclass
class FileHeader:
    magic: str
    version: int
    endianness: int
    compression: int
    product_id: int
    message_type: int
    file_header_size: int
    body_header_size: int
    frame_header_size: int
    file_created_utc_ticks: int
    last_written_utc_ticks: int
    total_frames_written: int
    total_payload_bytes: int
    reserved1: int
    node_id: str


@dataclass
class BodyHeader:
    config_magic: str
    config_version: int
    body_header_size: int
    frame_header_size: int
    reserved1: int
    total_frame_size: int
    n_samples_per_chirp: int
    n_chirps_per_frame: int  # semantically "per TX" in your format (but we also handle "total" via inference)
    bits_per_sample: int
    n_receivers: int
    n_transmitters: int
    sample_type: int           # 0=real, 1=complex(IQ)
    data_order: int            # 0=ByChannel
    iq_order: int              # 0..3
    sample_format: int         # 0=16-bit aligned
    reserved2: int
    start_freq_ghz: float
    bandwidth_ghz: float
    idle_time_us: float
    tx_start_time_us: float
    adc_start_time_us: float
    ramp_end_time_us: float
    sample_rate_ksps: float
    slope_mhz_per_us: float
    frame_periodicity_ms: float
    bytes_per_element: int
    bytes_per_sample: int
    samples_per_frame: int
    bytes_per_frame: int
    max_range_m: float
    max_velocity_mps: float
    range_resolution_m: float
    velocity_resolution_mps: float
    _reserved3: bytes


@dataclass
class FrameHeader:
    frame_magic: str
    header_type: int
    frame_header_size: int
    frame_timestamp_utc_ticks: int
    frame_number: int
    frame_payload_size: int


# ==============================
# Public API
# ==============================

def parse_dopplium_raw(
    filename: str,
    *,
    max_frames: Optional[int] = None,
    cast: str = "float32",          # 'float32' | 'float64' | 'int16' (for real sample_type==0)
    return_complex: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium raw file into a numpy array shaped [samples, chirpsPerTx, channel, frame].

    - If nTx==1: channel == nRx (receiver index).
    - If nTx>1 : channel == tx*nRx + rx  (0-based; tx in [0..nTx-1], rx in [0..nRx-1]).
    """
    with open(filename, "rb") as f:
        # Peek magic + endianness
        magic = f.read(4).decode("ascii")
        if magic != "DOPP":
            raise ValueError("Invalid magic; not a Dopplium file.")
        f.seek(6, io.SEEK_SET)
        endianness_byte = f.read(1)
        if len(endianness_byte) != 1:
            raise ValueError("Unable to read endianness byte.")
        endianness = endianness_byte[0]
        endian_prefix = "<" if endianness == 1 else ">"
        if endianness not in (0, 1):
            # default little-endian if unknown
            endian_prefix = "<"

        # Rewind and read headers with correct endianness
        f.seek(0, io.SEEK_SET)
        FH = _read_file_header(f, endian_prefix)
        if FH.message_type != 3:
            raise ValueError("This file is not RawData (message_type != 3).")
        BH = _read_body_header(f, endian_prefix)

        if verbose:
            _print_header_summary(FH, BH)

        # Sizes & checks
        S = int(BH.n_samples_per_chirp)
        Cptx_hdr = int(BH.n_chirps_per_frame)   # typically "per TX"
        nRx = int(BH.n_receivers)
        nTx = int(BH.n_transmitters)
        if BH.data_order not in (0, 1):
            raise NotImplementedError(f"Unsupported data_order={BH.data_order}. Only 0 (ByChannel) and 1 (ByChirp) are supported.")
        if BH.sample_format != 0:
            raise NotImplementedError("Only sample_format=0 (16-bit aligned) is supported.")
        if BH.bits_per_sample != 16:
            raise NotImplementedError("Expected 16 bits per sample.")

        bytes_per_frame = int(BH.bytes_per_frame)
        bytes_per_element = int(BH.bytes_per_element)
        bytes_per_sample = int(BH.bytes_per_sample)
        ints_per_element = bytes_per_element // 2
        elements_per_sample = bytes_per_sample // bytes_per_element
        if elements_per_sample * bytes_per_element != bytes_per_sample:
            # Non-integer elements_per_sample; round but warn
            elements_per_sample = int(round(bytes_per_sample / bytes_per_element))
        block_len_ints = S * elements_per_sample * ints_per_element  # int16 count for one (channel, chirp)

        # Determine number of frames from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        frame_unit = BH.frame_header_size + bytes_per_frame
        n_frames_total = max(0, bytes_after_headers // frame_unit)
        n_frames = n_frames_total if max_frames is None else min(n_frames_total, max_frames)

        # Infer "total chirps on wire" per frame from header bytes
        n_int16_hdr = bytes_per_frame // 2
        denom = block_len_ints * nRx
        if denom == 0:
            raise ValueError("Invalid header values (zero-sized denominator for chirp inference).")
        Ctot_hdr_float = n_int16_hdr / denom
        Ctot_hdr = int(round(Ctot_hdr_float))
        if abs(Ctot_hdr - Ctot_hdr_float) > 1e-6 and verbose:
            print(f"Warning: total-chirps inference non-integer ({Ctot_hdr_float:.3f}); rounding to {Ctot_hdr}.")

        # Decide chirpsPerTx we expose on dim-2
        # Primary: if (Cptx_hdr * nTx == Ctot_hdr) -> header is per-TX
        # Else if (Cptx_hdr == Ctot_hdr) -> header was "total chirps"
        # Else: general fallback -> chirpsPerTx = ceil(Ctot_hdr / max(nTx,1))
        if nTx > 1 and Cptx_hdr * nTx == Ctot_hdr:
            chirps_per_tx = Cptx_hdr
            interpret = "per-tx"
        elif Cptx_hdr == Ctot_hdr:
            chirps_per_tx = max(1, math.ceil(Ctot_hdr / max(nTx, 1)))
            interpret = "total"
        else:
            chirps_per_tx = max(1, math.ceil(Ctot_hdr / max(nTx, 1)))
            interpret = "inferred"

        if verbose:
            print(f"Chirp interpretation: {interpret} | total_on_wire={Ctot_hdr} | "
                  f"chirpsPerTx(dim-2)={chirps_per_tx} | nTx={nTx}")

        # Channels dimension
        if nTx > 1:
            n_chan_out = nTx * nRx   # tx-rx combos
        else:
            n_chan_out = nRx

        # Allocate output
        if BH.sample_type == 0:
            out_dtype = _map_out_dtype(sample_type=0, cast=cast, return_complex=False)
            data = np.zeros((S, chirps_per_tx, n_chan_out, n_frames), dtype=out_dtype)
        else:
            if return_complex:
                out_dtype = np.complex64 if cast == "float32" else np.complex128
                data = np.zeros((S, chirps_per_tx, n_chan_out, n_frames), dtype=out_dtype)
            else:
                data = np.zeros((S, chirps_per_tx, n_chan_out, n_frames), dtype=np.int16)

        # Read frames
        f.seek(FH.file_header_size + BH.body_header_size, io.SEEK_SET)
        frame_headers = []

        dt_int16 = np.dtype("<i2" if endian_prefix == "<" else ">i2")

        for fi in range(n_frames):
            FR = _read_frame_header(f, endian_prefix)
            frame_headers.append(FR)
            if FR.frame_payload_size != bytes_per_frame:
                raise ValueError(f"Frame {fi+1}: payload size mismatch: header={FR.frame_payload_size}, "
                                 f"expected={bytes_per_frame}")

            # Read payload exactly as many int16 as header declares
            n_int16_this = bytes_per_frame // 2
            buf = f.read(n_int16_this * 2)
            if len(buf) != n_int16_this * 2:
                raise EOFError(f"Unexpected EOF while reading frame {fi+1} payload.")
            raw = np.frombuffer(buf, dtype=dt_int16, count=n_int16_this)

            # Normalize block ordering to (block_len_ints, nRx, Ctot_hdr)
            n_blocks = nRx * Ctot_hdr
            if raw.size != n_blocks * block_len_ints:
                raise ValueError(f"Frame {fi+1}: payload size does not match expected nBlocks*blockLenInts.")
            
            if BH.data_order == 0:
                # ByChannel: on-wire grouping is [for c in Ctot, for rx in nRx] contiguous blocks
                # MATLAB: reshape(raw, blockLenInts, nRx, Ctot) with column-major (Fortran order)
                # Equivalent with C-order: reverse dimensions then transpose
                buf_reshaped = raw.reshape((Ctot_hdr, nRx, block_len_ints))
                buf_reshaped = np.transpose(buf_reshaped, (2, 1, 0))  # -> (blockLen, nRx, Ctot)
            elif BH.data_order == 1:
                # ByChirp: on-wire grouping is [for rx in nRx, for c in Ctot] contiguous blocks
                # MATLAB: reshape(raw, nRx, blockLenInts, Ctot) then permute([2,1,3])
                # Equivalent with C-order: reverse dimensions then combine transposes
                buf_reshaped = raw.reshape((Ctot_hdr, block_len_ints, nRx))
                buf_reshaped = np.transpose(buf_reshaped, (1, 2, 0))  # -> (blockLen, nRx, Ctot)
            else:
                raise ValueError(f"Unsupported data_order={BH.data_order} (should have been caught earlier).")

            # Loop over "total on wire" chirps
            for c in range(Ctot_hdr):
                tx_idx = (c % max(nTx, 1))
                c_tx = c // max(nTx, 1)  # 0..ceil(Ctot/nTx)-1
                for rx in range(nRx):
                    seg = buf_reshaped[:, rx, c]
                    if seg.size != block_len_ints:
                        raise ValueError(f"Frame {fi+1}: payload indexing error at chirp {c}, rx {rx}. "
                                       f"Expected {block_len_ints} elements, got {seg.size}.")

                    if BH.sample_type == 0:
                        # REAL
                        vals = seg.astype(_map_real_float_dtype(cast)) if out_dtype != np.int16 else seg
                        if nTx == 1:
                            data[:, c_tx, rx, fi] = vals
                        else:
                            ch_out = tx_idx * nRx + rx
                            data[:, c_tx, ch_out, fi] = vals
                    else:
                        # COMPLEX
                        z = _decode_iq(seg, S, BH.iq_order, cast, return_complex=return_complex)
                        if return_complex:
                            if nTx == 1:
                                data[:, c_tx, rx, fi] = z
                            else:
                                ch_out = tx_idx * nRx + rx
                                data[:, c_tx, ch_out, fi] = z
                        else:
                            # uncommon: store only I to keep int16 shape
                            if nTx == 1:
                                data[:, c_tx, rx, fi] = z.real.astype(np.int16)
                            else:
                                ch_out = tx_idx * nRx + rx
                                data[:, c_tx, ch_out, fi] = z.real.astype(np.int16)

            # Validate that we processed the expected amount of data
            expected_int16 = n_blocks * block_len_ints
            if expected_int16 != n_int16_this:
                if verbose:
                    print(f"Warning: frame {fi+1} expected to process {expected_int16} int16 "
                          f"(nRx={nRx} * Ctot={Ctot_hdr} * blockLen={block_len_ints}), "
                          f"but header says {n_int16_this}.")

        headers = {
            "file": FH,
            "body": BH,
            "frame": frame_headers,
        }

        if verbose:
            print(f"\nParsed data shape: {tuple(data.shape)}  "
                  f"[samples, chirpsPerTx, channels, frames]")

        return data, headers


# ==============================
# Helpers
# ==============================

def _file_size(fh: io.BufferedReader) -> int:
    cur = fh.tell()
    fh.seek(0, io.SEEK_END)
    size = fh.tell()
    fh.seek(cur, io.SEEK_SET)
    return size


def _read_file_header(f: io.BufferedReader, ep: str) -> FileHeader:
    # Layout:
    # 4s magic, H version, B endianness, B compression, B product_id, B message_type,
    # H file_header_size, I body_header_size, I frame_header_size,
    # q file_created_ticks, q last_written_ticks,
    # I total_frames_written, I total_payload_bytes, I reserved1,
    # 32s node_id
    fmt = f"{ep}4sHBBBBHI I qqIII32s".replace(" ", "")
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read file header.")
    (magic, version, endianness, compression, product_id, message_type,
     file_header_size, body_header_size, frame_header_size,
     file_created_ticks, last_written_ticks, total_frames_written,
     total_payload_bytes, reserved1, node_id_bytes) = struct.unpack(fmt, raw)
    magic = magic.decode("ascii")
    node_id = node_id_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
    return FileHeader(
        magic, version, endianness, compression, product_id, message_type,
        file_header_size, body_header_size, frame_header_size,
        file_created_ticks, last_written_ticks,
        total_frames_written, total_payload_bytes, reserved1, node_id
    )


def _read_body_header(f: io.BufferedReader, ep: str) -> BodyHeader:
    # See MATLAB mapping; matches 192 bytes
    fmt = (
        f"{ep}"
        "4s"   # config_magic
        "H"    # config_version
        "H"    # body_header_size
        "H"    # frame_header_size
        "H"    # reserved1
        "I"    # total_frame_size
        "I"    # n_samples_per_chirp
        "I"    # n_chirps_per_frame (per TX)
        "H"    # bits_per_sample
        "H"    # n_receivers
        "H"    # n_transmitters
        "B"    # sample_type
        "B"    # data_order
        "B"    # iq_order
        "B"    # sample_format
        "H"    # reserved2
        "d"    # start_freq_ghz
        "d"    # bandwidth_ghz
        "d"    # idle_time_us
        "d"    # tx_start_time_us
        "d"    # adc_start_time_us
        "d"    # ramp_end_time_us
        "d"    # sample_rate_ksps
        "d"    # slope_mhz_per_us
        "d"    # frame_periodicity_ms
        "I"    # bytes_per_element
        "I"    # bytes_per_sample
        "I"    # samples_per_frame
        "I"    # bytes_per_frame
        "f"    # max_range_m
        "f"    # max_velocity_mps
        "f"    # range_resolution_m
        "f"    # velocity_resolution_mps
        "52s"  # _reserved3
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read body header.")
    unpacked = struct.unpack(fmt, raw)
    # Map into dataclass
    return BodyHeader(
        config_magic=unpacked[0].decode("ascii"),
        config_version=unpacked[1],
        body_header_size=unpacked[2],
        frame_header_size=unpacked[3],
        reserved1=unpacked[4],
        total_frame_size=unpacked[5],
        n_samples_per_chirp=unpacked[6],
        n_chirps_per_frame=unpacked[7],
        bits_per_sample=unpacked[8],
        n_receivers=unpacked[9],
        n_transmitters=unpacked[10],
        sample_type=unpacked[11],
        data_order=unpacked[12],
        iq_order=unpacked[13],
        sample_format=unpacked[14],
        reserved2=unpacked[15],
        start_freq_ghz=unpacked[16],
        bandwidth_ghz=unpacked[17],
        idle_time_us=unpacked[18],
        tx_start_time_us=unpacked[19],
        adc_start_time_us=unpacked[20],
        ramp_end_time_us=unpacked[21],
        sample_rate_ksps=unpacked[22],
        slope_mhz_per_us=unpacked[23],
        frame_periodicity_ms=unpacked[24],
        bytes_per_element=unpacked[25],
        bytes_per_sample=unpacked[26],
        samples_per_frame=unpacked[27],
        bytes_per_frame=unpacked[28],
        max_range_m=unpacked[29],
        max_velocity_mps=unpacked[30],
        range_resolution_m=unpacked[31],
        velocity_resolution_mps=unpacked[32],
        _reserved3=unpacked[33],
    )


def _read_frame_header(f: io.BufferedReader, ep: str) -> FrameHeader:
    fmt = f"{ep}4sHHqII"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read frame header.")
    (frame_magic_b, header_type, frame_header_size,
     frame_timestamp_utc_ticks, frame_number, frame_payload_size) = struct.unpack(fmt, raw)
    frame_magic = frame_magic_b.decode("ascii")
    if frame_magic != "FRME":
        raise ValueError("Invalid frame magic.")
    if frame_header_size < 24:
        raise ValueError("Unexpected frame_header_size in frame header.")
    return FrameHeader(
        frame_magic=frame_magic,
        header_type=header_type,
        frame_header_size=frame_header_size,
        frame_timestamp_utc_ticks=frame_timestamp_utc_ticks,
        frame_number=frame_number,
        frame_payload_size=frame_payload_size,
    )


def _map_out_dtype(sample_type: int, cast: str, return_complex: bool):
    if sample_type == 0:  # real
        if cast == "float64":
            return np.float64
        if cast == "int16":
            return np.int16
        return np.float32
    else:
        if return_complex:
            return np.complex128 if cast == "float64" else np.complex64
        else:
            return np.int16


def _map_real_float_dtype(cast: str):
    return np.float64 if cast == "float64" else np.float32


def _decode_iq(seg_int16: np.ndarray, S: int, iq_order: int, cast: str, return_complex: bool) -> np.ndarray:
    """
    Decode a single (channel,chirp) complex vector of length S from an int16 segment of length 2*S.
    Orders:
      0 = IQ           (I,Q interleaved)
      1 = QI           (Q,I interleaved)
      2 = NonInterleaved (IIII... QQQQ...)
      3 = BlockInterleaved ([I0 I1 Q0 Q1 I2 I3 Q2 Q3 ...])
    """
    if seg_int16.size != 2 * S:
        # try to coerce if header rounding gave slight mismatch
        if seg_int16.size < 2 * S:
            raise ValueError("Segment too small for complex IQ.")
        seg_int16 = seg_int16[: 2 * S]

    if iq_order == 0:  # IQ
        I = seg_int16[0::2]
        Q = seg_int16[1::2]
    elif iq_order == 1:  # QI
        Q = seg_int16[0::2]
        I = seg_int16[1::2]
    elif iq_order == 2:  # NonInterleaved
        I = seg_int16[:S]
        Q = seg_int16[S:2 * S]
    elif iq_order == 3:  # BlockInterleaved
        if S % 2 == 0:
            # MATLAB used column-major reshape. Recreate with order='F'.
            g = seg_int16.reshape((4, S // 2), order="F")
            I = g[0:2, :].reshape(-1, order="F")
            Q = g[2:4, :].reshape(-1, order="F")
        else:
            # robust fallback for odd S
            I = np.empty(S, dtype=np.int16)
            Q = np.empty(S, dtype=np.int16)
            ii = qi = 0
            k = 0
            twoS = 2 * S
            while k < twoS:
                take = min(2, S - ii)
                if take > 0:
                    I[ii:ii + take] = seg_int16[k:k + take]
                    k += take
                    ii += take
                take = min(2, S - qi)
                if take > 0:
                    Q[qi:qi + take] = seg_int16[k:k + take]
                    k += take
                    qi += take
    else:
        raise ValueError(f"Unsupported iq_order: {iq_order}")

    if return_complex:
        Iflt = I.astype(_map_real_float_dtype(cast), copy=False)
        Qflt = Q.astype(_map_real_float_dtype(cast), copy=False)
        return Iflt + 1j * Qflt
    else:
        # return_complex=False: we still return a complex vector upstream but store only real later
        Iflt = I.astype(_map_real_float_dtype(cast), copy=False)
        Qflt = Q.astype(_map_real_float_dtype(cast), copy=False)
        return Iflt + 1j * Qflt


def _print_header_summary(FH: FileHeader, BH: BodyHeader) -> None:
    print("--- Dopplium Raw Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  FrameHdr={BH.frame_header_size}  TotalFramesWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    print("\n-- Radar Config --")
    stype = "Real" if BH.sample_type == 0 else "Complex"
    print(f"Samples/Chirp={BH.n_samples_per_chirp}  ChirpsPerTX/Frame={BH.n_chirps_per_frame}  Rx={BH.n_receivers}  Tx={BH.n_transmitters}")
    print(f"SampleType={stype}  IQOrder={BH.iq_order}  DataOrder={BH.data_order}  BitsPerSample={BH.bits_per_sample}")
    print(f"Bytes/Element={BH.bytes_per_element}  Bytes/Sample={BH.bytes_per_sample}  Bytes/Frame={BH.bytes_per_frame}  TotalFrameSize={BH.total_frame_size}")
    print(f"StartFreq={BH.start_freq_ghz:.3f} GHz  BW={BH.bandwidth_ghz:.3f} GHz  Fs={BH.sample_rate_ksps:.1f} ksps  Slope={BH.slope_mhz_per_us:.3f} MHz/us")
    print(f"FramePeriod={BH.frame_periodicity_ms:.3f} ms  RampEnd={BH.ramp_end_time_us:.3f} us")
