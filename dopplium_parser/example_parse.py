import numpy as np
import matplotlib.pyplot as plt
from dopplium_parser import parse_dopplium_raw

def mag2db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Magnitude to dB, safe for zeros."""
    return 20.0 * np.log10(np.abs(x) + eps)

def main():
    # --- Parse file ---
    data, hdr = parse_dopplium_raw(
        "PATH_TO_BIN",
        cast="float32",
        return_complex=True,
        verbose=True
    )
    # data shape: [samples, chirpsPerTx, channels, frames]
    S, Cptx, K, F = data.shape
    print("Parsed shape [S, chirpsPerTx, channels, frames]:", data.shape)

    # --- Select one channel and one frame ---
    frame_idx = 0  # pick the first frame
    ch_idx = 0     # pick channel 0 (for multi-TX this is TX0-RX0)
    # If you want a specific (tx, rx) when nTx>1: ch_idx = tx * nRx + rx

    slab = data[:, :, ch_idx, frame_idx]  # shape [S, Cptx], complex

    # --- 2D FFT: range (over samples) x Doppler (over chirpsPerTx) ---
    # FFT over axis 0 (samples) and axis 1 (chirpsPerTx)
    X = np.fft.fft2(slab, axes=(0, 1))

    # --- fftshift along SECOND dimension only (chirpsPerTx) ---
    X_shifted = np.fft.fftshift(X, axes=1)

    # --- Convert to dB ---
    X_db = mag2db(X_shifted)

    # --- Plot (imagesc equivalent) ---
    plt.figure()
    im = plt.imshow(
        X_db,
        origin="lower",        # like imagesc
        aspect="auto",         # stretch to fill axes
        interpolation="nearest"
    )
    plt.colorbar(im, label="Magnitude (dB)")
    plt.title(f"2D FFT (frame {frame_idx}, channel {ch_idx})")
    plt.xlabel("Doppler bins")
    plt.ylabel("Range bins")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
