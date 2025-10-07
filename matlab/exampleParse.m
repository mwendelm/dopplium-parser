% Example usage of parseDoppliumRaw
[data, hdr] = parseDoppliumRaw("20251006_162727_3TxGood.bin");

% Data shape: [samples, chirpsPerTx, channels, frames]
dims = size(data);
fprintf('Parsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
    dims(1), dims(2), dims(3), dims(4));

% --- Select one channel and one frame ---
frameIdx = 1;  % first frame
chIdx    = 1;  % first channel (if multi-TX: linear index tx*nRx+rx)
slab = data(:,:,chIdx,frameIdx);  % shape [samples, chirpsPerTx]

% --- Optional windowing ---
% win_r = hann(size(slab,1));
% win_d = hann(size(slab,2));
% slab = slab .* (win_r * win_d.');

% --- 2D FFT: range (dim 1) x Doppler (dim 2) ---
X = fft2(slab);

% --- fftshift along Doppler dimension (dim 2) ---
X_shifted = fftshift(X, 2);

% --- Convert to dB ---
X_db = mag2db(abs(X_shifted));

% --- Plot (imagesc equivalent to Python imshow) ---
figure;
imagesc(X_db);
axis xy;            % ensure origin is lower-left like Python's origin="lower"
xlabel('Doppler bins (fftshifted)');
ylabel('Range bins');
colorbar;
title(sprintf('2D FFT (frame %d, channel %d)', frameIdx, chIdx));
