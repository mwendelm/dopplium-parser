% Example: Parse and visualize ADC data with physical axes
% This example demonstrates parsing ADC/RawData and computing range and velocity axes

[data, hdr] = doppliumParser("new.bin");

% Data shape: [samples, chirpsPerTx, channels, frames]
dims = size(data);
fprintf('Parsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
    dims(1), dims(2), dims(3), dims(4));

% -------------------------------------------------------------------------
% Compute physical axes from radar parameters
% -------------------------------------------------------------------------
c = 3e8;  % Speed of light (m/s)

% Extract radar parameters from header
bandwidth_hz = hdr.body.bandwidth_ghz * 1e9;              % GHz -> Hz
start_freq_hz = hdr.body.start_freq_ghz * 1e9;           % GHz -> Hz
center_freq_hz = start_freq_hz + bandwidth_hz / 2;       % Center frequency
wavelength = c / center_freq_hz;                          % Wavelength (m)

n_samples = hdr.body.n_samples_per_chirp;
n_chirps_per_tx = hdr.body.n_chirps_per_frame;
ramp_end_time_s = hdr.body.ramp_end_time_us * 1e-6;     % μs -> s

% Range resolution and axis
range_resolution = c / (2 * bandwidth_hz);                % meters
range_axis = (0:n_samples-1) * range_resolution;          % meters

% Velocity resolution and axis (Doppler)
frame_time = n_chirps_per_tx * ramp_end_time_s;          % Total frame time
velocity_resolution = wavelength / (2 * frame_time);      % m/s
velocity_axis = ((-n_chirps_per_tx/2):(n_chirps_per_tx/2-1)) * velocity_resolution;  % m/s (centered)

% Display computed parameters
fprintf('\n--- Radar Parameters ---\n');
fprintf('Center Frequency: %.2f GHz\n', center_freq_hz / 1e9);
fprintf('Bandwidth: %.2f MHz\n', bandwidth_hz / 1e6);
fprintf('Wavelength: %.4f m\n', wavelength);
fprintf('Range Resolution: %.4f m\n', range_resolution);
fprintf('Max Range: %.2f m\n', range_axis(end));
fprintf('Velocity Resolution: %.4f m/s\n', velocity_resolution);
fprintf('Max Velocity: ±%.2f m/s\n', max(abs(velocity_axis)));

% -------------------------------------------------------------------------
% Select data and compute 2D FFT
% -------------------------------------------------------------------------
frameIdx = 52;  % Select frame
chIdx    = 1;    % Select channel (if multi-TX: linear index tx*nRx+rx)
slab = data(:,:,chIdx,frameIdx);  % shape [samples, chirpsPerTx]

% Optional windowing (uncomment to reduce sidelobes)
% win_r = hann(size(slab,1));
% win_d = hann(size(slab,2));
% slab = slab .* (win_r * win_d.');

% 2D FFT: range (dim 1) x Doppler (dim 2)
X = fft2(slab);

% fftshift along Doppler dimension (dim 2) to center zero velocity
X_shifted = fftshift(X, 2);

% Convert to dB
X_db = mag2db(abs(X_shifted));

% -------------------------------------------------------------------------
% Plot with physical axes
% -------------------------------------------------------------------------
figure;
imagesc(velocity_axis, range_axis, X_db);
axis xy;  % Origin at lower-left
xlabel('Velocity (m/s)');
ylabel('Range (m)');
colorbar;
title(sprintf('Range-Doppler Map (frame %d, channel %d)', frameIdx, chIdx));
grid on;
