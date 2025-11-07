clear
close all
%%

if ismac
    folder="/Users/mareike/Library/CloudStorage/OneDrive-DelftUniversityofTechnology";
    addpath(genpath("../../Processing_Code_Dataset_2025"))
elseif ispc
    folder="C:\Users\mwendelmuth\OneDrive - Delft University of Technology";
    addpath(genpath("../../Processing_Code_Dataset_2025"))
else
    error("No idea what operating system")
end
%%
file="20251022_123849_120_breathing_sitting_1_Node 120.bin";
folder=fullfile(folder,"Dopplium_Recordings","2025-11-05",file);

% Example usage of parseDoppliumRaw
[data, hdr] = parseDoppliumRaw(folder);
%%
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
%% range plot
chirpIdx=50;
sub_data=data(:,:,:,1:end);
decluttered_data=sub_data;%-mean(sub_data,4);

MTI=windowed_fft(decluttered_data,1,128,1);
MTI=fft(decluttered_data);
X_db = mag2db(abs(MTI));

figure;
imagesc(squeeze(X_db(:,chirpIdx,chIdx,:)));
axis xy;            % ensure origin is lower-left like Python's origin="lower"
xlabel('Frames');
ylabel('Range bins');
colorbar;
title(sprintf(' FFT (chirp %d, channel %d)', chirpIdx, chIdx));