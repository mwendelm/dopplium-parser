clear
%close all
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
file="20251022_143100_122_empty2.bin";
file2="20251022_135106_122_corner_2_7m.bin";
%file="20251022_122651_120_target_at_minus_50cm_Node 120.bin";
folder_empty=fullfile(folder,"Dopplium_Recordings","2025-11-12","Node122",file);
folder2=fullfile(folder,"Dopplium_Recordings","2025-11-12","Node122",file2);
%folder=fullfile(folder,"measurement_Rutkay","Node120",file);

% Example usage of parseDoppliumRaw
[data_empty, hdr] = parseDoppliumRaw(folder_empty);
[data, hdr] = parseDoppliumRaw(folder2);
%%
% Data shape: [samples, chirpsPerTx, channels, frames]
dims = size(data_empty);
fprintf('Parsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
    dims(1), dims(2), dims(3), dims(4));

% --- Select one channel and one frame ---
frameIdx = 100;  % first frame
chIdx    = 1;  % first channel (if multi-TX: linear index tx*nRx+rx)
slab = data_empty(:,:,chIdx,frameIdx);  % shape [samples, chirpsPerTx]

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
chirpIdx=40;
%data_empty(6:8,:,:,:)=0;
sub_data=data(:,:,:,:);
decluttered_data=data-mean(data_empty,4);
%decluttered_data=sub_data;%-mean(sub_data,4);

%MTI=windowed_fft(decluttered_data,1,80,1);
MTI=fft(decluttered_data);
X_db = mag2db(abs(MTI));

figure;
imagesc(squeeze(X_db(:,chirpIdx,chIdx,:)));
axis xy;            % ensure origin is lower-left like Python's origin="lower"
xlabel('Frames');
ylabel('Range bins');
colorbar;
title(sprintf('Dopplium FFT (chirp %d, channel %d)', chirpIdx, chIdx));