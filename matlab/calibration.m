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
load("calib.mat")
%%
%file="20251022_143100_122_empty2.bin";
file_corner="4-2.bin";
file_empty="empty.bin";

folder_corner=fullfile(folder,"Dataset_2025","24-11-2025","calibration","backNode",file_corner);
folder_empty=fullfile(folder,"Dataset_2025","24-11-2025","calibration","backNode",file_empty);

% Example usage of parseDoppliumRaw
%[data_empty, hdr] = doppliumParser(folder_empty);
[data_empty, hdr_left] = doppliumParser(folder_empty); 
[data_corner, hdr_right] = doppliumParser(folder_corner); 
%%
% Data shape: [samples, chirpsPerTx, channels, frames]
dims = size(data_corner);
fprintf('Parsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
    dims(1), dims(2), dims(3), dims(4));
%% range plot
range_axis=linspace(0,80*0.13,256);

chirpIdx=40;
decluttered_data=data_corner(1:75,:,:,:)-mean(data_empty(1:75,:,:,:),4);

MTI=windowed_fft(decluttered_data,1,256,1);
%MTI=fft(decluttered_data);
X_db = mag2db(abs(MTI));

%% beamforming
subcube=MTI(1:100,:,:,:);
%include calibration
c=repmat(reshape(calib_phase,[1 1 12 1]),[100 120 1 size(MTI,4)]);
subcube=subcube.*c;
cube_pattern=zeros([2,8,size(subcube,1),size(subcube,2),size(subcube,4)]);
cube_pattern(2,1,:,:,:)=squeeze(subcube(:,:,1,:));
cube_pattern(2,2,:,:,:)=squeeze(subcube(:,:,2,:));
cube_pattern(2,3,:,:,:)=squeeze(subcube(:,:,3,:));
cube_pattern(2,4,:,:,:)=squeeze(subcube(:,:,4,:));
cube_pattern(2,5,:,:,:)=squeeze(subcube(:,:,5,:));
cube_pattern(2,6,:,:,:)=squeeze(subcube(:,:,6,:));
cube_pattern(2,7,:,:,:)=squeeze(subcube(:,:,7,:));
cube_pattern(2,8,:,:,:)=squeeze(subcube(:,:,8,:));

cube_pattern(1,3,:,:,:)=squeeze(subcube(:,:,9,:));
cube_pattern(1,4,:,:,:)=squeeze(subcube(:,:,10,:));
cube_pattern(1,5,:,:,:)=squeeze(subcube(:,:,11,:));
cube_pattern(1,6,:,:,:)=squeeze(subcube(:,:,12,:));

azim_cube=fftshift(windowed_fft(cube_pattern,2,32,0),2); %azimuth
%elev_cube=fftshift(windowed_fft(cube_pattern,1,16,0),1); %elevation


%% azim map
figure
angle_axis=linspace(-1,1,32);
imagesc(range_axis(1:100),angle_axis,squeeze(mag2db(abs(azim_cube(2,:,:,40,20)))))
colorbar
xlabel('Range axis (m)')
ylabel('Azim (au)')

% find peak
[~,loc]=max(squeeze(mag2db(abs(azim_cube(2,:,:,40,20)))),[],"all");
[row,col]=ind2sub([32,100],loc);

range=round(range_axis(col),2)
col
angle=round(angle_axis(row),2)
