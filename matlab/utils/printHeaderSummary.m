function printHeaderSummary(FH, BH)
% PRINTHEADERSUMMARY Display formatted summary of file and body headers
%   printHeaderSummary(FH, BH)
%
%   INPUTS
%     FH : file header struct
%     BH : body header struct

    fprintf('--- Dopplium Raw Data ---\n');
    fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
        FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
    fprintf('FileHdr=%d  BodyHdr=%d  FrameHdr=%d  TotalFramesWritten=%d\n', ...
        FH.file_header_size, BH.body_header_size, BH.frame_header_size, FH.total_frames_written);
    fprintf('NodeId="%s"\n', FH.node_id);

    fprintf('\n-- Radar Config --\n');
    fprintf('Samples/Chirp=%d  ChirpsPerTX/Frame=%d  Rx=%d  Tx=%d\n', ...
        BH.n_samples_per_chirp, BH.n_chirps_per_frame, BH.n_receivers, BH.n_transmitters);
    fprintf('SampleType=%s  IQOrder=%d  DataOrder=%d  BitsPerSample=%d\n', ...
        tern(BH.sample_type==0,'Real','Complex'), BH.iq_order, BH.data_order, BH.bits_per_sample);
    fprintf('Bytes/Element=%d  Bytes/Sample=%d  Bytes/Frame=%d  TotalFrameSize=%d\n', ...
        BH.bytes_per_element, BH.bytes_per_sample, BH.bytes_per_frame, BH.total_frame_size);
    fprintf('StartFreq=%.3f GHz  BW=%.3f GHz  Fs=%.1f ksps  Slope=%.3f MHz/us\n', ...
        BH.start_freq_ghz, BH.bandwidth_ghz, BH.sample_rate_ksps, BH.slope_mhz_per_us);
    fprintf('FramePeriod=%.3f ms  RampEnd=%.3f us\n', ...
        BH.frame_periodicity_ms, BH.ramp_end_time_us);
end
