function [data, headers] = parseDoppliumRaw(filename, opts)
% PARSEDOPPLIUMRAW Parse Dopplium Raw Data file into [S, Cptx, K, F]
%   [data, headers] = parseDoppliumRaw(filename, opts)
%
%   OUTPUTS
%     data    : array shaped [samples, chirpsPerTx, channel, frame]
%               - if nTx==1 => channel == nRx (receiver index)
%               - if nTx>1  => channel == tx-rx combination, linearized as (tx-1)*nRx + rx
%     headers : struct with parsed file/body/frame headers and derived info
%
%   OPTS (all optional)
%     .maxFrames      : limit number of frames to read (default: Inf = all)
%     .cast           : 'double'|'single'|'int16' for output samples (default 'single')
%     .returnComplex  : true/false, if complex IQ => return complex numbers (default true)
%     .verbose        : true/false (default true)
%
%   Notes:
%   - n_chirps_per_frame in the body header is CHIRPS PER TX.
%     Total chirps on the wire per frame = nTx * n_chirps_per_frame (TX interleaved).
%   - Supported payload order: data_order == 0 (ByChannel) and data_order == 1 (ByChirp).
%   - Supported packing: sample_format == 0 (16-bit aligned int16 containers).

if nargin < 2, opts = struct; end
opts = setDefault(opts, 'maxFrames', Inf);
opts = setDefault(opts, 'cast', 'single');
opts = setDefault(opts, 'returnComplex', true);
opts = setDefault(opts, 'verbose', true);

% -------------------------------------------------------------------------
% Open & endianness
% -------------------------------------------------------------------------
[fid, msg] = fopen(filename, 'r', 'ieee-le');
assert(fid > 0, ['Failed to open file: ' msg]);
cleanup = onCleanup(@() fclose(fid));

magic = fread(fid, [1,4], '*char');
assert(strcmp(magic, 'DOPP'), 'Invalid magic. Not a Dopplium file.');

fseek(fid, 6, 'bof');
endianness = fread(fid, 1, 'uint8');
machinefmt = 'ieee-le';
if endianness == 0
    machinefmt = 'ieee-be';
elseif endianness ~= 1
    warning('Unknown endianness value %d. Assuming little-endian.', endianness);
end

fseek(fid, 0, 'bof');
FH = readFileHeader(fid, machinefmt);
assert(FH.file_header_size >= 80, 'Unexpected file_header_size.');
assert(FH.message_type == 3, 'This file is not RawData (message_type ~= 3).');

BH = readBodyHeader(fid, machinefmt);
assert(BH.body_header_size >= 192, 'Unexpected body_header_size (BH).');
assert(BH.frame_header_size >= 24, 'Unexpected frame_header_size (BH).');

if opts.verbose
    printHeaderSummary(FH, BH);
end

% -------------------------------------------------------------------------
% Dimensions & types (trust header bytes for sizing)
% -------------------------------------------------------------------------
S      = double(BH.n_samples_per_chirp);
Cptx   = double(BH.n_chirps_per_frame); % chirps per TX
nRx    = double(BH.n_receivers);
nTx    = double(BH.n_transmitters);
Ctot   = Cptx * max(nTx,1);              % total chirps on wire (TX interleaved)
bytesPerFrame = double(BH.bytes_per_frame);

if ~(BH.data_order == 0 || BH.data_order == 1)
    error('Unsupported data_order=%d. Only 0 (ByChannel) and 1 (ByChirp) are supported.', BH.data_order);
end
assert(BH.sample_format == 0, 'Only 16-bit aligned samples supported (sample_format==0).');
assert(double(BH.bits_per_sample) == 16, 'Expected 16 bits per sample.');

% Header-guided sizing
bytesPerElement   = double(BH.bytes_per_element);     % should be 2 if int16 containers
bytesPerSample    = double(BH.bytes_per_sample);      % bytes per (real or complex) sample
intsPerElement    = bytesPerElement / 2;              % normally 1
elementsPerSample = bytesPerSample / bytesPerElement; % 1 for real, 2 for complex IQ
if abs(elementsPerSample - round(elementsPerSample)) > 1e-9
    warning('Non-integer elementsPerSample detected from header (%.3f). Rounding.', elementsPerSample);
end
elementsPerSample = round(elementsPerSample);

% Expected ints per (channel,chirp) block, from header bytes
blockLenInts = S * elementsPerSample * intsPerElement;

% Determine number of frames from file size
fileInfo = dir(filename);
bytesAfterHeaders = fileInfo.bytes - FH.file_header_size - BH.body_header_size;
bytesPerUnit = BH.frame_header_size + bytesPerFrame;
nFramesTotal = floor(bytesAfterHeaders / bytesPerUnit);
nFrames = min(nFramesTotal, opts.maxFrames);
if isfinite(opts.maxFrames) && opts.maxFrames > nFramesTotal
    warning('Requested maxFrames exceeds file content. Reading %d frames.', nFramesTotal);
end

% Output dimensions
chirpsPerTx = Cptx;
if nTx > 1
    nChanOut = nTx * nRx;
else
    nChanOut = nRx;
end

% Allocate output
outClass = mapOutClass(BH.sample_type, opts.cast, opts.returnComplex);
if BH.sample_type == 0 % real
    data = zeros(S, chirpsPerTx, nChanOut, nFrames, outClass);
else % complex
    if opts.returnComplex
        t = mapFloat(opts.cast);
        data = complex(zeros(S, chirpsPerTx, nChanOut, nFrames, t), ...
                       zeros(S, chirpsPerTx, nChanOut, nFrames, t));
    else
        data = zeros(S, chirpsPerTx, nChanOut, nFrames, 'int16'); % uncommon path
    end
end

% -------------------------------------------------------------------------
% Read frames
% -------------------------------------------------------------------------
fseek(fid, FH.file_header_size + BH.body_header_size, 'bof');
frames = repmat(emptyFrameHeader(), nFrames, 1);

for f = 1:nFrames
    FR = readFrameHeader(fid, machinefmt);
    frames(f) = FR;

    if FR.frame_payload_size ~= bytesPerFrame
        error('Frame %d payload size mismatch: header=%d, expected=%d', ...
              f, FR.frame_payload_size, bytesPerFrame);
    end

    % Ground truth count from header:
    nInt16_hdr = bytesPerFrame / 2;
    raw = fread(fid, nInt16_hdr, '*int16', 0, machinefmt);
    if numel(raw) ~= nInt16_hdr
        error('Unexpected EOF while reading frame %d payload.', f);
    end

    % Theoretical count (for info only)
    nInt16_theo = blockLenInts * nRx * Ctot;
    if nInt16_theo ~= nInt16_hdr
        warning(['Frame %d: header bytes imply %d int16, but theoretical calc suggests %d. ' ...
                 'Proceeding with header-derived sizing.'], f, nInt16_hdr, nInt16_theo);
    end

    % ---- Normalize block ordering to (blockLenInts, rx, c) ----
    nBlocks = nRx * Ctot;
    assert(numel(raw) == nBlocks * blockLenInts, ...
        'Frame %d: payload size does not match expected nBlocks*blockLenInts.', f);

    switch BH.data_order
        case 0 % ByChannel: on-wire grouping is [for c=1..Ctot, for rx=1..nRx] contiguous blocks
            buf = reshape(raw, blockLenInts, nRx, Ctot); % (ints, rx, c)

        case 1 % ByChirp: on-wire grouping is [for rx=1..nRx, for c=1..Ctot] contiguous blocks
            buf = reshape(raw, nRx, blockLenInts, Ctot); % (ints, c, rx)
            buf = permute(buf, [2, 1, 3]); % reorder to (ints, rx, c) for uniform processing

        otherwise
            error('Unsupported data_order=%d (should have been caught earlier).', BH.data_order);
    end

    % Populate output from buf(:, rx, c)
    if BH.sample_type == 0
        % ------------------------ REAL ------------------------
        % elementsPerSample should be 1 -> blockLenInts = S
        for c = 1:Ctot
            tx   = mod(c-1, nTx) + 1;
            c_tx = floor((c-1)/max(nTx,1)) + 1;
            for rx = 1:nRx
                seg = buf(:, rx, c); % int16 column
                if nTx == 1
                    data(:, c_tx, rx, f) = cast(seg, outClass);
                else
                    chOut = (tx-1)*nRx + rx;
                    data(:, c_tx, chOut, f) = cast(seg, outClass);
                end
            end
        end

    else
        % ------------------------ COMPLEX ------------------------
        % elementsPerSample should be 2 -> blockLenInts = 2*S
        for c = 1:Ctot
            tx   = mod(c-1, nTx) + 1;
            c_tx = floor((c-1)/max(nTx,1)) + 1;
            for rx = 1:nRx
                seg = buf(:, rx, c); % int16 column of length elementsPerSample*S
                if elementsPerSample == 2
                    z = decodeIQ(seg, S, BH.iq_order, opts); % complex vector (Sx1)
                else
                    % Fallback: treat first half as I, second half as Q
                    half = numel(seg)/2;
                    I = seg(1:half);
                    Q = seg(half+1:end);
                    z = complex(cast(I, mapFloat(opts.cast)), cast(Q, mapFloat(opts.cast)));
                end

                if nTx == 1
                    if opts.returnComplex
                        data(:, c_tx, rx, f) = cast(z, mapFloat(opts.cast));
                    else
                        data(:, c_tx, rx, f) = int16(real(z));
                    end
                else
                    chOut = (tx-1)*nRx + rx;
                    if opts.returnComplex
                        data(:, c_tx, chOut, f) = cast(z, mapFloat(opts.cast));
                    else
                        data(:, c_tx, chOut, f) = int16(real(z));
                    end
                end
            end
        end
    end
end

% -------------------------------------------------------------------------
% Outputs
% -------------------------------------------------------------------------
headers.file  = FH;
headers.body  = BH;
headers.frame = frames;

if opts.verbose
    fprintf('\nParsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
        size(data,1), size(data,2), size(data,3), size(data,4));
    if nTx > 1
        fprintf('Multi-TX: TX interleaved by chirp. nTx=%d, total chirps on wire per frame=%d (= nTx * chirpsPerTx)\n', ...
            nTx, Ctot);
    else
        fprintf('Single-TX: total chirps per frame=%d\n', Cptx);
    end
end
end

% ====================== Helpers ======================
function FH = readFileHeader(fid, machinefmt)
    FH.magic                   = char(fread(fid, [1,4], '*char'));
    FH.version                 = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.endianness              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.compression             = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.product_id              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.message_type            = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.file_header_size        = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.body_header_size        = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.frame_header_size       = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.file_created_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.last_written_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.total_frames_written    = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.total_payload_bytes     = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.reserved1               = fread(fid, 1, 'uint32', 0, machinefmt);
    nodeBytes                  = fread(fid, 32, '*uint8', 0, machinefmt);
    FH.node_id                 = deblank(char(nodeBytes(:)')); % null-terminated ASCII
end

function BH = readBodyHeader(fid, machinefmt)
    BH.config_magic            = char(fread(fid, [1,4], '*char'));
    BH.config_version          = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.body_header_size        = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.frame_header_size       = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.reserved1               = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.total_frame_size        = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.n_samples_per_chirp     = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.n_chirps_per_frame      = fread(fid, 1, 'uint32', 0, machinefmt); % per TX
    BH.bits_per_sample         = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.n_receivers             = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.n_transmitters          = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.sample_type             = fread(fid, 1, 'uint8',  0, machinefmt); % 0=real, 1=complex/IQ
    BH.data_order              = fread(fid, 1, 'uint8',  0, machinefmt); % 0=ByChannel, 1=ByChirp
    BH.iq_order                = fread(fid, 1, 'uint8',  0, machinefmt); % 0..3
    BH.sample_format           = fread(fid, 1, 'uint8',  0, machinefmt); % 0=16b aligned
    BH.reserved2               = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.start_freq_ghz          = fread(fid, 1, 'double', 0, machinefmt);
    BH.bandwidth_ghz           = fread(fid, 1, 'double', 0, machinefmt);
    BH.idle_time_us            = fread(fid, 1, 'double', 0, machinefmt);
    BH.tx_start_time_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.adc_start_time_us       = fread(fid, 1, 'double', 0, machinefmt);
    BH.ramp_end_time_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.sample_rate_ksps        = fread(fid, 1, 'double', 0, machinefmt);
    BH.slope_mhz_per_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.frame_periodicity_ms    = fread(fid, 1, 'double', 0, machinefmt);
    BH.bytes_per_element       = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.bytes_per_sample        = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.samples_per_frame       = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.bytes_per_frame         = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.max_range_m             = fread(fid, 1, 'single', 0, machinefmt);
    BH.max_velocity_mps        = fread(fid, 1, 'single', 0, machinefmt);
    BH.range_resolution_m      = fread(fid, 1, 'single', 0, machinefmt);
    BH.velocity_resolution_mps = fread(fid, 1, 'single', 0, machinefmt);
    BH.reserved3               = fread(fid, 52, '*uint8', 0, machinefmt);
end

function FR = readFrameHeader(fid, machinefmt)
    FR.frame_magic               = char(fread(fid, [1,4], '*char'));
    FR.header_type               = fread(fid, 1, 'uint16', 0, machinefmt);
    FR.frame_header_size         = fread(fid, 1, 'uint16', 0, machinefmt);
    FR.frame_timestamp_utc_ticks = fread(fid, 1, 'int64', 0, machinefmt);
    FR.frame_number              = fread(fid, 1, 'uint32', 0, machinefmt);
    FR.frame_payload_size        = fread(fid, 1, 'uint32', 0, machinefmt);

    if ~strcmp(FR.frame_magic, 'FRME')
        error('Invalid frame magic at frame read position.');
    end
    if FR.frame_header_size < 24
        error('Unexpected frame_header_size in frame header.');
    end
end

function s = emptyFrameHeader()
    s = struct( ...
        'frame_magic',              '', ...
        'header_type',              0, ...
        'frame_header_size',        0, ...
        'frame_timestamp_utc_ticks',0, ...
        'frame_number',             0, ...
        'frame_payload_size',       0);
end

function z = decodeIQ(segInt16, S, iqOrder, opts)
    switch iqOrder
        case 0 % IQ
            I = segInt16(1:2:end);
            Q = segInt16(2:2:end);
        case 1 % QI
            Q = segInt16(1:2:end);
            I = segInt16(2:2:end);
        case 2 % NonInterleaved (IIII... QQQQ...)
            I = segInt16(1:S);
            Q = segInt16(S+1:2*S);
        case 3 % BlockInterleaved: [I0 I1 Q0 Q1 I2 I3 Q2 Q3 ...]
            if mod(S,2) ~= 0
                I = zeros(S,1,'int16'); Q = zeros(S,1,'int16');
                ii = 1; qi = 1; k = 1;
                while k <= 2*S
                    take = min(2, S - (ii-1));
                    if take>0, I(ii:ii+take-1) = segInt16(k:k+take-1); k = k+take; ii = ii+take; end
                    take = min(2, S - (qi-1));
                    if take>0, Q(qi:qi+take-1) = segInt16(k:k+take-1); k = k+take; qi = qi+take; end
                end
            else
                g = reshape(segInt16, 4, []);
                I = reshape(g(1:2, :), [], 1);
                Q = reshape(g(3:4, :), [], 1);
            end
        otherwise
            error('Unsupported iq_order value: %d', iqOrder);
    end

    if opts.returnComplex
        Iflt = cast(I, mapFloat(opts.cast));
        Qflt = cast(Q, mapFloat(opts.cast));
        z = complex(Iflt, Qflt);
    else
        z = complex(cast(I, mapFloat(opts.cast)), cast(Q, mapFloat(opts.cast)));
    end
end

function t = mapOutClass(sample_type, castOpt, returnComplex)
    if sample_type == 0
        switch lower(castOpt)
            case 'double', t = 'double';
            case 'single', t = 'single';
            case 'int16',  t = 'int16';
            otherwise,     t = 'single';
        end
    else
        if returnComplex
            switch lower(castOpt)
                case 'double', t = 'double';
                otherwise,     t = 'single';
            end
        else
            t = 'int16';
        end
    end
end

function t = mapFloat(castOpt)
    switch lower(castOpt)
        case 'double', t = 'double';
        otherwise,     t = 'single';
    end
end

function s = setDefault(s, field, val)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = val;
    end
end

function printHeaderSummary(FH, BH)
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

function y = tern(cond, a, b)
    if cond, y = a; else, y = b; end
end
