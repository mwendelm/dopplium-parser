function [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts)
% PARSERDCMAPS Parse Dopplium RDC Maps data (v3/v4/v5 message_type=2)
%   [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts)
%
%   Returns:
%     data    : array shaped [range_bins, doppler_bins, channels, cpis]
%     headers : struct with fields .file, .body, .cpi
%
%   opts.startFrame is a zero-based CPI index. opts.maxFrames/opts.maxCPIs
%   limit the number of CPIs read after that start index.

% Version/message validation (match Python behavior)
if ~(FH.version == 3 || FH.version == 4 || FH.version == 5) || FH.message_type ~= 2
    error('parseRDCMaps:InvalidMessageType', ...
        ['This file is not RDCMaps/RDCh (version=%d, message_type=%d, ' ...
         'expected version=3/4/5, type=2).'], ...
        FH.version, FH.message_type);
end

if nargin < 6 || isempty(opts)
    opts = struct;
end
if isfield(opts, 'start_frame') && ~isfield(opts, 'startFrame')
    opts.startFrame = opts.start_frame;
end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'maxFrames'), opts.maxFrames = Inf; end
if ~isfield(opts, 'startFrame'), opts.startFrame = 0; end
maxCpis = opts.maxFrames;
if isfield(opts, 'maxCPIs')
    maxCpis = opts.maxCPIs;
end
maxCpis = coerceOptionalNonnegativeInt(maxCpis, 'maxCPIs', Inf, true);
startFrame = coerceOptionalNonnegativeInt(opts.startFrame, 'startFrame', 0, false);

% Read body header
fseek(fid, FH.file_header_size, 'bof');
BH = readRDCHBodyHeader(fid, machinefmt);

if opts.verbose
    printHeaderSummary(FH, BH);
end

nRange = double(BH.n_range_bins);
nDopp = double(BH.n_doppler_bins);
nChan = double(BH.n_channels);

[payloadClass, isComplex, bytesPerElement] = mapDataType(BH.data_type);
expectedPayloadBytes = nRange * nDopp * nChan * bytesPerElement;

fileInfo = dir(filename);
bytesAfterHeaders = double(fileInfo.bytes) - double(FH.file_header_size) - double(BH.body_header_size);
cpiHeaderBytes = max(22, double(BH.cpi_header_size));
cpiUnit = cpiHeaderBytes + expectedPayloadBytes;
if cpiUnit <= 0
    nCpisEstimate = 0;
else
    nCpisEstimate = max(0, floor(bytesAfterHeaders / cpiUnit));
end
nCpisAvailable = max(0, nCpisEstimate - startFrame);
nCpis = min(nCpisAvailable, maxCpis);

if opts.verbose
    fprintf('\nExpected payload size per CPI: %d bytes\n', expectedPayloadBytes);
    fprintf('Bytes after headers: %d\n', bytesAfterHeaders);
    fprintf('Estimated CPIs in file: %d\n', nCpisEstimate);
end

% Allocate output
if isComplex
    data = complex(zeros(nRange, nDopp, nChan, nCpis, payloadClass), ...
                   zeros(nRange, nDopp, nChan, nCpis, payloadClass));
else
    data = zeros(nRange, nDopp, nChan, nCpis, payloadClass);
end

% Read CPIs
fseek(fid, FH.file_header_size + BH.body_header_size + startFrame * cpiUnit, 'bof');
cpiHeaders = repmat(emptyCPIHeader(), 0, 1);
cpisRead = 0;

while cpisRead < nCpis
    cpiIndex = startFrame + cpisRead;
    cpiOrdinal = cpiIndex + 1;
    if ftell(fid) >= fileInfo.bytes
        break;
    end

    try
        CH = readCPIHeader(fid, machinefmt);
        cpiHeaders(end+1,1) = CH; %#ok<AGROW>

        if opts.verbose && (cpisRead == 0 || mod(cpisRead + 1, 10) == 0)
            fprintf('  Reading CPI %d/%d: number=%d, size=%d bytes\n', ...
                cpisRead + 1, nCpis, CH.cpi_number, CH.cpi_payload_size);
        end

        if double(CH.cpi_payload_size) ~= expectedPayloadBytes
            if opts.verbose
                warning('parseRDCMaps:PayloadSizeMismatch', ...
                    'CPI %d payload size mismatch: expected=%d, got=%d', ...
                    cpiIndex, expectedPayloadBytes, CH.cpi_payload_size);
            end
        end

        if mod(double(CH.cpi_payload_size), bytesPerElement) ~= 0
            error('parseRDCMaps:InvalidPayloadSize', ...
                'CPI %d payload bytes (%d) not divisible by bytesPerElement (%d).', ...
                cpiOrdinal, CH.cpi_payload_size, bytesPerElement);
        end

        % Honor cpi_header_size for forward compatibility.
        extraHeaderBytes = double(CH.cpi_header_size) - 22;
        if extraHeaderBytes < 0
            error('parseRDCMaps:InvalidHeaderSize', ...
                'CPI %d has invalid cpi_header_size=%d (<22).', cpiOrdinal, CH.cpi_header_size);
        elseif extraHeaderBytes > 0
            fseek(fid, extraHeaderBytes, 'cof');
        end

        nElements = double(CH.cpi_payload_size) / bytesPerElement;
        payload = readTypedPayload(fid, nElements, payloadClass, isComplex, machinefmt);

        try
            cpiData = reshape(payload, nRange, nDopp, nChan);
        catch ME
            error('parseRDCMaps:ReshapeError', ...
                ['CPI %d: Cannot reshape payload. Expected %dx%dx%d=%d elements, got %d. ' ...
                 'Underlying error: %s'], ...
                 cpiIndex, nRange, nDopp, nChan, nRange*nDopp*nChan, numel(payload), ME.message);
        end

        cpisRead = cpisRead + 1;
        data(:,:,:,cpisRead) = cpiData;
    catch ME
        if strcmp(ME.identifier, 'parseRDCMaps:UnexpectedEOF')
            if opts.verbose
                fprintf('Reached end of file after reading %d CPIs.\n', cpisRead);
            end
            break;
        end
        rethrow(ME);
    end
end

if cpisRead < nCpis
    data = data(:,:,:,1:cpisRead);
end

headers.file = FH;
headers.body = BH;
headers.cpi  = cpiHeaders;

if opts.verbose
    fprintf('\nParsed data shape: [%d, %d, %d, %d] [range_bins, doppler_bins, channels, cpis]\n', ...
        size(data,1), size(data,2), size(data,3), size(data,4));
    fprintf('Total CPIs read: %d\n', cpisRead);
end
end

function BH = readRDCHBodyHeader(fid, machinefmt)
BH.config_magic                     = char(fread(fid, [1,4], '*char'));
BH.config_version                   = fread(fid, 1, 'uint16', 0, machinefmt);
BH.body_header_size                 = fread(fid, 1, 'uint16', 0, machinefmt);
BH.cpi_header_size                  = fread(fid, 1, 'uint16', 0, machinefmt);
BH.reserved1                        = fread(fid, 1, 'uint16', 0, machinefmt);
BH.n_range_bins                     = fread(fid, 1, 'uint32', 0, machinefmt);
BH.n_doppler_bins                   = fread(fid, 1, 'uint32', 0, machinefmt);
BH.n_channels                       = fread(fid, 1, 'uint32', 0, machinefmt);
BH.range_min_m                      = fread(fid, 1, 'single', 0, machinefmt);
BH.range_max_m                      = fread(fid, 1, 'single', 0, machinefmt);
BH.range_resolution_m               = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_min_mps                 = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_max_mps                 = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_resolution_mps          = fread(fid, 1, 'single', 0, machinefmt);
BH.physical_velocity_resolution_mps = fread(fid, 1, 'single', 0, machinefmt);
BH.data_type                        = fread(fid, 1, 'uint8',  0, machinefmt);
BH.nfft_range                       = fread(fid, 1, 'uint32', 0, machinefmt);
BH.nfft_doppler                     = fread(fid, 1, 'uint32', 0, machinefmt);
BH.range_window_type                = fread(fid, 1, 'uint8',  0, machinefmt);
BH.doppler_window_type              = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_range                   = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_doppler                 = fread(fid, 1, 'uint8',  0, machinefmt);
BH.range_half_spectrum              = fread(fid, 1, 'uint8',  0, machinefmt);
BH.coherent_integration_time_ms     = fread(fid, 1, 'single', 0, machinefmt);
BH.channel_order                    = fread(fid, 30, '*int8', 0, machinefmt);
BH.physical_range_resolution_m      = fread(fid, 1, 'single', 0, machinefmt);
BH.is_db_scale                      = fread(fid, 1, 'uint8',  0, machinefmt);
BH.data_format                      = fread(fid, 1, 'uint8',  0, machinefmt);
BH.reserved3                        = fread(fid, 150, '*uint8', 0, machinefmt);
BH.integration_time_ms              = BH.coherent_integration_time_ms; % backward-compatible alias

if ~strcmp(BH.config_magic, 'RDCH')
    warning('parseRDCMaps:MagicMismatch', ...
        'Unexpected RDCh body magic "%s" (expected "RDCH").', BH.config_magic);
end
end

function CH = readCPIHeader(fid, machinefmt)
CH.cpi_magic               = char(fread(fid, [1,4], '*char'));
CH.cpi_header_size         = fread(fid, 1, 'uint16', 0, machinefmt);
CH.cpi_timestamp_utc_ticks = fread(fid, 1, 'int64',  0, machinefmt);
CH.cpi_number              = fread(fid, 1, 'uint32', 0, machinefmt);
CH.cpi_payload_size        = fread(fid, 1, 'uint32', 0, machinefmt);

if isempty(CH.cpi_payload_size)
    error('parseRDCMaps:UnexpectedEOF', 'Failed to read CPI header.');
end
if ~strcmp(CH.cpi_magic, 'CPII')
    error('parseRDCMaps:InvalidCPIMagic', ...
        'Invalid CPI magic "%s" (expected "CPII").', CH.cpi_magic);
end
end

function payload = readTypedPayload(fid, nElements, payloadClass, isComplex, machinefmt)
if isComplex
    raw = fread(fid, 2*nElements, ['*' payloadClass], 0, machinefmt);
    if numel(raw) ~= 2*nElements
        error('parseRDCMaps:UnexpectedEOF', 'Unexpected EOF while reading complex payload.');
    end
    payload = complex(raw(1:2:end), raw(2:2:end));
else
    payload = fread(fid, nElements, ['*' payloadClass], 0, machinefmt);
    if numel(payload) ~= nElements
        error('parseRDCMaps:UnexpectedEOF', 'Unexpected EOF while reading payload.');
    end
end
end

function [payloadClass, isComplex, bytesPerElement] = mapDataType(dataType)
switch double(dataType)
    case 0
        payloadClass = 'single'; isComplex = true;  bytesPerElement = 8;   % complex64
    case 1
        payloadClass = 'double'; isComplex = true;  bytesPerElement = 16;  % complex128
    case 2
        payloadClass = 'single'; isComplex = false; bytesPerElement = 4;   % float32
    case 3
        payloadClass = 'double'; isComplex = false; bytesPerElement = 8;   % float64
    case 4
        payloadClass = 'int16';  isComplex = false; bytesPerElement = 2;   % int16
    case 5
        payloadClass = 'int32';  isComplex = false; bytesPerElement = 4;   % int32
    otherwise
        warning('parseRDCMaps:UnknownDataType', ...
            'Unknown data_type=%d. Falling back to complex64.', dataType);
        payloadClass = 'single'; isComplex = true;  bytesPerElement = 8;
end
end

function s = emptyCPIHeader()
s = struct( ...
    'cpi_magic',               '', ...
    'cpi_header_size',         uint16(0), ...
    'cpi_timestamp_utc_ticks', int64(0), ...
    'cpi_number',              uint32(0), ...
    'cpi_payload_size',        uint32(0));
end

function printHeaderSummary(FH, BH)
fprintf('--- Dopplium RDCh Data ---\n');
fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
    FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
fprintf('FileHdr=%d  BodyHdr=%d  CPIHdr=%d  TotalCPIsWritten=%d\n', ...
    FH.file_header_size, BH.body_header_size, BH.cpi_header_size, FH.total_frames_written);
fprintf('NodeId="%s"\n', FH.node_id);

fprintf('\n-- RDCh Configuration --\n');
fprintf('Dimensions: Range=%d, Doppler=%d, Channels=%d\n', ...
    BH.n_range_bins, BH.n_doppler_bins, BH.n_channels);
fprintf('DataType=%d  DataFormat=%d  Scale=%s\n', ...
    BH.data_type, BH.data_format, tern(BH.is_db_scale~=0,'dB','linear'));
fprintf('Range: [%.2f, %.2f] m  FFTRes=%.4f m  PhysicalRes=%.4f m\n', ...
    BH.range_min_m, BH.range_max_m, BH.range_resolution_m, BH.physical_range_resolution_m);
fprintf('Velocity: [%.2f, %.2f] m/s  FFTRes=%.4f m/s  PhysicalRes=%.4f m/s\n', ...
    BH.velocity_min_mps, BH.velocity_max_mps, BH.velocity_resolution_mps, BH.physical_velocity_resolution_mps);
fprintf('FFT sizes: range=%d doppler=%d  fftshift=[%d %d]  half-spectrum=%d  integration=%.3f ms\n', ...
    BH.nfft_range, BH.nfft_doppler, BH.fftshift_range, BH.fftshift_doppler, ...
    BH.range_half_spectrum, BH.coherent_integration_time_ms);
end
