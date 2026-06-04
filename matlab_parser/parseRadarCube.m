function [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts)
% PARSERADARCUBE Parse Dopplium RadarCube data (v3/v4/v5 message_type=3)
%   [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts)
%
%   Returns:
%     data    : array shaped [range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]
%     headers : struct with fields .file, .body, .cpi
%
%   opts.startFrame is a zero-based CPI index. opts.maxFrames/opts.maxCPIs
%   limit the number of CPIs read after that start index.

if ~(FH.version == 3 || FH.version == 4 || FH.version == 5) || FH.message_type ~= 3
    error('parseRadarCube:InvalidMessageType', ...
        ['This file is not RadarCube (version=%d, message_type=%d, ' ...
         'expected version=3/4/5, type=3).'], ...
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
BH = readRadarCubeBodyHeader(fid, machinefmt);

if opts.verbose
    printHeaderSummary(FH, BH);
end

nRange = double(BH.n_range_bins);
nDopp  = double(BH.n_doppler_bins);
nAz    = double(BH.n_azimuth_bins);
nEl    = double(BH.n_elevation_bins);

[payloadClass, isComplex, bytesPerElement] = mapDataType(BH.data_type);
expectedPayloadBytes = nRange * nDopp * nAz * nEl * bytesPerElement;

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
    data = complex(zeros(nRange, nDopp, nAz, nEl, nCpis, payloadClass), ...
                   zeros(nRange, nDopp, nAz, nEl, nCpis, payloadClass));
else
    data = zeros(nRange, nDopp, nAz, nEl, nCpis, payloadClass);
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
                warning('parseRadarCube:PayloadSizeMismatch', ...
                    'CPI %d payload size mismatch: expected=%d, got=%d', ...
                    cpiIndex, expectedPayloadBytes, CH.cpi_payload_size);
            end
        end

        if mod(double(CH.cpi_payload_size), bytesPerElement) ~= 0
            error('parseRadarCube:InvalidPayloadSize', ...
                'CPI %d payload bytes (%d) not divisible by bytesPerElement (%d).', ...
                cpiOrdinal, CH.cpi_payload_size, bytesPerElement);
        end

        % Honor cpi_header_size for forward compatibility.
        extraHeaderBytes = double(CH.cpi_header_size) - 22;
        if extraHeaderBytes < 0
            error('parseRadarCube:InvalidHeaderSize', ...
                'CPI %d has invalid cpi_header_size=%d (<22).', cpiOrdinal, CH.cpi_header_size);
        elseif extraHeaderBytes > 0
            fseek(fid, extraHeaderBytes, 'cof');
        end

        nElements = double(CH.cpi_payload_size) / bytesPerElement;
        payload = readTypedPayload(fid, nElements, payloadClass, isComplex, machinefmt);

        try
            cpiData = reshape(payload, nRange, nDopp, nAz, nEl);
        catch ME
            error('parseRadarCube:ReshapeError', ...
                ['CPI %d: Cannot reshape payload. Expected %dx%dx%dx%d=%d elements, got %d. ' ...
                 'Underlying error: %s'], ...
                 cpiIndex, nRange, nDopp, nAz, nEl, nRange*nDopp*nAz*nEl, numel(payload), ME.message);
        end

        cpisRead = cpisRead + 1;
        data(:,:,:,:,cpisRead) = cpiData;
    catch ME
        if strcmp(ME.identifier, 'parseRadarCube:UnexpectedEOF')
            if opts.verbose
                fprintf('Reached end of file after reading %d CPIs.\n', cpisRead);
            end
            break;
        end
        rethrow(ME);
    end
end

if cpisRead < nCpis
    data = data(:,:,:,:,1:cpisRead);
end

headers.file = FH;
headers.body = BH;
headers.cpi  = cpiHeaders;

if opts.verbose
    fprintf('\nParsed data shape: [%d, %d, %d, %d, %d] [range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]\n', ...
        size(data,1), size(data,2), size(data,3), size(data,4), size(data,5));
    fprintf('Total CPIs read: %d\n', cpisRead);
end
end

function BH = readRadarCubeBodyHeader(fid, machinefmt)
BH.config_magic                     = char(fread(fid, [1,4], '*char'));
BH.config_version                   = fread(fid, 1, 'uint16', 0, machinefmt);
BH.body_header_size                 = fread(fid, 1, 'uint16', 0, machinefmt);
BH.cpi_header_size                  = fread(fid, 1, 'uint16', 0, machinefmt);
BH.reserved1                        = fread(fid, 1, 'uint16', 0, machinefmt);
BH.n_range_bins                     = fread(fid, 1, 'uint32', 0, machinefmt);
BH.n_doppler_bins                   = fread(fid, 1, 'uint32', 0, machinefmt);
BH.n_azimuth_bins                   = fread(fid, 1, 'uint32', 0, machinefmt);
BH.n_elevation_bins                 = fread(fid, 1, 'uint32', 0, machinefmt);
BH.range_min_m                      = fread(fid, 1, 'single', 0, machinefmt);
BH.range_max_m                      = fread(fid, 1, 'single', 0, machinefmt);
BH.range_resolution_m               = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_min_mps                 = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_max_mps                 = fread(fid, 1, 'single', 0, machinefmt);
BH.velocity_resolution_mps          = fread(fid, 1, 'single', 0, machinefmt);
BH.physical_velocity_resolution_mps = fread(fid, 1, 'single', 0, machinefmt);
BH.azimuth_min_deg                  = fread(fid, 1, 'single', 0, machinefmt);
BH.azimuth_max_deg                  = fread(fid, 1, 'single', 0, machinefmt);
BH.azimuth_resolution_deg           = fread(fid, 1, 'single', 0, machinefmt);
BH.elevation_min_deg                = fread(fid, 1, 'single', 0, machinefmt);
BH.elevation_max_deg                = fread(fid, 1, 'single', 0, machinefmt);
BH.elevation_resolution_deg         = fread(fid, 1, 'single', 0, machinefmt);
BH.data_type                        = fread(fid, 1, 'uint8',  0, machinefmt);
BH.nfft_range                       = fread(fid, 1, 'uint32', 0, machinefmt);
BH.nfft_doppler                     = fread(fid, 1, 'uint32', 0, machinefmt);
BH.nfft_azimuth                     = fread(fid, 1, 'uint32', 0, machinefmt);
BH.nfft_elevation                   = fread(fid, 1, 'uint32', 0, machinefmt);
BH.range_window_type                = fread(fid, 1, 'uint8',  0, machinefmt);
BH.doppler_window_type              = fread(fid, 1, 'uint8',  0, machinefmt);
BH.azimuth_window_type              = fread(fid, 1, 'uint8',  0, machinefmt);
BH.elevation_window_type            = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_range                   = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_doppler                 = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_azimuth                 = fread(fid, 1, 'uint8',  0, machinefmt);
BH.fftshift_elevation               = fread(fid, 1, 'uint8',  0, machinefmt);
BH.range_half_spectrum              = fread(fid, 1, 'uint8',  0, machinefmt);
BH.coherent_integration_time_ms     = fread(fid, 1, 'single', 0, machinefmt);
BH.angle_estimation_algorithm       = fread(fid, 1, 'uint8',  0, machinefmt);
BH.physical_range_resolution_m      = fread(fid, 1, 'single', 0, machinefmt);
BH.is_db_scale                      = fread(fid, 1, 'uint8',  0, machinefmt);
BH.data_format                      = fread(fid, 1, 'uint8',  0, machinefmt);
BH.cpis_incoherently_integrated     = fread(fid, 1, 'uint16', 0, machinefmt);
BH.reserved3                        = fread(fid, 137, '*uint8', 0, machinefmt);
BH.integration_time_ms              = BH.coherent_integration_time_ms; % backward-compatible alias

if ~strcmp(BH.config_magic, 'RCUB')
    warning('parseRadarCube:MagicMismatch', ...
        'Unexpected RadarCube body magic "%s" (expected "RCUB").', BH.config_magic);
end
end

function CH = readCPIHeader(fid, machinefmt)
CH.cpi_magic               = char(fread(fid, [1,4], '*char'));
CH.cpi_header_size         = fread(fid, 1, 'uint16', 0, machinefmt);
CH.cpi_timestamp_utc_ticks = fread(fid, 1, 'int64',  0, machinefmt);
CH.cpi_number              = fread(fid, 1, 'uint32', 0, machinefmt);
CH.cpi_payload_size        = fread(fid, 1, 'uint32', 0, machinefmt);

if isempty(CH.cpi_payload_size)
    error('parseRadarCube:UnexpectedEOF', 'Failed to read CPI header.');
end
if ~strcmp(CH.cpi_magic, 'CPII')
    error('parseRadarCube:InvalidCPIMagic', ...
        'Invalid CPI magic "%s" (expected "CPII").', CH.cpi_magic);
end
end

function payload = readTypedPayload(fid, nElements, payloadClass, isComplex, machinefmt)
if isComplex
    raw = fread(fid, 2*nElements, ['*' payloadClass], 0, machinefmt);
    if numel(raw) ~= 2*nElements
        error('parseRadarCube:UnexpectedEOF', 'Unexpected EOF while reading complex payload.');
    end
    payload = complex(raw(1:2:end), raw(2:2:end));
else
    payload = fread(fid, nElements, ['*' payloadClass], 0, machinefmt);
    if numel(payload) ~= nElements
        error('parseRadarCube:UnexpectedEOF', 'Unexpected EOF while reading payload.');
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
        warning('parseRadarCube:UnknownDataType', ...
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
fprintf('--- Dopplium RadarCube Data ---\n');
fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
    FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
fprintf('FileHdr=%d  BodyHdr=%d  CPIHdr=%d  TotalCPIsWritten=%d\n', ...
    FH.file_header_size, BH.body_header_size, BH.cpi_header_size, FH.total_frames_written);
fprintf('NodeId="%s"\n', FH.node_id);

fprintf('\n-- RadarCube Configuration --\n');
fprintf('Dimensions: Range=%d, Doppler=%d, Azimuth=%d, Elevation=%d\n', ...
    BH.n_range_bins, BH.n_doppler_bins, BH.n_azimuth_bins, BH.n_elevation_bins);
fprintf('DataType=%d  DataFormat=%d  Scale=%s  AngleAlgo=%d\n', ...
    BH.data_type, BH.data_format, tern(BH.is_db_scale~=0,'dB','linear'), BH.angle_estimation_algorithm);
fprintf('Range: [%.2f, %.2f] m  FFTRes=%.4f m  PhysicalRes=%.4f m\n', ...
    BH.range_min_m, BH.range_max_m, BH.range_resolution_m, BH.physical_range_resolution_m);
fprintf('Velocity: [%.2f, %.2f] m/s  FFTRes=%.4f m/s  PhysicalRes=%.4f m/s\n', ...
    BH.velocity_min_mps, BH.velocity_max_mps, BH.velocity_resolution_mps, BH.physical_velocity_resolution_mps);
fprintf('Azimuth: [%.2f, %.2f] deg  Res=%.4f deg\n', ...
    BH.azimuth_min_deg, BH.azimuth_max_deg, BH.azimuth_resolution_deg);
fprintf('Elevation: [%.2f, %.2f] deg  Res=%.4f deg\n', ...
    BH.elevation_min_deg, BH.elevation_max_deg, BH.elevation_resolution_deg);
fprintf(['FFT sizes: range=%d doppler=%d az=%d el=%d  fftshift=[%d %d %d %d] ' ...
         'half-spectrum=%d  coherent-integration=%.3f ms  incoherent-cpis=%d\n'], ...
    BH.nfft_range, BH.nfft_doppler, BH.nfft_azimuth, BH.nfft_elevation, ...
    BH.fftshift_range, BH.fftshift_doppler, BH.fftshift_azimuth, BH.fftshift_elevation, ...
    BH.range_half_spectrum, BH.coherent_integration_time_ms, BH.cpis_incoherently_integrated);
end
