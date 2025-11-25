function [data, headers] = doppliumParser(filename, opts)
% DOPPLIUMPARSER Parse Dopplium data files
%   [data, headers] = doppliumParser(filename, opts)
%
%   Main entry point that reads the file header, detects version and
%   message type, then dispatches to the appropriate parser.
%
%   Supported message types:
%   Version 2:
%     0 - Unknown (unsupported)
%     1 - Detections (not yet implemented)
%     2 - Tracks (not yet implemented)
%     3 - RawData/ADC (currently supported)
%     4 - Aggregated (not yet implemented)
%   Version 3:
%     0 - Unknown (unsupported)
%     1 - ADCData (currently supported)
%     2 - RDCMaps (not yet implemented)
%     3 - RadarCube (not yet implemented)
%     4 - Detections (not yet implemented)
%     5 - Blobs (not yet implemented)
%     6 - Tracks (not yet implemented)
%
%   OUTPUTS
%     data    : parsed data (format depends on message type)
%     headers : struct with parsed file/body/frame headers and derived info
%
%   OPTS (all optional)
%     .maxFrames      : limit number of frames to read (default: Inf = all)
%     .cast           : 'double'|'single'|'int16' for output samples (default 'single')
%     .returnComplex  : true/false, if complex IQ => return complex numbers (default true)
%     .verbose        : true/false (default true)

% Add utils directory to path if not already there
utilsPath = fullfile(fileparts(mfilename('fullpath')), 'utils');
if ~contains(path, utilsPath)
    addpath(utilsPath);
end

% Set default options
if nargin < 2, opts = struct; end
opts = setDefault(opts, 'maxFrames', Inf);
opts = setDefault(opts, 'cast', 'single');
opts = setDefault(opts, 'returnComplex', true);
opts = setDefault(opts, 'verbose', true);

% -------------------------------------------------------------------------
% Open file & detect endianness
% -------------------------------------------------------------------------
[fid, msg] = fopen(filename, 'r', 'ieee-le');
assert(fid > 0, ['Failed to open file: ' msg]);
cleanup = onCleanup(@() fclose(fid));

% Read magic bytes
magic = fread(fid, [1,4], '*char');
assert(strcmp(magic, 'DOPP'), 'Invalid magic. Not a Dopplium file.');

% Detect endianness
fseek(fid, 6, 'bof');
endianness = fread(fid, 1, 'uint8');
machinefmt = 'ieee-le';
if endianness == 0
    machinefmt = 'ieee-be';
elseif endianness ~= 1
    warning('Unknown endianness value %d. Assuming little-endian.', endianness);
end

% -------------------------------------------------------------------------
% Read version to determine file header parser
% -------------------------------------------------------------------------
fseek(fid, 4, 'bof');
version = fread(fid, 1, 'uint16', 0, machinefmt);

% Read file header using version-specific parser
fseek(fid, 0, 'bof');
switch version
    case 2
        FH = readFileHeader(fid, machinefmt);
    case 3
        FH = readFileHeader(fid, machinefmt);  % V3 has same format as V2
    otherwise
        error('Unsupported file version: %d', version);
end

% Validate file header
assert(FH.file_header_size >= 80, 'Unexpected file_header_size.');

% -------------------------------------------------------------------------
% Dispatch based on version and message_type
% -------------------------------------------------------------------------
switch version
    case 2
        % Version 2 message type mappings
        switch FH.message_type
            case 0
                error('File has unknown message_type (0). Cannot parse.');
            case 1 % Detections
                [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts);
            case 2 % Tracks
                [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts);
            case 3 % RawData/ADC
                [data, headers] = parseADCData(fid, FH, machinefmt, filename, opts);
            case 4 % Aggregated
                [data, headers] = parseAggregated(fid, FH, machinefmt, filename, opts);
            otherwise
                error('Unsupported Version 2 message_type: %d', FH.message_type);
        end

    case 3
        % Version 3 message type mappings
        switch FH.message_type
            case 0
                error('File has unknown message_type (0). Cannot parse.');
            case 1 % ADCData
                [data, headers] = parseADCData(fid, FH, machinefmt, filename, opts);
            case 2 % RDCMaps
                [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts);
            case 3 % RadarCube
                [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts);
            case 4 % Detections
                [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts);
            case 5 % Blobs
                [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts);
            case 6 % Tracks
                [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts);
            otherwise
                error('Unsupported Version 3 message_type: %d', FH.message_type);
        end

    otherwise
        error('Unsupported version: %d', version);
end

end
