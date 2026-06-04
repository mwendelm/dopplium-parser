function [data, headers] = doppliumParser(filename, opts)
% DOPPLIUMPARSER Parse Dopplium data files
%   [data, headers] = doppliumParser(filename, opts)
%
%   Main entry point that reads the file header, detects version and
%   message type, then dispatches to the appropriate parser.
%
%   Supported message types:
%   Canonical mapping (spec v3/v4/v5):
%     0 - Unknown (unsupported)
%     1 - ADCData (currently supported)
%     2 - RDCMaps (currently supported)
%     3 - RadarCube (currently supported)
%     4 - Detections (currently supported)
%     5 - Blobs (currently supported)
%     6 - Tracks (currently supported)
%
%   OUTPUTS
%     data    : parsed data (format depends on message type)
%     headers : struct with parsed file/body/frame headers and derived info
%
%   OPTS (all optional)
%     .maxFrames      : limit number of frames to read (default: Inf = all)
%     .startFrame     : zero-based frame/CPI index to start from (default: 0)
%                       Supported for ADCData, RDCMaps, and RadarCube only.
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
startFrameRequested = (isfield(opts, 'startFrame') && ~isempty(opts.startFrame)) || ...
                      (isfield(opts, 'start_frame') && ~isempty(opts.start_frame));
if isfield(opts, 'start_frame') && ~isfield(opts, 'startFrame')
    opts.startFrame = opts.start_frame;
end
opts = setDefault(opts, 'maxFrames', Inf);
opts = setDefault(opts, 'startFrame', 0);
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

% Read file header
fseek(fid, 0, 'bof');
if version == 2 || version == 3 || version == 4 || version == 5
    FH = readFileHeader(fid, machinefmt);
else
    error('Unsupported file version: %d', version);
end

% Validate file header
assert(FH.file_header_size >= 80, 'Unexpected file_header_size.');

% -------------------------------------------------------------------------
% Dispatch based on message_type.
% Version 2 legacy compatibility:
%   older writers used message_type=3 for ADC/raw data.
% -------------------------------------------------------------------------
messageType = double(FH.message_type);
if version == 2 && messageType == 3
    if opts.verbose
        warning(['Legacy version-2 file uses message_type=3 for ADC/raw data. ' ...
                 'Treating it as canonical ADCData (type=1).']);
    end
    messageType = 1;
end

switch messageType
    case 0
        error('File has unknown message_type (0). Cannot parse.');
    case 1 % ADCData
        [data, headers] = parseADCData(fid, FH, machinefmt, filename, opts);
    case 2 % RDCMaps
        if version == 2
            error('Version 2 RDCMaps (message_type=2) not implemented in this parser.');
        end
        [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts);
    case 3 % RadarCube
        if version == 2
            error('Version 2 message_type=3 is only supported as legacy ADC mapping.');
        end
        [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts);
    case 4 % Detections
        if startFrameRequested
            error('startFrame is only supported for ADCData, RDCMaps, and RadarCube files.');
        end
        [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts);
    case 5 % Blobs
        if startFrameRequested
            error('startFrame is only supported for ADCData, RDCMaps, and RadarCube files.');
        end
        [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts);
    case 6 % Tracks
        if startFrameRequested
            error('startFrame is only supported for ADCData, RDCMaps, and RadarCube files.');
        end
        [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts);
    otherwise
        error('Unsupported message_type=%d for version=%d.', FH.message_type, version);
end

end
