function [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts) %#ok<INUSD>
% PARSERADARCUBE Parse Dopplium Radar Cube data (v3 message_type=3) - NOT YET IMPLEMENTED
%   [data, headers] = parseRadarCube(fid, FH, machinefmt, filename, opts)
%
%   Radar Cube with all dimensions estimated (Version 3 format)
%
%   INPUTS
%     fid        : file identifier (positioned after file header)
%     FH         : file header struct (already parsed)
%     machinefmt : endianness ('ieee-le' or 'ieee-be')
%     filename   : file path (for size calculation)
%     opts       : options struct
%
%   OUTPUTS
%     data    : (not implemented)
%     headers : (not implemented)

    error('parseRadarCube:NotImplemented', ...
          'Radar Cube format (v3 message_type=3) is not yet implemented.');
end
