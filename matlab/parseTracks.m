function [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts) %#ok<INUSD>
% PARSETRACKS Parse Dopplium Tracks data (message_type=2) - NOT YET IMPLEMENTED
%   [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts)
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

    error('parseTracks:NotImplemented', ...
          'Tracks format (message_type=2) is not yet implemented.');
end
