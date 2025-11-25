function [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts) %#ok<INUSD>
% PARSEBLOBS Parse Dopplium Blobs data (v3 message_type=5) - NOT YET IMPLEMENTED
%   [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts)
%
%   Clustered detection data (Version 3 format)
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

    error('parseBlobs:NotImplemented', ...
          'Blobs format (v3 message_type=5) is not yet implemented.');
end
