function [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts) %#ok<INUSD>
% PARSERDCMAPS Parse Dopplium RDC Maps data (v3 message_type=2) - NOT YET IMPLEMENTED
%   [data, headers] = parseRDCMaps(fid, FH, machinefmt, filename, opts)
%
%   Range Doppler Channel maps (Version 3 format)
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

    error('parseRDCMaps:NotImplemented', ...
          'RDC Maps format (v3 message_type=2) is not yet implemented.');
end
