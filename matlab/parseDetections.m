function [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts) %#ok<INUSD>
% PARSEDETECTIONS Parse Dopplium Detections data (message_type=1) - NOT YET IMPLEMENTED
%   [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts)
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

    error('parseDetections:NotImplemented', ...
          'Detections format (message_type=1) is not yet implemented.');
end
