function FH = readFileHeader(fid, machinefmt)
% READFILEHEADER Read Dopplium file header - Versions 2 & 3 format (80 bytes minimum)
%   FH = readFileHeader(fid, machinefmt)
%
%   INPUTS
%     fid        : file identifier from fopen
%     machinefmt : 'ieee-le' or 'ieee-be'
%
%   OUTPUTS
%     FH : struct containing file header fields
%          .magic, .version, .endianness, .compression, .product_id,
%          .message_type, .file_header_size, .body_header_size,
%          .frame_header_size, .file_created_utc_ticks,
%          .last_written_utc_ticks, .total_frames_written,
%          .total_payload_bytes, .reserved1, .node_id

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
