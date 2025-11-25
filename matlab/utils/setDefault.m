function s = setDefault(s, field, val)
% SETDEFAULT Set default value for struct field if not present or empty
%   s = setDefault(s, field, val)
%
%   INPUTS
%     s     : struct
%     field : field name (string)
%     val   : default value
%
%   OUTPUTS
%     s : struct with field set to val if it was missing or empty

    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = val;
    end
end
