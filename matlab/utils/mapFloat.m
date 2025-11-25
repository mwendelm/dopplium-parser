function t = mapFloat(castOpt)
% MAPFLOAT Map cast option to floating point type
%   t = mapFloat(castOpt)
%
%   INPUTS
%     castOpt : 'double'|'single'|...
%
%   OUTPUTS
%     t : 'double' or 'single' (default)

    switch lower(castOpt)
        case 'double', t = 'double';
        otherwise,     t = 'single';
    end
end
