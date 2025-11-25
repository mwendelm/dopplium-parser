function t = mapOutClass(sample_type, castOpt, returnComplex)
% MAPOUTCLASS Determine output data type based on sample type and cast option
%   t = mapOutClass(sample_type, castOpt, returnComplex)
%
%   INPUTS
%     sample_type    : 0=real, 1=complex
%     castOpt        : 'double'|'single'|'int16'
%     returnComplex  : true/false
%
%   OUTPUTS
%     t : output class string ('double', 'single', or 'int16')

    if sample_type == 0
        switch lower(castOpt)
            case 'double', t = 'double';
            case 'single', t = 'single';
            case 'int16',  t = 'int16';
            otherwise,     t = 'single';
        end
    else
        if returnComplex
            switch lower(castOpt)
                case 'double', t = 'double';
                otherwise,     t = 'single';
            end
        else
            t = 'int16';
        end
    end
end
