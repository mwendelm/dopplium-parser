function value = coerceOptionalNonnegativeInt(value, name, defaultValue, allowInf)
% COERCEOPTIONALNONNEGATIVEINT Convert optional nonnegative integer options.

if nargin < 2 || isempty(name)
    name = 'value';
end
if nargin < 3
    defaultValue = [];
end
if nargin < 4
    allowInf = false;
end

if isempty(value)
    value = defaultValue;
    return;
end
if ischar(value)
    value = str2double(value);
end
if ~(isnumeric(value) || islogical(value)) || ~isscalar(value)
    error('%s must be a nonnegative integer scalar.', name);
end

numericValue = double(value);
if isnan(numericValue)
    error('%s must be a nonnegative integer scalar.', name);
end
if isinf(numericValue)
    if allowInf && numericValue > 0
        value = Inf;
        return;
    end
    error('%s must be finite.', name);
end

value = fix(numericValue);
if value < 0
    error('%s must be >= 0', name);
end
end
