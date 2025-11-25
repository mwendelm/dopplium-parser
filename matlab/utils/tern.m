function y = tern(cond, a, b)
% TERN Ternary operator - return a if cond is true, else b
%   y = tern(cond, a, b)
%
%   INPUTS
%     cond : condition (logical)
%     a    : value to return if true
%     b    : value to return if false
%
%   OUTPUTS
%     y : a or b

    if cond, y = a; else, y = b; end
end
