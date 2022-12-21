function str = num2strcell( vect, ndig)
%S = num2strcell(X) output a cell array of string corresponding to each
%value in numerical array X (i.e. S{i} = num2str(X{i})). S is the same size
%as X.
%S = num2strcell(X, ndigits) to ensure a minimum value of digits
%S = num2strcell(X, format) for a specific format, e.g. '%0.2f'
%
%S = num2strcell(string, X) to insert numbers into a given string
%e.g. num2strcell('X%d', 2:4) outputs {'X2','X3','X4'}
%
% See also num2str


if isnumeric(vect)

str = cell(size(vect));
for i=1:numel(vect)
    str{i} = num2str(vect(i));
    if nargin>1 
        if isstr(ndig) % format
            str{i} = num2str(vect(i), ndig);
        elseif  length(str{i})<ndig
        zer = repmat('0', 1, ndig - length(str{i}));
        str{i} = [zer str{i}];
        end
    end
end

else
    istr = vect;
    vect = ndig;
    str = cell(size(vect));
    for i=1:numel(vect),
        str{i} = sprintf(istr, vect(i));
    end
end