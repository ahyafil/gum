function   C = color2mat(C, n)
% color2mat(C,n) converts C into n-by-3 RGB matrix
%
% C can be:
% - a color symbol (one character from 'ymcrgbwk') or an string with these
% characters (e.g. 'br' creates a 2-by-3 where the first colour is blue and
% the second is red)
% - a colormap string( e.g. 'jet','hsv')
% - 'flat' to interpolate colors from current colormap
% - a cell array of RGB values or colour characters (or a combination of
% them), e.g. {'k',[0 .3 .7],'r'}

assert(isscalar(n) && n>=0,'n should be a non-negative integer');

if ischar(C)
    switch lower(C)
        case  colormapstrings() % colormap string, e.g. 'jet'
            C = eval(C);
            if n>1
                corvec = 1 + (size(C,1)-1) * (0:n-1)/(n-1);
            else
                corvec = 1;
            end
            C = C(floor(corvec),:);
        case {'flat','colormap'} %interpolate colors from colormap
            C = colormap;  %current colormap
            if n>1
                corvec = 1 + (size(C,1)-1) * (0:n-1)/(n-1);
            else
                corvec = 1;
            end
            C = C(floor(corvec),:);
        otherwise
            if all(ismember(lower(C),'ymcrgbwk'))  %colour symbols (e.g. 'k')
                assert(numel(C)==n || isscalar(C),'the number of characters must match n');
                C = num2cell(C);
            else
                error('incorrect colour string:%s',C);
            end

    end
end

% if cell array, convert to matrix of RGB values
if iscell(C)
    cor_mat = zeros(length(C),3);
    for c=1:length(C)
        if isnumeric(C{c}) % RGB value
            cor_mat(c,:) = C{c};
        elseif ischar(C{c}) % letter (e.g. 'k', 'b', etc.)
            cor_mat(c,:) = rem(floor((strfind('kbgcrmyw', C{c}) - 1) * [0.25 0.5 1]), 2);
        else
            error('incorrect value for colour: should be vector of RGB value or single character');
        end
    end
    C = cor_mat;
end

% list of all colormapstrings
function C = colormapstrings()
C =  {'parula','jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism',...
    'spring','autumn','summer','winter', 'bone','lines','flag'};
end
end