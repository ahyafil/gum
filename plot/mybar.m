function h = mybar(ref, centers, E, varargin)
% MYBAR plots bars with error bars.
%
% mybar(X,Y,E) plots bars of height Y at X-valus specified by vector X, together with symmetrical vertical error bars
% specified by vector E, i.e. bars join points Y(i)-E(i) and Y(i)+E(i). If
% Y and E are m-by-n matrices (m is the length of vector X), mybar plots
% group of bars (warning: dimensions are inversed compared to built-in bar
% function).
%
% mybar(X,Y,L,U) allows plotting of asymetrical error bars. L and U
% are vectors specifying the length of bars respectively below and above
% its center, i.e. join points Y(i)-L(i) and Y(i)+U(i).
% mybar(X,Y,U,'upper') plots only upper section of error bars.
% mybar(X,Y,L,'lower') plots  only lower section.
%
% mybar(Y,X,E,'horizontal') or mybar(Y,X,L,U,'horizontal') draws horizontal
% bars instead of vertical bars. Error bars join points X(i)-E(i) and X(i)+E(i).
% [ mybar(X,Y,E,'vertical') is equivalent to mybar(X,Y,E) ]
%
% mybar(...,Sline) where Sline is a character string made of symbols defining
% line types, plot symbols and colors for bars, e.g. 'r' or 'k--'
% (see plot documentation).
%
% mybar(...,'errorbar', Sbar) where Sbar is a character string made of symbols defining
% line types, plot symbols and colors for error bars, e.g. 'k-' or 'ro'
% (see plot documentation).
%
% mybar(...,parameter, value) to specify properties of the
% bars and error bars. Possible parameters are:
% 'BaseValue': baseline location for bars (scalar, default:0).
% 'Barwidth' : relative width of individual bars (scalar between 0 and 1, default 0.8)
% 'FaceColor': Bar fill color, can take values: 'flat' (default) | 'none' | RGB triplet | color string
% 'LineStyle' : line style of bar edges
% 'LineColor': Defines color for all lines: bar outlines, error bars and
% ticks. Can take values: 'flat' | 'none' | RGB triplet | color string.
% 'LineWidth': Scalar defining width for all lines: bar outlines, error bars and
% ticks.
% 'EdgeColor': Specifically bar outline color, can take values : 'flat' | 'none' | RGB triplet | color string
% 'Edgestyle': Line style for bar outline (default: continuous line '-')
% 'ErrorBarStyle': line style for error bars.
% 'ErrorBarColor': RGB or character controlling color of error bars (including ticks).
% 'ErrorBarWidth': controls the width of the line composing error bars (including ticks).
% 'ErrorBarMarker': controls the marker placed at each end of error bars.
% 'ErrorBarMarkerSize': controls the size of the marker placed at each end of
% error bars.
% 'ErrorBarMarkerFaceColor': control the color of markers at each end of error
% bars
% 'TickLength' : controls the length of the horizontal ticks on each side of each end of the bars.
% Set to 0 to avoid plotting ticks.
% 'TickColor': RGB or character controlling color of ticks only.
% 'TickWidth': controls the width of the line composing ticks.
%
% h = mybar(...) provides plot handle(s) for the bars and error bars.
%
% See also myerrorbar, bar, barh, errorbar
%
% Inspired by barweb
%

% make work with matrix
% define value for ticklength
% !! allow  values to be cell
% add exampe in help
% deal with colors




%% parse X, Y, L and U

errorbarpars = {};

% CHECK x AND y
if ~isnumeric(ref)
    error('X should be numeric');
end
if ~isnumeric(centers)
    error('Y should be numeric');
end

% check error
if nargin>3 && isnumeric(varargin{1}) % myerrorbar(X,Y,L,U,...) syntax
    L = E;
    U = varargin{1};
    varargin(1) = [];
    if isempty(L) && isempty(U) %no bar to be drawn
        h = [];
        return;
    end
    if ~isnumeric(L)
        error('L should be numeric');
    end
    if ~isnumeric(U)
        error('U should be numeric');
    end
else % myerrorbar(X,Y,E,...) syntax
    if ~isnumeric(E)
        error('E should be numeric');
    end
    L = E;
    U = E;
end


%% parse options

% default value
vertical = true; % be default plot vertical error bars
linestyle = 'none'; % line style for bars
linecolor = ''; % color for all lines
linewidth = []; % line width for all ines
basevalue = 0;
barwidth = .8; % bar width
facecolor = ''; % face color for bars
edgecolor = ''; % face color for bar outline
edgestyle = ''; %line type for bar outline
ticklength = []; % length of ticks

v = 1;
while v<=length(varargin)
    varg = varargin{v};
    if ~ischar(varg)
        error('incorrect class for argument %d', v+3);
    end
    switch lower(varg)
        case 'horizontal'
            vertical = false;
        case 'vertical'
            vertical = true;
        case {'upper','lower'}
            errorbarpars{end+1} = varg;
                    case   'linestyle'
            v = v + 1;
            linestyle = varargin{v};
        case   'linecolor'
            v = v + 1;
            linecolor = varargin{v};
        case   'linewidth'
            v = v + 1;
            linewidth = varargin{v};
        case   'basevalue'
            v = v + 1;
            basevalue = varargin{v};
        case   'barwidth'
            v = v + 1;
            barwidth = varargin{v};
        case   'facecolor'
            v = v + 1;
            facecolor = varargin{v};
        case   'edgecolor'
            v = v + 1;
            edgecolor = varargin{v};
        case   'edgestyle'
            v = v + 1;
            edgestyle = varargin{v};
        case   {'errorbarstyle','errorbarwidth','errorbarcolor', 'errorbarmarker','errorbarmarkersize','errorbarmarkerfacecolor',...
                'tickcolor','ticklength', 'tickwidth','errorbar'}
            errorbarpars(end+1:end+2) = varargin(v:v+1);
            v = v + 1;
        otherwise
            % check whether it is line specification
            [XL,XC,XM,MSG] = colstyle(varg);
            if ~isempty(MSG)
                error('incorrect parameter: %s', varg);
            end
            % fill in non empty line specs
            if ~isempty(XL)
                linewidth = XL;
            end
            if ~isempty(XC)
                linecolor = XC;
            end
            if ~isempty(XM)
                linemarker = XM;
            end
    end
    
    v = v+1;
end

%%  plot error bar

% % whether error bars modulate values along centers defined by Y or X
% if vertical
%     centers = centers;
%     ref = ref;
% else
%     centers = ref;
%     ref = centers;
% end

%% check sizes

% % ! from barweb, not too sure that this is necessary
if size(centers,2) == 1
    %  centers = centers';
end

[numbars, numgroups] = size(centers); % number of bars in a group (group=same color) and number of groups

if isempty(L) % no error values below
    L = zeros(numbars,numgroups);
    errorbarpars{end+1} = 'upper';
end
if isempty(U) % no error values above
    U = zeros(numbars,numgroups);
    errorbarpars{end+1} = 'below';
end

if vertical
refstr = 'X';
ctstr  = 'Y';
else
 refstr = 'Y';
ctstr  = 'X';
end

%if vertical
    if ~isvector(ref)
        error('%s should be a vector', refstr);
    end
    if length(ref) ~=numbars
        error('The length of %s must match the number of rows of %s', refstr, ctstr);
    end
    if ~isequal(size(L),[numbars, numgroups]) || ~isequal(size(U),[numbars, numgroups])
        error('error bars matrix must have the same size as %s', ctstr);
    end
% else
%     if ~isvector(centers)
%         error('Y should be a vector (horizontal bars)');
%     end
%     if length(centers) ~=numbars
%         error('The length of Y must match the number of rows of X (horizontal bars)'),
%     end
%     if ~isequal(size(L),[numbars, numgroups]) || ~isequal(size(U),[numbars, numgroups])
%         error('error bars matrix must have the same size as X');
%     end
% end



%% plot bars


if vertical
    ref_field = 'XData';
    ref_offset = 'XOffset';
    center_field = 'YData';
else
    ref_field = 'XData'; 'YData';
    ref_offset = 'XOffset'; %'YOffset';
    center_field = 'XData';
end

%if row vector, add another row otherwise matlab automatically plots it as non-grouped bar
if numbars==1 && numgroups>1
    ref = [ref ref+1];
    centers = [centers; zeros(1,numgroups)];
    % E = [E; zeros(1,numbars)];
    %  change_axis = 1;
end

% which parameter/value pairs to add
barpars = {};
if ~isempty(basevalue)
    barpars = [barpars {'basevalue',basevalue}];
end
%if ~isempty(facecolor),
%    barpars = [barpars {'facecolor',facecolor}];
%end
if ~isempty(edgestyle)
    barpars = [barpars {'edgestyle',edgestyle}];
end
if isempty(edgecolor) && ~isempty(linecolor)
    edgecolor = linecolor;
end
% if ~isempty(edgecolor),
%     barpars = [barpars {'edgecolor',edgecolor}];
% elseif ~isempty(linecolor),
%     barpars = [barpars {'edgecolor',linecolor}];
% end
if ~isempty(linewidth)
    barpars = [barpars {'linewidth',linewidth}];
end
if ~isempty(linestyle)
    barpars = [barpars {'linestyle',linestyle}];
end

% Plot bars
if vertical
    h.bar = bar(ref, centers, barwidth,barpars{:});
else
  %  h.bar = barh(centers, ref, barwidth,barpars{:});
     h.bar = barh( ref, centers,barwidth,barpars{:});
end

% set face color for each series
if ~isempty(facecolor)
    if size(facecolor,1) == 1 % if same color for each series
        facecolor = repmat(facecolor,numgroups,1);
    end
    for b=1:numgroups
        set(h.bar(b), 'facecolor', facecolor(b,:));
    end
    if numgroups==1 && size(facecolor,1)>1 % different colours within same group
           if size(facecolor,1)>length(ref)
               facecolor = facecolor(1:length(ref),:);
           end
        set(h.bar, 'FaceColor','flat','CData',facecolor);
    end
end

% set face color for each series
if ~isempty(edgecolor)
    if size(edgecolor,1) == 1 % if same color for each series
        edgecolor = repmat(edgecolor,numgroups,1);
    end
    for b=1:numgroups
        set(h.bar(b), 'edgecolor', edgecolor(b,:));
    end
end


% if just one group, remove zeros created at point 2
if numbars==1 && numgroups>1
    ref(2) = [];
    centers(2,:) = [];
    for b=1:numgroups
        set(h.bar(b),ref_field,ref,center_field,centers(b));
    end
end

ish = ishold;
hold on;
pause(1e-6); % pause is required to retrieve offset value

%% plot error bars

% add value for line color and width to parameters for error bars
if ~isempty(linewidth)
    errorbarpars = [{'linewidth',linewidth} errorbarpars];
end

% default ticklength
if isempty(ticklength)
    if numgroups>1
        
        % compute offset for each bar series
        for i = 1:numgroups
            if verLessThan('matlab', '8.4')
                R_eb =get(get(h.bar(i),'children'),ref_field);
                offset(i) = mean(R_eb([1 3],1))-ref(1);
            else
                offset(i) = h.bar(i).(ref_offset);
            end
        end
        ticklength = .6*min(diff(offset)); % 60% of distance between two bar series
        
    elseif length(ref)<=1
        ticklength = .6;
    else
        ticklength = .6 * nanmin(diff(sort(ref)));
    end
end
errorbarpars = [{'ticklength',ticklength} errorbarpars];

% retrieve the position on X axis of each bar series
for i = 1:numgroups
    if verLessThan('matlab', '8.4')
        R_eb = get(get(h.bar(i),'children'),ref_field);
        R_eb = mean(R_eb([1 3],:));
        if numbars==1 && numgroups>1
            R_eb(2) = [];
        end
    else
        R_eb = bsxfun(@plus, h.bar(i).(ref_field), [h.bar(i).(ref_offset)]');
    end
    if isempty(linecolor) % no defined color
        colorpars ={};
    elseif size(linecolor,1) ==1 %same color for each series
        colorpars = {'color',linecolor};
    else % different colors for each series
                colorpars = {'color',linecolor(i,:)};
  %  errorbarpars = [{'color',linecolor} errorbarpars];
    end
    
    
    if vertical
        h.error(i) = myerrorbar(R_eb, centers(:,i), L(:,i),U(:,i), 'vertical',errorbarpars{:},colorpars{:},'none');
    else
        h.error(i) = myerrorbar(centers(:,i),R_eb, L(:,i),U(:,i), 'horizontal',errorbarpars{:},'none' );
    end
    
end

if ~ish
    hold off;
end
