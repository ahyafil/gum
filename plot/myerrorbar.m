function h = myerrorbar(X, Y, E, varargin)
% MYERRORBAR plots error bars, with additional features compared to
% built-in function errorbar: it can plot error bars below and/or above center values, controls independently the style of line and error bars,
% controls the width of horizontal ticks, and allows to draw vertical as well horizontal bars.
%
% myerrorbar(X,Y,E) plots vector X against vector Y, together with symmetrical vertical error bars
% specified by vector E, i.e. bars join points Y(i)-E(i) and Y(i)+E(i).
%
% myerrorbar(X,Y,L,U) allows plotting of asymetrical error bars. L and U
% are vectors specifying the length of bars respectively below and above
% its center, i.e. join points Y(i)-L(i) and Y(i)+U(i).
% myerrorbar(X,Y,U,'upper') plots only upper section of error bars.
% myerrorbar(X,Y,L,'lower') plots  only lower section.
%
% myerrorbar(...,orientation) set the orientation of the error bars:
% 'vertical' (default option) or 'horizontal'. For horizontal, error bars
% join points X(i)-E(i) and X(i)+E(i)
%
% myerrorbar(...,Sline) where Sline is a character string made of symbols defining
% line types, plot symbols and colors for line joining X and Y, e.g. 'k--' or 'ro'
% (see plot documentation). Use 'none' to plot only error bars with no line
% joining center points Y(i).
%
% myerrorbar(...,'errorbar', Sbar) where Sbar is a character string made of symbols defining
% line types, plot symbols and colors for error bars, e.g. 'k-' or 'ro'
% (see plot documentation). 
%
% myerrorbar(...,parameter, value) to specify additional properties of the
% line and error bars. Parameters that control line properties are the same as for the "plot" function:
%'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'.See plot help for more information.
% Additional parameters controlling error bars properties are:
% 'ErrorBarStyle': line style for error bars.
% 'ErrorBarColor': RGB or character controlling color of error bars (including ticks).
% 'ErrorBarWidth': controls the width of the line composing error bars (including ticks).
% 'ErrorBarMarker': controls the marker placed at each end of error bars.
% 'ErrorBarMarkerSize': controls the size of the marker placed at each end of
% error bars.
% 'ErrorBarMarkerFaceColor': control the color of markers at each end of error
% bars
% 'TickLength' : controls the length of the horizontal ticks at each end of the bars.
% Set to 0 to avoid plotting ticks.
% 'TickColor': RGB or character controlling color of ticks only.
% 'TickWidth': controls the width of the line composing ticks.
% h = myerrorbar(...) provides plot handle(s) for the line and error bars. First
% handle represents  the line joining centers Y (if present), second handle represents error bar itself, third and fourth handles represent
% ticks at the end of the error bars (if present).
%
% Example:
% figure;
% h = myerrorbar(0:20, sin(pi*(0:.1:2)), .1*(10:31), 'r');
%
% figure;
% h = myerrorbar(1:4, 5:8, {[],[.1 .6 .2 .9]}, 'ticklength' .1); % plot only upper bars
%
% See also errorbar


%Test!! And change examples
% make line appears above in gca when drawing only upper or lower

%% parse X, Y, L and U

% CHECK x AND y
if ~isnumeric(X) || ~isvector(X)
    error('X should be a numeric vector');
end
if ~isnumeric(Y) || ~isvector(Y)
    error('Y should be a numeric vector');
end
n = length(X); % number of elements in vector
if length(Y) ~=n
    error('numbers of elements in X and Y do not match'),
end

drawlower = 1;   % by default draw ticks at both ends
drawabove = 1;

% check error
if nargin>3 && isnumeric(varargin{1}) % myerrorbar(X,Y,L,U,...) syntax
    L = E;
    U = varargin{1};
    varargin(1) = [];
    if isempty(L) && isempty(U) %no bar to be drawn
        h = [];
        return;
    end
    if isempty(L) % no error values below
        L = zeros(1,n);
        drawlower = 0;
    end
    if isempty(U) % no error values above
        U = zeros(1,n);
        drawabove = 0;
    end
    if ~isnumeric(L) || ~isvector(L)
        error('L should be a numeric vector');
    end
    if ~isnumeric(U) || ~isvector(U)
        error('U should be a numeric vector');
    end
    if length(L) ~= n
        error('numbers of elements in X and L do not match'),
    end
    if length(U) ~= n
        error('numbers of elements in X and E do not match'),
    end
else % myerrorbar(X,Y,E,...) syntax
    if ~isnumeric(E) || ~isvector(E)
        error('E should be a numeric vector');
    end
    if length(E) ~= n
        error('numbers of elements in E and Y do not match'),
    end
    L = E;
    U = E;
end


%% parse options

% default value
plotline = 1;
vertical = true; % be default plot vertical error bars
linestyle = ''; %line type for main line
linecolor = ''; % color for main line
linewidth = []; % line width for line
linemarker = ''; %marker for main line
linemarkersize = []; %marker for main line
linemarkerfacecolor = ''; % marker face color for line
errorbarstyle = ''; %line type for error bars
errorbarcolor = ''; % color for error bars
errorbarwidth = []; % line width for error bars
errorbarmarker = ''; %marker for error bars
errorbarmarkersize = []; %marker size for error bars
errorbarmarkerfacecolor = ''; % marker face color for bars
tickcolor = ''; % color for ticks
ticklength = []; % length of ticks
tickwidth = []; % line width for ticks


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
        case 'upper'
            drawlower = 0;
        case 'lower'
            drawabove = 0;
        case 'none'
            plotline = 0;
                case   'color'
            v = v + 1;
            linecolor = varargin{v};
            errorbarcolor = varargin{v};
        case   'linecolor'
            v = v + 1;
            linecolor = varargin{v};
        case   'linestyle'
            v = v + 1;
            linestyle = varargin{v};
        case   'linewidth'
            v = v + 1;
            linewidth = varargin{v};
        case   'marker'
            v = v + 1;
            linemarker = varargin{v};
        case   'markersize'
            v = v + 1;
            linemarkersize = varargin{v};
        case   'markerfacecolor'
            v = v + 1;
            linemarkerfacecolor = varargin{v};
        case 'errorbarwidth'
            v = v+1;
            errorbarwidth = varargin{v};
        case   'errorbarcolor'
            v = v + 1;
            errorbarcolor = varargin{v};
        case 'errorbarstyle'
             v = v + 1;
            errorbarstyle = varargin{v};
        case   'errorbarmarker'
            v = v + 1;
            errorbarmarker = varargin{v};
        case   'errorbarmarkersize'
            v = v + 1;
            errorbarmarkersize = varargin{v};
        case   'errorbarmarkerfacecolor'
            v = v + 1;
            errorbarmarkerfacecolor = varargin{v};
        case   'tickcolor'
            v = v + 1;
            tickcolor = varargin{v};
        case 'ticklength'
            v = v + 1;
            ticklength = varargin{v};
        case   'tickwidth'
            v = v + 1;
            tickwidth = varargin{v};
        case 'errorbar'
            v = v + 1;
            [XL,XC,XM,MSG] = colstyle(varargin{v});
            % check whether it is line specification
            if ~isempty(MSG)
                error('Invalid line spec for error bar');
            end
            % fill in non empty line specs
            if ~isempty(XL)
                errorbarstyle = XL;
            end
            if ~isempty(XC)
                errorbarcolor = XC;
            end
            if ~isempty(XM)
                errorbarmarker = XM;
            end
        otherwise
            % check whether it is line specification
            [XL,XC,XM,MSG] = colstyle(varg);
            if ~isempty(MSG)
                error('incorrect parameter: %s', varg);
            end
            % fill in non empty line specs
            if ~isempty(XL)
                linestyle = XL;
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



% default barwidth
if isempty(ticklength)
    if length(X)<=1
        ticklength = .3;
    else
        ticklength = .3 * nanmin(diff(sort(X)));
    end
elseif ticklength ==0 % do not draw ticks at each side of error bar
    drawlower = 0;
    drawabove = 0;
end

%%  plot error bar

% whether error bars modulate values along centers defined by Y or X
if vertical
    centers = Y;
    ref = X;
else
    centers = X;
    ref = Y;
end

% modify vectors into vector [X(1) X(1) nan X(2) X(2) nan X(3)...]
RR = kron(ref(:)',[1 1 1]); % make 3 repetitions for each version of ref
RR(3:3:end) = nan;

% error bar
EE = nan(1,3*n); % vertical vector
EE(1:3:end) = centers - L;
EE(2:3:end) = centers + U;

% which parameter/value pairs to add
barlinepars = {}; % for line
barmarkerpars = {}; % for marker
if ~isempty(errorbarstyle)
    barlinepars = [barlinepars {'linestyle',errorbarstyle}];
end
if ~isempty(errorbarcolor)
    barlinepars = [barlinepars {'color',errorbarcolor}];
else %default black
    barlinepars = [barlinepars {'color','k'}];
end
if ~isempty(errorbarwidth)
    barlinepars = [barlinepars {'linewidth',errorbarwidth}];
end
if ~isempty(errorbarmarker)
    barmarkerpars = [barmarkerpars {'marker',errorbarmarker}];
end
if ~isempty(errorbarmarkersize)
    barmarkerpars = [barmarkerpars {'markersize',errorbarmarkersize}];
end
if ~isempty(errorbarmarkerfacecolor)
    barmarkerpars = [barmarkerpars {'markerfacecolor',errorbarmarkerfacecolor}];
end

% plot line (we need to draw separately line and markers to avoid drawing
% markers at centers when plotting only lower or upper bar)
withbarline = ~strcmpi(errorbarstyle,'none');
if withbarline % line
    if vertical
        XX = RR;
        YY = EE;
    else
        XX = EE;
        YY = RR;
    end
    
    h.error = plot(XX, YY,barlinepars{:});
    
end
if ~isempty(errorbarmarker) && ~strcmpi(errorbarmarker, 'none')
    if drawlower
        h.error(1+withbarline) = plot(XX(1:3:end), YY(1:3:end),'linestyle', 'none', barmarkerpars{:}); % markers below
    end
    if drawupper
        h.error(1+withbarline+drawlower) = plot(XX(2:3:end), YY(2:3:end),'linestyle', 'none', barmarkerpars{:}); % markers below
    end
end


ish = ishold;
hold on;

%% plot line
if plotline
    
    % which parameter/value pairs to add
    linepars = {};
    if ~isempty(linestyle)
        linepars = [linepars {'linestyle',linestyle}];
    end
    if ~isempty(linecolor)
        linepars = [linepars {'color',linecolor}];
    end
    if ~isempty(linewidth)
        linepars = [linepars {'linewidth',linewidth}];
    end
    if ~isempty(linemarker)
        linepars = [linepars {'marker',linemarker}];
    end
    if ~isempty(linemarkersize)
        linepars = [linepars {'markersize',linemarkersize}];
    end
    if ~isempty(linemarkerfacecolor)
        linepars = [linepars {'markerfacecolor',linemarkerfacecolor}];
    end
    
    % plot
    h.line = plot(X, Y, linepars{:});
end



%% draw ticks

% X values for ticks
Rtick = RR;
if strcmp(get(gca, 'xscale'),'log') % log-scale : multiply values
    Rtick(1:3:end) = ref / exp(ticklength/2);
    Rtick(2:3:end) = ref * exp(ticklength/2);
else % linear scale : add values
    Rtick(1:3:end) = ref - ticklength/2;
    Rtick(2:3:end) = ref + ticklength/2;
end

% which parameter/value pairs to add
tickpars = {};
if ~isempty(errorbarstyle)
    tickpars = [tickpars {'linestyle',errorbarstyle}];
end
if ~isempty(tickcolor)
    tickpars = [tickpars {'color',tickcolor}];
elseif ~isempty(errorbarcolor)
    tickpars = [tickpars {'color',errorbarcolor}];
else % default black
    tickpars = [tickpars {'color','k'}];
end
if ~isempty(tickwidth)
    tickpars = [tickpars {'linewidth',tickwidth}];
elseif ~isempty(errorbarwidth)
    tickpars = [tickpars {'linewidth',errorbarwidth}];
end

% lower bar
if drawlower
%    LL = repelem(centers-L,3);
LL = kron(centers(:)-L(:),[1 1 1]')'; 
LL(3:3:end) = nan;
    if vertical
        h.tick(1) = plot(Rtick, LL, tickpars{:});
    else
        h.tick(1) = plot(LL, Rtick, tickpars{:});
    end
end

% upper bar
if drawabove
  %  UU = repelem(centers+U,3);
       UU = kron(centers(:)+U(:),[1 1 1]')';

    UU(3:3:end) = nan;
    if vertical
        h.tick(1+drawlower) = plot(Rtick, UU, tickpars{:});
    else
        h.tick(1+drawlower) = plot(UU,Rtick,  tickpars{:});
    end
end

if ~ish
    hold off;
end
end


%% minimum, ignoring nans
function m = nanmin(x)
x(isnan(x)) = [];
m = min(x);
end