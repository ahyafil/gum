function varargout = wu(varargin)
% WU is a very handy and customizable function for plotting curves and bars with error bars.
%
%* BASIC SYNTAX:
% wu(Y) or wu(Y, 'mean') if Y is a m-by-n numerical matrix will plot a n-point curve with 1 to n integers along X-axis, and the
% mean of each row in Y along Y-axis, together with error bars centered on the mean whose length
% is the standard variation for each row in Y. Nan values are ignored.
% If Y is a 3-dimension matrix, a curve with errors bars will be plotted
% for each submatrix Y(:,:,i).
% If Y is a 4-dimension or 5-dimension matrix, figure is divided in subplot
% corresponding to the different Y(:,:,:,k,l) submatrices.
% If Y is a cell array whose elements is composed of numerical vectors, mean and
% standard errors in computed within each array. A single curve with error
% bars will be drawn for vectorial cell array, multiple for 2-dimensional
% cell arrays, and multiple subplots for 3- and 4-dimension cell arrays.
%
% wu(Y,'median') uses medians of each row instead of means.
%
% wu(Y,errortype) specifies how error values are computed. Possible values
% are:
% - 'std' :standard deviation (default option for 'mean')
% - 'ste': standard error of the mean
% - 'correctedste': standard error of the mean removing mean value over
% all data in the same line (i.e. removing variance due to random factor),
% provides a visual hint of significance of t-test
% - 'quartile': plots 1st and 3rd quartile
% - 'noerror' to simply plot mean values without errors.
%
% wu(M,E) allows to directly specify the values for the means and errors. M
% and E must be numerical matrices (up to 4 dimensions) of the same size.
% Use wu(M,[]) to only plot mean values (with no error bars).
%
% wu(X,Y) or wu(X,M,E) allows to specify in vector X the values along the
% X-axis.
%
% wu([],M,L,U) or wu(X,M,L,U) allows to specify in L and U the length of bars below and
% above its centers, for asymmetrical error bars.
%
% wu(..., plottype) defines plot type between the following:
% - 'curve' draws a simple curve joining values for mean (default option)
%  - 'bar' draws a bar plot with one bar for each mean value
%  - 'xy' compares values in first column of M in X-axis against value in
%  second column in M axis. M must have 2 rows ( but can be more than
%  2-dimensional), or  Y must have two columns.
%  - 'imagesc' displays matrix M as an image (where M is at least 2 dimensional, or equivalently Y is 3-dimension or more). Error values are not plotted.
%
% wu(...., 'curve','errorstyle',errorstyle) defines how error are plotted
% when mean values are plotted as a curve. Possible values are:
% - 'bar' : error bars centered on mean value (default when less than 6
% data points per curve)
% - 'curve' : two curves above and below the main curve, join respectively
% points M+E and points M-E (default when 6 or more data points per curve)
% - 'area' : shades an area defines betwen points M-E and M+E
%
% wu(..., {variablename1 variablename2 ...}) specifies labels for the
% independent factors in matrix Y (or M and E), i.e. labels for dimension
% 1, 2, etc. in matrix M. This is used to label X-axis, add legend and
% title. Use '' for a variablename to avoid naming it. Accepts also string
% arrays
%
% wu(...., { {v1label1 v1label2...},{v2label1 v2label2...}) specifies labels for
% the different levels within each variable, i.e. for the different rows,
% columns, etc. of matrix M (corresponding to columns of Y, elements Y(:,:,i), Y(:,:,:,i), etc.).
% These labels are used to set ticknames along X-axis, legend, and
% subtitles. Leave Use {} for a variable to avoid labelling its level.
%
% wu(...., 'pvalue',P) where P is a numerical matrix of p-values of size equal to that of M
% allows to draw lines to signal points reaching statistical significance.
% wu(Y, 'pvalue','on') or wu(X,Y, 'pvalue','on') computes p-value by
% applying two-sided t-test on the columns of Y.
% wu(....,'threshold',alpha) adjusts the threshold for statistical
% significance (by default 0.05).
% wu(....,'correction','on') uses Bonferroni correction for multiple
% comparison on threshold, i.e. uses as threshold alpha/numel(M).
% wu(....'statsstyle',str) applies style str to significance line (e.g.
% '*','--')
%
%* PLOT SPECS:
% wu(...., specname, specvalue) allows to specify plot options. Available
% specnames are:
% -'ylabel': label for dependent variable (used to label Y-axis)
% - 'color' : defines colour for mean values (bars or curves). Value can
% be: a single color (either RGB vector or character such as 'k' or 'b')
% sue the same color for each plot; a cell array of colours (each element
% being RGB or symbol) or a n x 3 matrix to associate a different color
% with each curve/bar series (i.e. each column of M); a string array of
% a built-in colormap (e.g. 'jet', 'hsv', 'gray') to use graded colors
% within the colormap; or 'flat' to use graded colors from the current
% colormap (allows to use user-defined colormap).
% - 'errorstyle' : defines how error values E are represented. Possible
% values are 'bar' (an error bar associated with each value, default option), 'line' (a
% dashed error line plotted above the mean line at M+E, and another below at M-E),
% and 'fill' (a light colour surface covering all area between M-E and M+E).
% - 'axis' : defines on which axis dependent measures (M and E) are
% plotted. Default is 'y'. Use 'x' to swap axes (e.g. for horizontal bars).
% - 'legend': use 'color' to use coloured text for legend (default),
% 'standard' to use matlab built-in legend style, or 'none' to avoid
% plotting legend
% - 'axis' : on which axis dependent measure is plotted along:
%'Y' (e.g. plot vertical bars, default value) or 'X' (underconstruction)
% - 'permute': permute the order of dimensions: 'auto' to optimize figure
% readability, or a permutation order
% - 'layout': whether to use tiled layout (when more than one subplot)
%
% !!check that it works
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
% 'XTickRotate' : rotates Xtick labels by specified angle (for matlab
% version prior to 2014b, requires function rotateXlabels)
%
% Options for bar style only:
% 'BaseValue': baseline location for bars (scalar, default:0).
% 'Barwidth' : relative width of individual bars (scalar between 0 and 1, default 0.8)
% 'FaceColor': Bar fill color, can take values: 'flat' (default) | 'none' | RGB triplet | color string
% 'Shades': boolean [default:true], whether face color within each bar
% series uses different shades of the same colour
% 'LineColor': Defines color for all lines: bar outlines, error bars and
% ticks. Can take values: 'flat' | 'none' | RGB triplet | color string.
% 'LineWidth': Scalar defining width for all lines: bar outlines, error bars and
% ticks.
% 'EdgeColor': Specifically bar outline color, can take values : 'flat' | 'none' | RGB triplet | color string
% 'Edgestyle': Line style for bar outline (default: continuous line '-')
% 'VerticalLabel': if value is set to 'on', places labels as text over each
% bar instead along the axis (only for single bar series)
%
% Options for 'curve'  and 'area' error style only:
% 'errorlinestyle' : line style for curves at M+E and M-E (default: '--'
% for 'curve' error style, 'none' for 'area' error style)
% 'shift': value S. the different curves are for better  horizontally shifted by values
% in the range [-S +S] for better visibility. Use 'auto' for automatic
% computation of the range. Default: 0 (no shift).
%
% Options for 'xy' plot only:
% 'IdentityLine' : plots identity line (i.e. line x=y) if set to true
% (default:false)
%
% Options for imagesc only:
% 'clim':
%
%* OUTPUT ARGUMENTS:
% [M E h] = wu(...). h is the plot handle for the different graphic
% objects
%
% See also mybar, myerrorbar

%TODO:
% check help for options
% add check for xy ( just 2 variables)
% add parameters for myerrorbar and mybar
% error bars for binary data : http://stats.stackexchange.com/questions/11541/how-to-calculate-se-for-a-binary-measure-given-sample-size-n-and-known-populati
% change names for nbp and nbv
% option 'single' to have just one bar series/curve per subplot.
%clean-up and graphic handles
%'xy' : allow 3D matrix (2xAxB) on single subplot
% check imagesc
% look back at p values. use 'barwebpairs' for displaying significance
% add line specs for line.
% xtickrotation (see commented lines)
% add 'offset' to have non-overlapping error bars

%
%
%OPTIONS :
%'xtick'     : 'normal' to place X values according to values of first
%parameter levels (if numerical) ; 'index' to use regular spacing of levels
%along X axis
%'pvalue'    : p-values, same size as Ymean

%default values
X = [];
L = [];
U = [];
cor =  [];
clim = 'auto';
linestyle = "";
marker = ".";
markersize = 6;
linewidth = 2;
%ticklength = [];
plottype = 'curve';
edgecolor = '';
linecolor = '';
errorstyle = '';   % '', {'bar', 'bars', 'errorbar', 'errorbars'}, {'fill', 'area', 'marge'},  {'line', 'lines', 'errorline', 'errorlines'}
errorlinestyle = ''; % default style for curve errors
pval = [];
threshold = .05;
permut = [];
correction = 'off';
statsstyle = '-';
titl = '';
y_label = '';
name ='';
doclf = 'noclf';
xtick = 'normal'; %'normal' or 'index' (does not use level values even if real numbers) / i think its the same as 'factordirection' parameter in boxplot
maxis = 'y'; % axis for dependent variable
xtickangle = 0; % angle of rotation for xtick labels
VerticalLabel = 0; % places labels for first variable as vertical text above bars
legend_style = 'standard';
IdentityLine = 0; % identity line for 'xy' plot
shiftrange = 0; % shift in X axis for different curves
collapse = 0; % whether to collapse second & third dimension
layout = [];

%% determine syntax
numargs = cellfun(@isnumeric, varargin) | cellfun(@islogical, varargin) | cellfun(@isdatetime, varargin);  %which arguments are numeric
numargs(5) = false;                 %in case there is less than 3 args
syntax = find(numargs==0,1)-1; % number of consecutive numeric arguments
%syntax = sum(numargs(1:3));

if syntax<=2 && nargin > syntax && (syntax==0 || isvector(varargin{syntax})) && iscell(varargin{syntax+1}) && all(cellfun(@isnumeric,varargin{syntax+1}(:)))
    syntax = syntax + 1;   % for cell arrays of values
    rawinput = 1;
else
    rawinput = 0;
end

do_median = any(cellfun(@(x) isequal(x,'median'), varargin(syntax+1:end))); % whether to use median
if do_median
    avg_fun = @nanmedian;
    errorbars = 'quartile'; % default option
else
    avg_fun = @nmean;
    errorbars = 'ste';
end

switch syntax
    case 1  % just raw data : wu(Y, ...)
        Yraw = varargin{1};
        if ~rawinput    %N-D array : mean over first dimension
            M = avg_fun(Yraw,1);
            M = shiftdim(M,1);
        else % wu(Ycell, ...)
            M = cellfun(@(x) avg_fun(x,1), Yraw); %N-D cell array : mean within cells
        end
    case 2
        if rawinput % wu(X, Ycell,...)
            X = varargin{1};
            Yraw = varargin{2};
            M = cellfun(avg_fun, Yraw); %N-D cell array : mean within cells

        elseif isequal(size(varargin{1}), size(varargin{2}))  % wu(Ymean, Yerror, ...)
            M = varargin{1};
            L = varargin{2};
            U = L;
        elseif isempty(varargin{2})   % wu(Ymean, [], ...) ; mean, no error bars
            M = varargin{1};
            L = NaN(size(M));
            U = L; %symmetric bars
        else % wu(X, Y)
            X = varargin{1};
            Yraw = varargin{2};
            M = avg_fun(Yraw,1);
            M = shiftdim(M, 1); % pass on dimension 2 to dimension 1, dim 3 to dm 2, etc.
        end
    case 3 % wu(X, M, E)
        X = varargin{1};
        M = varargin{2};
        if ~isempty(varargin{3})
            L = varargin{3};
        else  % no error bar
            L = NaN(size(M));
        end
        if isrow(M) && length(X)>1 % if M is provided as row instead of column, correct
            M = M';
            L = L';
        end
        U = L; %symmetric bars
    case 4 % case(X, M, L, U)
        X = varargin{1};
        M = varargin{2};
        if ~isempty(varargin{3})
            L = varargin{3};
        else  % no error bar
            L = NaN(size(M));
        end
        if ~isempty(varargin{4})
            U = varargin{4};
        else  % no error bar
            U = NaN(size(M));
        end
        if isrow(M) && (length(X)>1) % if M is provided as row instead of column, correct
            M = M';
            L = L';
            U = U';
        end
    otherwise
        error('incorrect syntax');
end

if isempty(M)
    plothandle = [];
    return;
    varargout = cell(1,nargout);
end

if ~isequal(size(M),size(L)) && ~isempty(L)
    d = 1;
    while size(M,d)==size(L,d)
        d = d+1;
    end
    error('M and L should have the same size, differed on dimension %d (%d vs %d)',d,size(M,d),size(L,d));
end

% check that data is real
if ~all([isreal(X) isreal(M) isreal(L) isreal(U)])
    warning('MATLAB:plot:IgnoreImaginaryXYPart','Imaginary parts of complex X and/or Y arguments ignored');
end
imaginary_warning = warning('off','MATLAB:plot:IgnoreImaginaryXYPart'); % turn it off for all called functions


%size and dimensionality
siz = size(M);
nDim = length(siz);
if nDim==2 && siz(2)==1
    nDim = 1;
end
nPar = size(M,1); % number of parameters in M
nVar = size(M,2); % number of variables in M


%%

%default names for variables and levels
varnames = repmat({''}, 1, nDim);
levels = cell(1, nDim);
for d = 1:nDim
    if d==1 && ~isempty(X)
        if isdatetime(X)
            levels{d} = X;
        else
            %   levels{d} = num2strcell(X);
        end
    elseif siz(d)>1 && d>1
        levels{d} = num2strcell(1:siz(d));
    else % if only one level, no need to add a label
        levels{d} = {};
    end
end


%% %%%% OPTION INPUTS %%%%%
errorbarpars = {};
v = syntax+1; % where options start
while v<=length(varargin)
    varg = varargin{v};
    switch class(varg)
        case 'char'
            switch lower(varg)
                case {'mean','median'}
                    % already processed before
                case {'color','facecolor'}
                    v = v +1;
                    cor = varargin{v};
                case 'style'
                    v = v + 1;
                    marker = string(varargin{v});
                case 'linewidth' % can be used both for bars and lines
                    errorbarpars(end+1:end+2) = {'ErrorBarWidth',varargin{v+1}};
                    v = v+1;
                    linewidth = varargin{v};

                    %                 case 'ticklength',
                    %                     v = v+1;
                    %                     ticklength = varargin{v};
                case   {'barwidth','errorbarstyle','errorbarwidth','errorbarcolor', 'errorbarmarker','errorbarmarkersize','errorbarmarkerfacecolor',...
                        'tickcolor','ticklength', 'tickwidth','errorbar','basevalue'}
                    errorbarpars(end+1:end+2) = varargin(v:v+1);
                    v = v + 1;
                case 'linecolor'
                    v = v+1;
                    linecolor = varargin{v};
                case 'edgecolor'
                    v = v+1;
                    edgecolor = varargin{v};
                case 'errorbarplottype'
                    v = v+1;
                    errorbars = varargin{v};
                case 'errorlinestyle'
                    v = v+1;
                    errorlinestyle = varargin{v};
                case 'pvalue'
                    v = v+1;
                    pval = varargin{v};
                case 'threshold'
                    v = v+1;
                    threshold = varargin{v};
                case 'correction'
                    v = v+1;
                    correction = varargin{v};
                case 'statsstyle'
                    v = v+1;
                    statsstyle = varargin{v};
                    if strcmpi(statsstyle,'symbol')
                        threshold = [.05 .01 .001];
                    end
                case 'ylabel'
                    v = v+1;
                    y_label = varargin{v};
                case 'title'
                    v = v+1;
                    titl = varargin{v};
                case 'name'
                    v = v+1;
                    name = varargin{v};
                case 'errorstyle'
                    v = v+1;
                    errorstyle = varargin{v};
                case 'marker'
                    v = v+1;
                    marker = string(varargin{v});
                case 'markersize'
                    v = v+1;
                    markersize = varargin{v};
                case 'legend'
                    v = v+1;
                    legend_style = varargin{v};
                case 'linestyle'
                    errorbarpars(end+1:end+2) = varargin(v:v+1); % if 'bar' type
                    v = v+1;
                    linestyle = string(varargin{v}); % if 'curve' type

                    %                 case defcolor,
                    %                     cor = {varg};
                case {'-',':','-.','--','none'}
                    linestyle = string(varg);
                case {'.','o','x','+','*','s','d','v','^','<','>','p','h'}
                    marker = string(varg);
                case {'ste', 'std', 'correctedste', 'quartile','noerror'}
                    errorbars = varg;
                case {'curve', 'bar', 'xy', 'imagesc'}  %'noplot'
                    plottype = varg;
                case 'line'
                    plottype = 'curve';
                case {'jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism','autumn'}
                    cor = eval(varg);
                case 'clim'
                    v = v +1;
                    clim = varargin{v};
                case {'clf', 'noclf'}
                    doclf = varg;
                case 'permute'
                    v = v+1;
                    permut = varargin{v};
                case 'layout'
                    v = v+1;
                    layout = varargin{v};
                case 'xtick'
                    v = v+1;
                    xtick = varargin{v};
                case 'axis'
                    v = v+1;
                    maxis = varargin{v};
                    if strcmpi(maxis, 'x')
                        errorbarpars{end+1} = 'horizontal';
                    else
                        errorbarpars{end+1} = 'vertical';
                    end

                case 'xtickrotate'
                    v = v +1;
                    xtickangle = varargin{v};
                case 'verticallabel'
                    v = v+1;
                    VerticalLabel = strcmpi(varargin{v},'on');
                case 'identityline'
                    v = v +1;
                    IdentityLine = varargin{v};
                case 'shift'
                    v = v +1;
                    shiftrange = varargin{v};
                case 'collapse'
                    v = v +1;
                    collapse = logical(varargin{v});

                otherwise
                    % check whether it is line specification
                    [XL,XC,XM,MSG] = colstyle(varg);
                    if ~isempty(MSG)
                        error('incorrect parameter: %s', varg);
                    end
                    % fill in non empty line specs
                    if ~isempty(XL)
                        linestyle = string(XL);
                    end
                    if ~isempty(XC)
                        cor = XC;
                    end
                    if ~isempty(XM)
                        marker = string(XM);
                    end
            end
        case 'string'
            varnames = cellstr(varg);
        case 'cell'
            if ~isempty(varg) && ischar(varg{1})  % labels for variables
                %varnames = varg;
                addvarnames = ~cellfun(@isempty,varg);
                varnames(addvarnames) = varg(addvarnames);
            else  % label for values / levels
                addlevels = ~cellfun(@isempty,varg);
                levels(addlevels) = varg(addlevels);
                %  else
                %      error('unknown input');
            end
        otherwise
            error('incorrect option class: %s', class(varg) );
    end
    v = v +1;
end

% 'xy' option : last dimension must have two values
if strcmp(plottype, 'xy') && nPar ~=2
    error('wu : for ''xy'' option, size of the first dimension of data matrix must be 2');
end

% factor axis (if measure axis is Y, then it is X, and vice versa)
if strcmpi(maxis, 'x')
    faxis = 'y';
else
    faxis = 'x';
end

%% process error bar values
if strcmp(plottype, 'imagesc')
    errorbars = 'noerror';
end
if isempty(L)
    if isnumeric(Yraw) || islogical(Yraw)
        switch errorbars
            case 'ste'
                if islogical(Yraw)
                    nobs = sum(~isnan(Yraw),1);
                    L = sqrt( shiftdim(M.*(1-M),-1) ./ nobs); % estimated s.e.m for binary observations
                else
                    L = nanste(Yraw,1);
                end

            case 'std'
                if islogical(L)
                    nobs = sum(~isnan(Yraw),1);
                    L = sqrt(shiftdim(M.*(1-M),-1) .* nobs); % estimated s.e.m for binary observations
                else
                    L = nstd(Yraw,1);
                end
            case 'correctedste'
                Ym = reshape(Yraw, [size(Yraw,1) prod(siz)]); %
                Ym = mean(Ym, 2);  % means across all conditions
                L = nan_ste(bsxfun(@minus,Yraw,Ym)); %remove these means from individual values
            case 'quartile'
                L = quantile(Yraw, .25); % 25th percentile
                L = M - reshape(L, size(M)); % remove mean/median
                U = quantile(Yraw, .75); % 75th percentile
                U = reshape(U, size(M)) - M;
            case 'noerror'
                L = NaN(siz);
        end
        if ~strcmp(errorbars,'quartile')
            L = shiftdim(L,1);
            U = L;
        end
    else
        switch errorbars
            case 'ste'
                L = cellfun(@ste, Yraw);
                U = L;
            case 'std'
                L = cellfun(@std, Yraw);
                U = L;
            case 'correctedste'
                warning('cannot correct for random factor variance if data is not paired, using classical standard error instead');
                L = cellfun(@ste, Yraw);
                U = L;
            case 'quartile'
                L = M - cellfun(@(x) quantile(x,.25), Yraw);
                U = cellfun(@(x) quantile(x,.75), Yraw) - M;
            case 'noerror'
                L = NaN(size(M));
                U = L;
        end
    end
end

%% parse labels and variable names

%!! check with default names given above

%!! to be removed ? (mystats)
removeunderscore = 0;
if removeunderscore
    varnames = strrep(varnames, '_', ' '); %remove '_'
end

%if more dimensions than thought (because of singleton dimensions), append
if length(levels)>nDim
    newdimm = length(levels);
    siz(nDim+1:newdimm) = 1;
    varnames(nDim+1:newdimm) = repmat({''}, 1, newdimm-nDim);
    nDim = newdimm;
end
if strcmpi(plottype, 'xy') && nDim>2 % X-Y plot
    nDim =  nDim-1;
end

%labels for levels
for i=1:length(levels)
    %check size
    assert(isempty(levels{i}) || length(levels{i})==siz(i),...
        'Numbers of labels for variable along dimension %d (%d) does not match number of levels (%d)',i, length(levels{i}),siz(i));
    %     if  size(levels{i})>siz(i),
    %         warning('labels for variable along dim %d is %d, higher than number of levels (%d); extra ones will not be used',i, length(levels{i}),siz(i));
    %     end

    % turn to string
    levels{i} = string(levels{i});

    % replace '_' chars
    if removeunderscore
        levels{i} = strrep(levels{i}, '_', ' ');
    end
end

if VerticalLabel && ~((strcmp(plottype, 'curve')&&nVar~=1) ||  (strcmp(plottype, 'bar')&&any([nVar nPar]~=1)))
    error('vertical labels only for single series bar/curve type');
end

%% process p-values
if isempty(pval)
    pval = nan(size(M));
elseif ischar(pval) && strcmp(pval,'on') % compute p-value from t-test applied to Y
    assert(exist('Yraw','var'),'to compute p-value, syntax must be with raw array Y');
    if ~rawinput % if numerical array, convert to cell array with one column
        Yraw = num2cell(Yraw,1);
        Yraw = shiftdim(Yraw, 1); % pass on dimension 2 to dimension 1, dim 3 to dm 2, etc.
    end
    [~, pval] = cellfun(@ttest, Yraw); % apply T-test
else
    assert( isequal(size(pval),size(M)),'pvalue should have the same size as mean values');
end
if strcmp(correction, 'on') % 'Bonferroni correction',
    threshold = threshold / numel(M);
end
ymax = max( max(M(:) +abs(U(:))), max(M(:)) ); % maximum value of mean point or error
ymin = min( min(M(:) -abs(L(:))), min(M(:)) ); % minimum value of mean point or error
if isnan(ymin) || ymin==0
    ymin = 1;
end
y_pval = ymax + (.05 + .02*(0:nVar-1)/max(nVar-1,1))*(ymax-ymin); % vertical positioning of significance lines

%% permute dimensions
if ~isempty(permut)
    if isequal(permut,'auto')
        % automatic: go from highest to lowest number of levels, but with
        % some penalty to avoid too many permutations
        weighted_siz = siz ./ 1.5.^(1:length(siz));
        [~,permut] = sort(weighted_siz,'descend');
    else
        assert(length(permut) ==length(siz) && all(ismember(1:length(siz),permut)),...
            'permute option: ORDER contains an invalid permutation index.');
    end

    siz = siz(permut);
    nPar = siz(1);
    nVar = siz(2);
    M = permute(M,permut);
    L = permute(L,permut);
    U = permute(U,permut);
    pval = permute(pval,permut);
    varnames = varnames(permut);
    levels = levels(permut);
end

% use values for xtick from xticklabels, if relevant
X = get_xtick(X, levels{1}, nPar, xtick);

%% default linestyle
def_linestyle = strlength(linestyle)==0;
if strcmp(plottype, 'xy')
    linestyle(def_linestyle) = "none";
else
    linestyle(def_linestyle) = "-";
end

%% one property for each level of dim2
collapse = collapse && nDim>2;

if isscalar(linestyle)
    if collapse && isscalar(linewidth) && isscalar(markersize) && isscalar(marker) && strcmp(plottype,'curve')
        % if we collapse over dim 2 and 3, by default we will vary
        % linestyle over dim 3
        linestyle = [linestyle ':' '-.' '--'];
        linestyle = linestyle(1+mod(0:siz(3)-1,length(linestyle)));
    else
        linestyle = repmat(linestyle, nVar,1);
    end
end
if isscalar(linewidth)
    linewidth = repmat(linewidth, nVar,1);
end
if isscalar(markersize)
    markersize = repmat(markersize, nVar,1);
end
if isscalar(marker)
    marker = repmat(marker, nVar,1);
end

%if ~iscell(linestyle)
%    linestyle = repmat( {linestyle}, 1, nVar);
%end

%% parse error style
if isempty(errorstyle)
    if nPar<=5
        errorstyle = 'bar';
    else
        errorstyle = 'fill';
    end
end

%% parse colors into n-by-3 RGB matrix
if isempty(cor)
    cor = defcolor;
    cor = cor(1+mod(0:nVar-1,length(cor)));
end
if strcmp(plottype, 'bar') && ischar(cor) && any(strcmp(cor, colormapstrings)) && nVar==1
    % in case bar plot with single series and colormap, use different color
    % for each x
    cor = color2mat(cor,nPar);
else
    cor = color2mat(cor,nVar);
end

if ~isempty(edgecolor)
    edgecolor = color2mat(edgecolor,nVar);
    errorbarpars(end+1:end+2) = {'EdgeColor',edgecolor};
end
if ~isempty(linecolor)
    linecolor = color2mat(linecolor,nVar);
    errorbarpars(end+1:end+2) = {'LineColor',linecolor};
end

% if not enough colours compared to number of variables,
% cycle through them
if size(cor,1) < nVar
    cor =  cor( mod(0:nVar-1,size(cor,1))+1 ,:  );
end

% compute light colours for margin
if strcmp(plottype, 'curve') && any(strcmp(errorstyle, {'fill', 'area', 'marge'}))
    lightcolour = .8+.2*cor; %light colors
    lightcolour = num2cell(lightcolour,2); % turn into cell array
end
cor = num2cell(cor,2); % turn into cell array


%% collapse 2nd and 3rd dimension
if collapse
    new_size = [siz(1) siz(2)*siz(3) siz(4:end)]; % new shape
    M = reshape(M, new_size);
    L = reshape(L, new_size);
    U = reshape(U, new_size);

    cor = repmat(cor(:),1,siz(3));
    lightcolour = repmat(cor,1,siz(3));
    marker = rep_for_collapse(marker,siz);
    markersize = rep_for_collapse(markersize,siz);
    linestyle = rep_for_collapse(linestyle,siz);
    linewidth = rep_for_collapse(linewidth,siz);
    if ~isempty(varnames{2}) && ~isempty(varnames{3})
        new_varname = [varnames{2} ',' varnames{3}];
    else
        new_varname = "";
    end
    varnames = [varnames(1) new_varname varnames(4:end)];
    levels = [levels(1) {interaction_levels(levels{2},levels{3})} levels(4:end)];

    if strcmp(plottype,'bar') % encode dim3 as shade of bar facecolor
        errorbarpars(end+1:end+2)= {'shades',0};

        for v=1:nVar
            towards_light = .2*min(siz(3),3);
            lighter = (1-towards_light)*cor{v,1}+ towards_light; % lightest shade
            towards_dark = .1*min(siz(3),3);
            darker = (1-towards_dark)*cor{v,1}; % darkest shade
            this_cmap = lighter + (0:siz(3)-1)'.*(darker-lighter)./(siz(3)-1);
            cor(v,:) = num2cell(this_cmap,2);
        end
    end

    nDim = nDim-1;
    nVar = nVar*siz(3);
    siz = new_size;
end

if isempty(errorlinestyle)
    switch errorstyle
        case  {'fill', 'area', 'marge'} %error areas
            errorlinestyle = 'none'; % default style for curve errors
        case {'curve', 'curves', 'errorcurve', 'errorcurves'} %error lines
            errorlinestyle = '--'; % default style for curve errors
    end
end

if isequal(shiftrange,'auto')
    shiftrange = .1 + .2/(1+exp(8-2*nVar)); % from .1 to .3 depending on number of curves
    if nVar>1
        shiftrange = shiftrange*min(diff(sort(X))); % scale by minimum step in X
    end
end
if nVar>1
    shift = linspace(-shiftrange,shiftrange,nVar);
else
    shift = 0;
end

%if ~strcmp(plottype, 'bar'),
%    cor = defcolor;
%end

%%


%use 2 levels even for dim 1
% if dimm == 1,
%     if length(levels) == length(Ymean),
%         levels =  [{''} levels];
%     else
%         levels = {[], {}};
%     end
% end



%% open figure
% if strcmp(plottype, 'noplot')
%     plothandle = [];
%     return;
% end

% !! what to do with this
if strcmp(doclf, 'clf')
    switch nDim
        case {1 2}
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0);
        case 3
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0, 250*length(levels{3}), 200);
        case 4
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0, 250*length(levels{3}), 200*length(levels{4}));
    end
    clf;
end

if strcmp(plottype,'imagesc') && nDim>2
    clim = [min(M(:)) max(M(:))];
end

% whether figure is currently on hold
is_hold = ishold;
hold on;

%% layout
if isempty(layout) && nDim>2
    % by default, we use layout if there's not any axes already in figure,
    % or version prior to 2019
    noTiling = verLessThan('matlab', '9.13'); % we need tilerowcol
    layout = isempty(get(gcf,'child')) || noTiling;
end

%% plot
is_xy = strcmp(plottype,'xy');
switch nDim %number of dimension
    case 1  % just one curve/bar series
        setclim(clim); titl = '';

        plothandle = pcurve(X(:), M(:), L(:), U(:), pval, [varnames {''}], levels);  %works without ' for curve/errorbars (maybe depends also if Ymean is row or column)

        % legend off;

    case 2  % multiple curve/bars in same panel
        setclim(clim); titl = '';
        plothandle = pcurve(X, M, L, U, pval, varnames, levels);

    case 3 % different panels
        dd = 3+is_xy;
        for s = 1:siz(dd)
            if layout
                [h_ax(s), nRowSubplot,nColSubplot] = subplot2(siz(dd),s);
            elseif s>1
                h_ax(s) = nexttile;
            else 
                h_ax(s) = gca;
            end
            setclim(clim);
            if ~isempty(levels{dd})
                titl = shortlabel(varnames{dd}, levels{dd}{s}); %title
            end
            if strcmpi(plottype,'xy')
                ph = pcurve(X, M(:,:,:,s), L(:,:,:,s), U(:,:,:,s),pval(:,:,:,s), varnames(1:3), levels(1:3));
            else
                ph = pcurve(X, M(:,:,s), L(:,:,s), U(:,:,s),pval(:,:,s), varnames(1:2), levels(1:2));
            end
            % if ~all(isnan(Yerr(:))) && all(all(isnan(Yerr(:,:,s)))),
            %     ph.error = [];
            % end
            plothandle(s) = ph;
            if s>1
                legend(gca,'off');
            end

            if layout
                if mod(s,nColSubplot)~=1
                    ylabel('');
                    set(gca, 'yticklabel',{});

                end
                if ceil(s/nColSubplot)<nRowSubplot
                    xlabel('');
                    set(gca, 'xticklabel',{});
                end
                axis tight;
            end
        end
        sameaxis(h_ax);

        %% 4 dimensions: different panels
    case 4 %
        dd = 3+is_xy;
        ee = 4+is_xy;

        for s1 = 1:siz(dd)
            for s2 = 1:siz(ee)
                s = s1+s2*(siz(dd)-1);
                if layout
                    subplot(siz(ee), siz(dd), s1 + siz(dd)*(s2-1));
                elseif s1*s2>1
                    h_ax(s) = nexttile;
                    else 
                h_ax(s) = gca;
                end

                setclim(clim); % colour limits
                titl = [shortlabel(varnames{dd}, levels{dd}{s1}) ', ' shortlabel(varnames{ee}, levels{ee}{s2})]; % title

                % plot
                if is_xy
                    plothandle(s1,s2) = pcurve(X, M(:,:,:,s1,s2), L(:,:,:,s1,s2), U(:,:,:,s1,s2), pval(:,:,:,s1, s2), ...
                        varnames(1:3), levels(1:3));
                else
                    plothandle(s1,s2) = pcurve(X, M(:,:,s1,s2), L(:,:,s1,s2), U(:,:,s1,s2), pval(:,:,s1, s2), ...
                        varnames(1:2), levels(1:2));
                end
                if s1>1 || s2>1
                    legend off;
                end
                if layout
                    if s1>1
                        ylabel('');
                        set(gca, 'yticklabel',{});

                    end
                    if s2<siz(4)
                        xlabel('');
                        set(gca, 'xticklabel',{});
                    end
                end
            end
            axis tight;
        end
        sameaxis;
end

if ~is_hold
    hold off;
end

warning(imaginary_warning.state,'MATLAB:plot:IgnoreImaginaryXYPart'); % turn warning for imaginary data back to previous value

switch nargout
    case 0
        varargout = {};
    case 1
        varargout = {M};
    case 2
        varargout = {M,L};
    case 3
        varargout = {M, L, plothandle};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% SUBFUNCTIONS %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plotting subfunction
    function phandle = pcurve(X, MM, LL, UU, PP, vnames, labelz)

        X = X(:)';
        hold on;

        phandle = struct('axis',gca);

        %% compute probability values to display
        %         if isnumeric(PP),
        %             if length(PP)==1,
        %                 PP = struct('FF1', Inf(1,nbv), 'FF2', Inf(1,nbp));  %dont' draw any signifchar
        %             else
        %                 PP = struct('FF1', PP, 'FF2', Inf);
        %             end
        %         else
        %             try
        %                 PP.FF1 = PP.(vnames{1});
        %             catch
        %                 warning('wu:probdisplay', 'prob displaying did not work');
        %             end
        %             try
        %                 PP.FF2 = PP.(vnames{2});
        %             catch
        %                 warning('wu:probdisplay', 'prob displaying did not work');
        %             end
        %         end
        %         if nbv == 2, %de-squeeze
        %             PP.FF2 = reshape(PP.FF2, [1 1 nbp]);
        %         end
        %         if nbp == 2,
        %             PP.FF1 = reshape(PP.FF1, [1 1 nbv]);
        %         end

        %% lables for legend
        add_legend = length(labelz)>1  && ~isempty(labelz{2}) && ~all(cellfun(@isempty, labelz{2})); %except when only one curve and no attached label
        if  add_legend
            legend_labels = labelz{2+strcmp(plottype,'xy')};
            %legend_labels = cell(1, nbv);
            %for w = 1:nbv
            % forlegend{w} = shortlabel(factoz{3}, labelz{2}{w});
            %   legend_labels{w} = labelz{2}{w};
            %end
        end

        %%%%%%%%%  BARS %%%%%%
        switch plottype
            case 'bar'

                phandle = mybar(X(:), MM, LL, UU, 'facecolor', cor, errorbarpars{:});
                phandle.mean = phandle.bar;
                phandle = rmfield(phandle, 'bar');

                %                 %display probability values
                %                 if nbv ==2 && all(~isinf(PP.FF2))  % same x-values
                %                     phandle.signif = [];
                %                     for p=1:nbp
                %                         if PP.FF2(1,1,p)<.1,
                %                             if (any(MM(p,:)>0)),
                %                                 hi = max(MM(p,:)+1.1*EE(p,:));
                %                             else
                %                                 hi = min(MM(p,:) - 1.1*EE(p,:));
                %                             end
                %                             phandle.pvalue(p) = text(p, hi , signifchar(PP.FF2(1,1,p)), 'FontSize', 24, 'HorizontalAlignment', 'center');
                %                         end
                %                     end
                %                 end
                %                 if nbp==2 && all(~isinf(PP.FF1))  % between x-values
                %                     phandle.pvalue=[];
                %                     whichcor = 1 + round((size(cor,1)-1) * (0:nbv-1)/(nbv-1));
                %                     lightcolour = cor(whichcor,:);
                %                     for w=1:nbv
                %                         if PP.FF1(1,1,w)<.1
                %                             phandle.pvalue(p) = text(1.5, max(MM(:,p)), signifchar(PP.FF1(1,1,w)), 'Color', lightcolour(w,:), ...
                %                                 'FontSize', 24, 'HorizontalAlignment', 'center');
                %                         end
                %                     end
                %                 end

                %                 if add_legend
                %                     if strcmpi(legend_style, 'standard')
                %                         hleg = legend(legend_labels{:}, 'Location', 'Best');
                %                     elseif strcmpi(legend_style, 'color')
                %                         hleg = legend_color(legend_labels{:}, 'Location', 'Best');
                %
                %                     end
                %                 end
                %                 if ~isempty(vnames{2})
                %                     hlegtitle = get(hleg,'title');
                %                     set(hlegtitle,'string',vnames{2});
                %                     set(hlegtitle, 'fontsize',get(gca, 'fontsize'));
                %                     set(hlegtitle, 'Position', [0.1 1.05304 1]);
                %                     set(hlegtitle, 'HorizontalAlignment', 'left');
                %                 end
                %legend boxoff


                %% %%%% LINES %%%%%%%%%
            case 'curve'
                %                 if nbp==2 && all(~isinf(PP.FF1)),
                %                     phandle.pvalue = [];
                %                 end


                % 'fill' option: compute vector for surface
                if any(strcmp(errorstyle, {'fill', 'area', 'marge'}))
                    Xfill = [X X(end:-1:1)];                                                %abscissa for the surface
                    Yfill = [ MM-LL ; flipud(MM+UU) ]; % ordinates for the surface
                end

                % plot errors
                if ~all(isnan(LL(:))) || ~all(isnan(UU(:)))
                    for w=1:nVar  % corresponding to each different curve
                        XX = X + shift(w);
                        switch lower(errorstyle)
                            case {'bar', 'bars', 'errorbar', 'errorbars'} %error bars
                                if strcmp(maxis, 'y') % vertical bars
                                    phandle.error(:,w) = myerrorbar(XX, MM(:,w), LL(:,w), UU(:,w),'vertical', 'color', cor{w}, errorbarpars{:},'none');
                                else % horizontal bars
                                    phandle.error(:,w) = myerrorbar(MM(:,w), XX,  LL(:,w), UU(:,w),'horizontal', 'color', cor{w}, errorbarpars{:},'none');
                                end

                            case  {'fill', 'area', 'marge'} %error areas
                                nonnan = ~isnan(Yfill(:,w));
                                XY = {Xfill(nonnan)+shift(w), Yfill(nonnan,w)};
                                if strcmp(maxis, 'x')
                                    XY = XY([2 1]);
                                end
                                if any(nonnan)
                                    phandle.error(:,w) = fill( XY{:}, lightcolour{w}, 'LineStyle', errorlinestyle);
                                    set(phandle.error(:,w),'FaceAlpha',.5);
                                else
                                    phandle.error(:,w) = gobjects(1);
                                end
                            case {'curve', 'curves', 'errorcurve', 'errorcurves'} %error lines
                                XY = {XX, [MM(:,w)-LL(:,w) MM(:,w)+UU(:,w)]};
                                if strcmp(maxis, 'x')
                                    XY = XY([2 1]);
                                end
                                %    phandle.error(:,w) = plot( XX, [MM(:,w)-LL(:,w) MM(:,w)+UU(:,w)], 'LineWidth', linewidth/2, 'Color', cor(w,:), 'LineStyle', errorlinestyle );
                                % else
                                phandle.error(:,w) = plot( XY{:} ,'LineWidth', linewidth(w)/2, 'Color', cor{w}, 'LineStyle', errorlinestyle );
                                % end
                        end
                    end
                end

                for w = 1:nVar % for each variable
                    XX = X + shift(w);
                    XY = {XX, MM(:,w)};
                    if strcmp(maxis, 'x')
                        XY = XY([2 1]);
                    end

                    %plot curve
                    %  if strcmp(maxis, 'y')
                    phandle.mean(w) = plot(XY{:}, 'Color', cor{w}, 'Marker', marker(w), 'markersize', markersize(w), ...
                        'Linestyle', linestyle(w), 'linewidth', linewidth(w));
                    % end

                    %                     %plot significances character between 2 x-values
                    %                     if nbp==2 && PP.FF1(1,1,w)<.1
                    %                         phandle.signif(w) = text(mean(X(1:2)), max(MM(:,w)), signifchar(PP.FF1(1,1,w)), 'FontSize', 24, ...
                    %                             'Color', cor(w,:), 'HorizontalAlignment', 'center');
                    %                     end
                    %
                    %                     if nbp==2 && PP.FF1(1,1,w)<.1,    % add significance character to legend label
                    %                         labelz{2}{w} = [labelz{2}{w} '(' signifchar(PP.FF1(1,1,w)) ')' ];
                    %                     end
                end

                %                 % add legend
                %                 if add_legend
                %                     if strcmp(legend_style, 'standard')
                %                         legend(phandle.M, legend_labels, 'Location', 'Best');
                %                     else
                %                         legend_color(phandle.M, legend_labels); %, 'Location', 'Best');
                %                     end
                %                 end

                %%%%%%%%%% XY TYPE
            case 'xy'

                %add vertical and horizontal error bars
                %  barwidd = ticklength;
                for w=1:size(MM,3)
                    phandle.error(:,1,w) = myerrorbar(MM(1,:,w), MM(2,:,w), LL(2,:,w), UU(2,:,w), ...
                        'vertical','color',cor{w}, errorbarpars{:},'linestyle',linestyle(w));
                    phandle.error(:,2,w) = myerrorbar(MM(1,:,w), MM(2,:,w), LL(1,:,w), UU(1,:,w),...
                        'horizontal', 'color',cor{w}, errorbarpars{:},'linestyle',linestyle(w));
                end
                %horizontal error bar (add bar width property)
                %                 dem = herrorbar(MM(1,:), MM(2,:), EE(1,:), EE(1,:));
                %                 delete(dem(2)); %remove the line
                %                 set(dem(1), 'Color', cor(1,:));
                %                 phandle.E(2) = dem(1);

                %plot curve
                for w=1:size(MM,3)
                    phandle.mean(w) = plot(MM(1,:,w), MM(2,:,w), 'Color', cor{w}, 'Marker',marker(w), ...
                        'MarkerSize', markersize(w), 'Linestyle', linestyle(w), 'linewidth', linewidth(w));
                end

                % plot identity line
                if IdentityLine
                    xl = xlim; yl = ylim;
                    xx = [min(xl(1),yl(1)) max(xl(2),yl(2))];
                    phandle.identityline = plot(xx,xx,'color',.5*[1 1 1]);
                    uistack(phandle.identityline, 'bottom');
                end

                %%%%%%%%%% IMAGESC TYPE
            case 'imagesc'
                ylabl = cellfun(@str2double, labelz{2});
                if any(isnan(ylabl)) || isempty(ylabl)
                    ylabl = [1 nVar];
                end
                phandle = imagesc(X, ylabl, MM');
                axis tight;
        end

        %% plot significance lines
        stats_symbols = {'*','**','***'};
        if ~all(isnan(PP(:)))
            [Xord, i_ord] = sort(X); % sort values along X bar to get contiguous
            if length(X)>1
                Xord = [2*Xord(1)-Xord(2) Xord 2*Xord(end)-Xord(end-1)]; % extrapolate for point before first and point after last
            else % if just one point
                Xord = Xord + (-1:1);
            end

            phandle.signif = [];
            for w=1:nVar % for each line/bar series
                sig = (PP(:,w)<threshold)'; % values that reach significance

                if strcmpi(statsstyle, 'symbol')
                    n_threshold = length(threshold);
                    sigsum = sum(sig); %
                    if ~isfield(phandle, 'S'), phandle.S = []; end
                    for tt =1:n_threshold
                        idx = sigsum==tt;
                        if any(idx)
                            if strcmp(maxis, 'y')
                                hh = text(X(idx), y_pval(w)*ones(1,sum(idx)), stats_symbols{tt},  'color', cor{w},...
                                    'horizontalalignment','center','verticalalignment','middle');
                            else
                                hh = text(y_pval(w)*ones(1,sum(idx)), X(idx),  stats_symbols{tt},  'color', cor{w},...
                                    'horizontalalignment','center','verticalalignment','middle');
                            end
                            phandle.signif = [phandle.signif hh(:)'];
                        end
                    end
                else % plot significance line
                    sigord = [false sig(i_ord) false]; % re-order according to increasing X values, and add false values for inexisintg points 0 and end+1
                    Ysig = nan(1,length(Xord)); % Y values for points is nan (do not draw) by default
                    Ysig(sigord) = 1; % and non-nan only for significant values
                    Xsig = Xord;
                    singlepoints = 1 + find(sigord(2:end-1) & ~sigord(1:end-2) & ~sigord(3:end)); % significant points between two non significant points
                    for ss = fliplr(singlepoints) % for each single oiunt (we go in reverse order to avoid confusion between indices as size changes
                        Xsig = [Xsig(1:ss-1) Xsig(ss)-.1*diff(Xsig(ss-1:ss)) Xsig(ss)+.1*diff(Xsig(ss:ss+1)) Xsig(ss+1:end)]; % create a short segment around that point
                        Ysig = [Ysig(1:ss-1) 1                               1                               Ysig(ss+1:end)];
                    end

                    if strcmp(maxis, 'y')
                        phandle.signif(w) = plot(Xsig, y_pval(w)*Ysig, statsstyle, 'color', cor{w});
                    else
                        phandle.signif(w) = plot(y_pval(w)*Ysig, Xsig, statsstyle, 'color', cor{w});
                    end
                end
                set( phandle.signif, 'Tag', 'Significant');
            end
        end
        if isfield(phandle, 'mean')
            set( phandle.mean, 'Tag', 'mean');
        end
        if isfield(phandle, 'error')
            %   set( phandle.error, 'Tag', 'error');
        end


        %% add legend
        if add_legend
            if strcmpi(legend_style, 'standard')
                hleg = legend(phandle.mean, legend_labels{:}, 'Location', 'Best','AutoUpdate','off');
            elseif strcmpi(legend_style, 'color')
                hleg = legend_color(phandle.mean, legend_labels{:}, 'Location', 'Best','AutoUpdate','off');

            end

            if ~isempty(vnames{2})
                hlegtitle = get(hleg,'title');
                set(hlegtitle,'string',vnames{2});
                set(hlegtitle, 'fontsize',get(gca, 'fontsize'));
                %      set(hlegtitle, 'Position', [0.1 1.05304 1]);
                %      set(hlegtitle, 'HorizontalAlignment', 'left');
            end
        end

        %% axis labels and tick labels
        if strcmp(plottype, 'xy')
            if ~isempty(labelz{1}) && ~isempty(labelz{1}{1}) && ~isequal(str2double(labelz{1}{1}),1)
                xlabel( labelz{1}{1});
            end
            if length(labelz{1})>1 && ~isempty(labelz{1}{2}) && ~isequal(str2double(labelz{1}{2}),2)
                ylabel( labelz{1}{2});
            end
        else
            if ~isempty(y_label)    % add label for dependent variable
                if strcmp(maxis, 'y')
                    ylabel(y_label);
                else
                    xlabel(y_label);
                end
            end
            if ~isempty(vnames{1})      %add label to indep variable axis
                if strcmp(maxis, 'y')
                    xlabel(vnames{1});
                else
                    ylabel(vnames{1});
                end
            end

            %% place labels for x axis
            if VerticalLabel % if vertical labels over bars
                y_vertlabels =  MM + 1.1*UU;
                y_vertlabels(isnan(UU)) = MM(isnan(UU)) + .05*diff(eval([faxis 'lim']));
                if nPar>1 && nVar>1
                    y_vertlabels = nanmax(y_vertlabels,[],2);
                end

                if nPar>1

                    for p=1:nPar
                        if strcmp(maxis, 'y')
                            phandle.label(p) = text(X(p),y_vertlabels(p), labelz{1}(p), 'rotation',90);
                        else
                            phandle.label(p) = text(y_vertlabels(p), X(p), labelz{1}(p));
                        end
                    end

                else % bars
                    legend off;
                    for vv=1:nVar
                        if verLessThan('matlab', '8.4')
                            R_eb =get(get(phandle.M(vv),'children'),ref_field);
                            this_x = mean(R_eb([1 3],1));

                        else
                            this_x = X(1) + phandle.mean(vv).([upper(faxis) 'Offset']);
                        end
                        if strcmp(maxis, 'y')
                            phandle.label(vv) = text(this_x,y_vertlabels(vv), labelz{2}(vv), 'rotation',90);
                        else
                            phandle.label(vv) = text(y_vertlabels(vv), this_x, labelz{2}(vv));
                        end

                    end
                end

            elseif ~isempty(labelz{1}) && ~isdatetime(X) && ~isequal(num2strcell(X), labelz{1}(:)') && nPar < 10
                %% labels as ticklabels

                [xtickval,xorder] = unique(X);
                set(gca, [faxis 'Tick'], unique(xtickval(~isnan(xtickval))));
                if ~isempty(labelz{1})
                    set(gca, [faxis 'TickLabel'], labelz{1}(xorder));
                end
            end

            % rotate xtick labels by specified angle
            if xtickangle ~=0
                if verLessThan('matlab', '8.4') % use user-supplied function
                    rotateXLabels( gca, xtickangle);
                else   % use built-in property
                    set(gca, 'XTickLabelRotation', xtickangle);
                end

            end

        end

        %add title
        if isfield(PP, 'interaction')
            title([titl ' p=' num2str(PP.interaction)]);
        elseif ~isempty(titl)
            title(titl);
        end
    end

%%%%%%%%%%%%%

%%%%

    function str = shortlabel(fname, vname)
        if isempty(str2double(vname)) || isempty(fname)
            str = vname;
        else
            str = [fname '=' vname];
        end
    end

%% set colors limit
    function setclim(clim)
        if isequal(clim, 'auto')
            set(gca, 'CLimmode', 'auto');
        else
            set(gca,'Clim',clim);
        end
    end

end

% replacement functions if does not have stats toolbox
function m = nmean(x,d)
if exist('nanmean','file')
    m = nanmean(x,d);
else
    m = nan_mean(x,d);
end
end


function s = nstd(x,d)
if exist('nanstd','file')
    s = nanstd(x,0,d);
else
    s = nan_std(x,d);
end
end

%% list of all colormapstrings
function C = colormapstrings()
C =  {'parula','jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism',...
    'spring','autumn','summer','winter', 'bone','lines','flag'};
end

%% get values for xticks from xticlabels, if relevant
function X = get_xtick(X, Lvl,nPar, xtick)

if strcmp(xtick, 'normal') && ~isempty(Lvl) && isempty(X)

    if ~any(cellfun(@iscell, Lvl))
        % try to convert all labels into X value
        X = cellfun(@str2double, Lvl(:)');
        nanValues = isnan(X) & ~strcmpi(Lvl(:)','nan');
        if any(nanValues) || isempty(X) % if its fails, simply use first integers
            X = 1:nPar;
        end
    else % if cell array
        X = 1:nPar;
    end

elseif isempty(X)
    X = 1:nPar;
end
end

%% modify vectors of linespecs when collapsing 2nd and 3rd dim
function x = rep_for_collapse(x, siz)
if isrow(x) % changes along dim2
    x = repmat(x,siz(3));
else % changes along dim3
    x = repelem(x,siz(2));
end
end

%% interaction levels when collapsing 2nd and 3rd dim
function str =interaction_levels(str1,str2)
n1 = numel(str1);
str1 = repmat(str1(:),numel(str2),1);
str2 = repelem(str2(:),n1,1);
str = str1 + ", " + str2; %concatenate
end

