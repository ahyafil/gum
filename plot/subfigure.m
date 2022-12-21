function varargout = subfigure(varargin)
% Create a figure within a grid-based layout. It's subplot for figures.
%
% subfigure(m,n,p), or subfigure(mnp), divides the screen into an m-by-n grid of
% tiles and creates a figure within the pth tile. Tiles are counted along the
% top row of the screen, then the second row, etc.
%
% If p is a vector, the figure is sized to cover all the subfigure positions
% listed in p.
%
% subfigure(m,n) creates a figure showing a m-by-n grid layout with tiles
% labeled in the order that they are numbered by subfigure. This is useful for
% planning screen layouts, especially when one or more subfigures will span
% multiple tiles (when p is a vector).
%
% h = subfigure(...) returns a handle to the figure.
%
% Every call to subfigure creates a new figure even if a figure exists at the
% location specified by m, n, and p. The existing figure is not made current or
% reused. Existing figures that are overlapped by new subfigures are not
% deleted. This behavior is dissimilar to subplot.
%
% Example 1: Four non-overlapping figures.
%
%     subfigure(2,2,1)
%     subfigure(2,2,2)
%     subfigure(2,2,3)
%     subfigure(2,2,4)
%
% Example 2: Three non-overlapping figures of various sizes.
%
%     subfigure(4,4,[1 13])
%     subfigure(4,4,[2 4])
%     subfigure(4,4,[6 16])
%
% Example 3: Show the grid for a 3 x 5 layout.
%
%     subfigure(3,5)
%
%
%  mnp = subfigure('pos') gives current position
% subfigure('reset') to go back to 0
%
% subfigure(..., title)

% by Steve Hoelzer
% 2009-03-25
%modified by alex :
%

% Padding to allow room for figure borders and menus
hpad = 20;
vpad = 90;

persistent m n p

%%process input

%subfigure('pos') and subfigure('reset') syntax
if nargin ==1 && ischar(varargin{1})
    if strcmp(varargin{1}, 'pos')
        varargout{1} = [m,n,p];
        return;
    elseif strcmp(varargin{1}, 'reset')
        if isempty(m)
            m = 4;
            n = 6;
        end
        p = 0;
        return
    end
end

%how many input arguments for figure position
isnumericarg = [cellfun(@isnumeric, varargin) 0];
nargin_pos = find(~isnumericarg,1) - 1;

% Process input arguments for position
switch nargin_pos
    case 0
        if isempty(m) %m,n,p did not exit
            m = 4;
            n = 6;
            p = 1;
        else
            p = mod(p,m*n); %starts back to 0 if end of the list
            p = p+1;
            
        end
    case 1
        
        if varargin{1}>100
            m = floor(varargin{1}/100);
            n = rem(floor(varargin{1}/10),10);
            p = rem(varargin{1},10);
        else
            m = 4;
            n = 6;
            p = varargin{1};
        end
        
    case 2
        m = varargin{1};
        n = varargin{2};
        p = [];
    case 3
        m = varargin{1};
        n = varargin{2};
        pp = varargin{3};
        if (pp==0)
            p = mod(p,m*n); %starts back to 0 if end of the list
            p = p +1;
        else
            p = pp;
        end
    otherwise
        error('Too many input arguments for figure position')
end

% Error checking
if ~isscalar(m) || ~isscalar(n)
    error('Gird dimensions must be scalar values.')
end
if m < 0 || n < 0
    error('Grid dimensions must be greator than zero.')
end
if any(p > m*n)
    error('Position value exceeds grid size.')
end

% process other argument inputs
varargin(1:nargin_pos) = [];
nargin_prop = length(varargin);
if nargin_prop >=1  %figure title
    mytitle = varargin{1};
else
    mytitle = '';
end
if nargin_prop >=2 % figure file name
    myfilename = varargin{2};
else
    myfilename = strrep(mytitle, ' ', '_');
end


%% compute figure position
if isempty(p)
    % Draw example grid using subplots
    p = m*n;
    f = figure('NumberTitle','Off',...
        'Name',sprintf('Subfigure tile numbering for a %i by %i grid',m,n));
    for i = 1:p
        h = subplot(m,n,i);
        set(h,'Box','On',...
            'XTick',[],...
            'YTick',[],...
            'XTickLabel',[],...
            'YTickLabel',[])
        text(0.5,0.5,int2str(i),...
            'FontSize',16,...
            'HorizontalAlignment','Center')
    end
    % Return handle if needed
    if nargout
        varargout{1} = f;
    end
    return
end


% Calculate tile size and spacing
scrsz = get(0,'ScreenSize');

if ~(scrsz(3) == 1) %if figure display is enabled
    hstep = floor( (scrsz(3)-hpad) / n );
    vstep = floor( (scrsz(4)-vpad) / m );
    vsz = vstep - vpad;
    hsz = hstep - hpad;
    
    % Row and column positions of each subfigure in p
    r = ceil(p/n);
    c = rem(p,n);
    c(c==0) = n; % Special case
    
    % Position of each subfigure in p in pixels (left, right, bottom, top)
    le = hpad + (c-1)*hstep;
    ri = le + hsz;
    bo = vpad + (m-r)*vstep;
    to = bo + vsz;
    
    % Position of a subfigure that covers all subfigures in p
    le = min(le); % Leftmost left
    ri = max(ri); % Rightmost right
    bo = min(bo); % Lowest bottom
    to = max(to); % Highest top
    
    % Calculate figure position
    pos = [le, bo, ri-le, to-bo]; % [left, bottom, width, height]
    
    
else  %if figure display is disabled
    pos = [100 100 660 520];
end

% Display figure
h = figure('Position',pos);

%box off
box off
set(gca, 'Fontsize', 16);

% Return handle if needed
if nargout
    varargout{1} = h;
end

%% process figure properties
if isempty(mytitle)
    set(gcf, 'NumberTitle', 'on');
else
    set(gcf, 'NumberTitle', 'off');
    set(gcf, 'Name', mytitle);
    set(gcf, 'FileName', [char(myfilename) '.fig']);
end