function [h, CC] = colorplot(varargin)
%colorplot(X, Y, C) or colorplot(Y,C) plots scatter plots where color of
%each point is defined by value in the corresponding vector C
%
%colorplot(...., 'cmap', cmap) where cmap is a colormap
%h = colorplot(...) returns the plot handles
%
%colorplot(...., 'cmap', cmap, 'clim', [Cmin Cmax]) defines limits for
%colormap
%
%Data points with NaN values in C are not plotted
%Adds C value for each point as custom cursor data (requires
%customDataCursor.m)
%
%See also grpplot, colormap

% create it with object
%merge with graf (use 'markerstyle', 'normalize')

if nargin<3 || ischar(varargin{3}) %colorplot(Y, C, ...) syntax
    Y = varargin{1};
    C = varargin{2};
    X = 1:length(Y);
    narg = 3;
else                            %colorplot(X, Y, C, ...) syntax
    X = varargin{1};
    Y = varargin{2};
    C = varargin{3};
    narg = 4;
end

%Nan values
isn = isnan(C);
if any(isn)
    C(isn) = [];
    X(isn) = [];
    Y(isn) = [];
end

if isempty(C)
    h = [];
    CC = [];
    return;
end

%default values
cmap = colormap; %use current colormap
clim = [];

%optional arguments
while (nargin>=narg)
    switch varargin{narg}
        case 'cmap'
                cmap = varargin{narg+1};   %user-defined colormap
                 narg = narg + 1;
        case 'clim'
                clim = varargin{narg+1};   %user-defined color limit values
                 narg = narg + 1;
        otherwise
            error('unknown parameter #%d', narg);
    end
   narg = narg + 1; 
end

%normalize C
C = C(:);
ncor = size(cmap,1); %number of colors in colormap

if isempty(clim)
Cmin = min(C);
Cmax = max(C);
if Cmax==Cmin
   Cmax = Cmin+1;  
end
else
    Cmin = clim(1);
    Cmax = clim(2);
end
Cnorm = (ncor-1)*(C-Cmin)/(Cmax-Cmin);  %the value is now between 0 and ncor

%derive color points
cmap(end+1,:) = cmap(end,:); %just a trick (for the interpolation part)
Cint = 1+floor(Cnorm); %integer part
Cdig = repmat(mod(Cnorm,1), 1,3); % decimal part
CC = (1-Cdig).*cmap(Cint,:) + Cdig.*cmap(Cint+1,:); %the color is linearly interpolated between two data points in the colormap

%plot
N = length(Y);
h = zeros(1,N);
hold on;
for i = 1:N
    h(i) = plot(X(i), Y(i), '.','markeredgecolor', CC(i,:));
end
set(gca, 'clim', [Cmin Cmax]);

%set custom data cursor
try %in case value does not exist
    for i=1:N
        customDataCursor(h(i),{C(i)});
    end
end