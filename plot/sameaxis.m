function varargout = sameaxis(handle, ranger)
%sameaxis(fig, [xmin xmax ymin ymax]) sets the same axis for all subplots of figure fig
%sameaxis([],[xmin xmax ymin ymax]) applies to gcf
%sameaxis(aa, [xmin xmax ymin ymax]) where aa is collection of handles for
%axes
%set xmin and/or ymin to -Inf if you do not wish to do this along the associated axis
%set xmi and/or ymin to +Inf in order to automatically adapt
%
%axlim = sameaxis
%
%See also sameclim

if nargin<1,
    handle = [];
end
if nargin<2,
    ranger = [];
end

if isempty(handle),
handle = gcf;
end
if isempty(ranger)
ranger = [Inf 0 Inf 0];
end


% for n=1:nargin,
% 	if length(varargin{n})>1,
% 		ranger = varargin{n};
% 	else handle = varargin{n};
% 	end
% end

%get the handles for subplots
%calcule nbh et nbv
if strcmp(get(handle,'type'),'figure'),
axes = get(handle, 'Children');
else
   axes = handle; 
end
if iscell(axes), axes = [axes{:}]; end

axesh = findobj(axes, 'Type', 'axes'); %select only axes objects


%exclude legend boxes
islegend = strcmp( get(axesh, 'tag'), 'legend');
axesh(islegend) = [];
%whichlegend = multfind( get(child, 'Tag'), 'legend');
%child(whichlegend) =[];

saxis = [];
for s=1:length(axesh)
  %  saxis(s,:) = axis(child(s));
      saxis(s,:) = [get(axesh(s), 'xlim') get(axesh(s), 'ylim')];
end

if ranger(1) == Inf,
	ranger(1:2) = [min(saxis(:,1)) max(saxis(:,2))];
end
if ranger(3) == Inf,
	ranger(3:4) = [min(saxis(:,3)) max(saxis(:,4))];
end


for s=1:length(axesh)
	if ranger(1)> -Inf,
		saxis(s,1:2) = ranger(1:2);
	end
	if ranger(3) > -Inf,
		saxis(s,3:4) = ranger(3:4);
	end
	axis(axesh(s), saxis(s,:));
end

if nargout > 0,
    varargout = {ranger};
end
    