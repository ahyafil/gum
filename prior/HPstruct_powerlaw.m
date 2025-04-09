function HH = HPstruct_powerlaw(HH, scale,HP, nFun,varargin)
% HP structure for power law basis functions

% HP here is only for logtau
assert(isrow(scale),'Power law basis functions for one-dimensional function only');

% default values
f = floor(nFun/2);
if mod(nFun,2) % odd number of basis functions
    alpha = [1./(f:-1:1) 1 1:f];
else
    alpha = [1./(f-.5:-1:-5) -5:f+.5];
end
if isfield(HP,'alpha')
    alpha = HP.alpha;
end

% set up hyperparameter values
HH.HP = alpha; % default values

% upper and lower bounds
HH.LB = zeros(1,nFun);
HH.UB = inf(1,nFun);

% labels and types
HH.fit = true(1,nFun);
if nFun ==1
    HH.label = "\alpha";
else
    HH.label = "\alpha_"+(1:nFun);
end
HH.type = repmat("basis",1,nFun);
end