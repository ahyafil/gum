function HH = HPstruct_gamma(HH, scale,HP, nFun)
% HP structure for gamma basis functions

% HP here is only for logtau
assert(isrow(scale),'Gamma basis functions for one-dimensional function only');

log_min_tau = log(min(diff(scale))); % mininum difference between consecutive points
log_max_tau = log(scale(end)-scale(1)); % overall span

if nFun==1
    logtau = (log_min_tau+log_max_tau)/2;
    alpha = exp(log_max_tau)-exp(logtau); % required std for basis function
else
    logtau = linspace(log_min_tau,log_max_tau,nFun); % default tau values: logarithmically spaced (tau = k*theta)
    alpha = exp(logtau(2)-logtau(1)); % required std for basis function
end
%if ~isempty(HP)
%    assert(length(HP)==nFun, 'number of tau values does not match number of gamma basis functions');
%    logtau = HP;
%end

% default values
tau = exp(logtau); % time scale
if isfield(HP,'tau')
    tau = HP.tau;
end

k = ones(1,nFun) / sqrt(alpha-1); % shape parameter computing for 'optimal' tiling of space (same for all funs)
k = max(k,1.2);
if isfield(HP,'k')
    k = HP.k;
end

theta = tau ./ k; % scale parameter

% set up hyperparameter values
HH.HP = [theta k]; % default values

% upper and lower bounds
if any(scale==0)
    theta_LB = ones(1,nFun);
else
    theta_LB = zeros(1,nFun);
end
HH.LB = [zeros(1,nFun) theta_LB];
HH.UB = [inf(1,2*nFun)];

% labels and types
HH.fit = true(1,2*nFun);
if nFun ==1
    HH.label = ["\theta","k"];
else
    HH.label = ["\theta_"+(1:nFun),"k_"+(1:nFun)];
    HH.type = repmat("basis",1,2*nFun);
end
end