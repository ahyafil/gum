function HH = HPstruct_L2gamma(HH, scale,nFun)
% HP structure for L2 prior with gamma basis functions

log_min_tau = log(min(diff(scale)));
log_max_tau = log(scale(end)-scale(1));

if nFun==1
    logtau = (log_min_tau+log_max_tau)/2;
    alpha = exp(log_max_tau)-exp(logtau); % required std for basis function
else
    logtau = linspace(log_min_tau,log_max_tau,nFun); % default tau values: logarithmically spaced (tau = k*theta)
    alpha = exp(logtau(2)-logtau(1)); % required std for basis function
end

tau = exp(logtau); % time scale
k = ones(1,nFun) / sqrt(alpha-1); % shape parameter computing for 'optimal' tiling of space (same for all funs)
k = max(k,1.2);
theta = tau ./ k; % scale parameter

if any(scale==0)
    theta_LB = ones(1,nFun);
else
    theta_LB = zeros(1,nFun);
end

HH.HP = [theta k  0];
HH.LB = [zeros(1,nFun) theta_LB -max_log_var];
HH.UB = [inf(1,2*nFun)    max_log_var];
HH.fit = true(1,2*nFun+1);
if nFun ==1
    HH.label = ["\theta","k","\log \alpha"];
else
    HH.label = ["\theta_"+(1:nFun),"k_"+(1:nFun),"\log \alpha"];
    HH.type = [repmat("basis",1,2*nFun),"cov"];

end
end