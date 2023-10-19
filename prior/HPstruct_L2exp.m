function HH = HPstruct_L2exp(HH, scale, HP, nExp)
% Hyperparameter structure for L2 with exponential basis functions

log_min_tau = log(min(diff(scale)));
log_max_tau = log(scale(end)-scale(1));
logtau = linspace(log_min_tau,log_max_tau,nExp); % default tau values: logarithmically spaced
HP = HPwithdefault(HP, [logtau 0]); % default values for log-scale and log-variance [tau 1 1];
HH.HP = HP;
HH.LB = [(log_min_tau-2)*ones(1,nExp)  -max_log_var];
HH.UB = [(log_max_tau+5)*ones(1,nExp)  max_log_var];

if nExp>1
    HH.label = [num2cell("log \tau"+(1:nExp)), {'\log \alpha'}];
else
    HH.label = {'log \tau','\log \alpha'};
end
HH.fit = true(1,nExp+1);
HH.type = [repmat("basis",1,nExp),"cov"];
end