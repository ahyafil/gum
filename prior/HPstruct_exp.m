function H = HPstruct_exp(H, scale, HP, nExp)
% Hyperparameter structure exponential basis functions

log_min_tau = log(min(diff(scale)));
log_max_tau = log(scale(end)-scale(1));
logtau = linspace(log_min_tau,log_max_tau,nExp); % default tau values: logarithmically spaced

% set hyperparameter values
H.HP = logtau; % default values for log-scale [tau];

if isfield(HP,'tau')
    H.HP(1:nExp) = log(HP.tau);
end

% upper and lower bounds
H.LB = (log_min_tau-2)*ones(1,nExp);
H.UB = (log_max_tau+5)*ones(1,nExp);

if nExp>1
    H.label = "log \tau"+(1:nExp);
else
    H.label = "log \tau";
end
H.fit = true(1,nExp);
H.type = repmat("basis",1,nExp);
end