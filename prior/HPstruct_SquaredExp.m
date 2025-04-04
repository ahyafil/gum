function HH = HPstruct_SquaredExp(HH, scale, HP, type, period, single_tau, basis, binning)
% creates hyperparameter structure for squared exponential kernel (and
% periodic kernel as well)

% data point resolution (mean difference between two points)
dt = zeros(1,size(scale,1));
for ss=1:size(scale,1)
    dt(ss) = mean(diff(unique(scale(ss,:))));
end
log_dt = log(dt);

% total span in each dimension
span = max(scale,[],2)' - min(scale,[],2)';

if strcmpi(type,'periodic')
    tau = period/6; % initial time scale: period/6 (i.e. pi/3 for 2pi)
    assert(isrow(scale),'periodic functions only defined over one dimension scales');
else
    tau =  sqrt(dt.*span); % geometric mean between the two
    tau = max(tau,span/10);
    if single_tau
        tau = mean(tau);
        span = max(span);
        log_dt = min(log_dt);
    end
end
nScale = length(tau);

%% set value of hyperparameters
HH.HP = [log(tau) 0]; % default values for log-scale and log-variance [tau 1 1];
if isfield(HP, 'value') && ~isempty(HP.value)
    assert(length(HP.value)==length(HH.HP), 'incorrect number of hyperparameters for Squared Exponential kernel');
    HH.HP = HP.value;
else
    if isfield(HP,'tau')
        HH.HP(1:nScale) = log(HP.tau);
    end
    if isfield(HP,'variance')
        HH.HP(end) =log(HP.variance)/2;
    end
end

%% labels
if nScale>1
    HH.label = num2cell("log \tau"+(1:nScale));
else
    HH.label = {'log \tau'};
end
HH.label{end+1} = 'log \alpha';
HH.fit = true(1,nScale+1);

%% upper and lower bounds on hyperparameters
if strcmp(basis, 'fourier')
    HH.LB(1:nScale) = log(2*tau/length(scale)); % lower bound on log-scale: if using spatial trick, avoid scale smaller than resolution
    HH.LB(nScale+1) = max(-max_log_var, min(log_dt)-3); % to avoid exp(HP) = 0
    HH.UB = [log(span)*log(2) max_log_var];  % scale not much larger than overall span to avoid exp(HP) = Inf
    HH.type = repmat("basis_cov",1,nScale+1);
else
    if ~strcmpi(type,'periodic')
        HH.UB(1:nScale) = min(log(101*tau),log(span)+2); %log(5*tau); % if not using spatial trick, avoid too large scale that renders covariance matrix singular
    else
        HH.UB(1:nScale) = log(period)+2;
    end
    HH.UB(nScale+1) = max_log_var;
    if ~isempty(binning)
        HH.LB = [log(binning) -max_log_var];
    else
        HH.LB = [max(-max_log_var,log_dt-2) -max_log_var];  % to avoid exp(HP) = Inf
    end
    HH.type =repmat("cov",1,nScale+1);

end
end