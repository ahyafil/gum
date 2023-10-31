function HH = HPstruct_SquaredExp(HH, scale, HP, type, period, single_tau, basis, binning)

if strcmpi(type,'periodic')
    tau = period/4; % initial time scale: period/4
else
    % data point resolution (mean difference between two points)
    dt = zeros(1,size(scale,1));
    for ss=1:size(scale,1)
        dt(ss) = mean(diff(unique(scale(ss,:))));
    end
    % tau = dt; % initial time scale: mean different between two points
    span = max(scale,[],2)' - min(scale,[],2)'; % total span in each dimension
    tau =  sqrt(dt.*span); % geometric mean between the two
    tau = max(tau,span/10);
    if single_tau
        tau = mean(tau);
        span = max(span);
    end
end
nScale = length(tau);

HP = HPwithdefault(HP, [log(tau) 0]); % default values for log-scale and log-variance [tau 1 1];
HH.HP = HP;
if nScale>1
    HH.label = num2cell("log \tau"+(1:nScale));
else
    HH.label = {'log \tau'};
end
HH.label{end+1} = 'log \alpha';
HH.fit = true(1,nScale+1);


% upper and lower bounds on hyperparameters
if strcmp(basis, 'fourier')
    HH.LB(1:nScale) = log(2*tau/length(scale)); % lower bound on log-scale: if using spatial trick, avoid scale smaller than resolution
    HH.LB(nScale+1) = max(-max_log_var, log(dt)-3); % to avoid exp(HP) = 0
    HH.UB = [log(span)+2 max_log_var];  % scale not much larger than overall span to avoid exp(HP) = Inf
HH.type = repmat("basis_cov",1,nScale+1);

else
    HH.UB(1:nScale) = min(log(101*tau),log(span)+2); %log(5*tau); % if not using spatial trick, avoid too large scale that renders covariance matrix singular
    HH.UB(nScale+1) = max_log_var;
    if ~isempty(binning)
        HH.LB = [log(binning)-2 -max_log_var];
    else
        HH.LB = [max(-max_log_var,log(dt)-2) -max_log_var];  % to avoid exp(HP) = Inf
    end
    HH.type =repmat("cov",1,nScale+1);

end
end