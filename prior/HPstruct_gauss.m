function HH = HPstruct_gauss(HH, scale,HP, nFun,params)
% HP structure for gaussian basis functions

% define mean hyperparameters
D = size(scale,1);

if params.samemean
    n_mean_HP = D;
else
    n_mean_HP = D*nFun;
end
n_HP = n_mean_HP+nFun;


% data point resolution (mean difference between two points)
dt = zeros(1,size(scale,1));
for ss=1:size(scale,1)
    dt(ss) = mean(diff(unique(scale(ss,:))));
end
log_dt = log(dt);

% total span in each dimension
max_scale = max(scale,[],2)';
min_scale = min(scale,[],2)';
span =  max_scale- min_scale;
%tau = sqrt(mean(dt.*span)); % geometric mean between the two
%  span = max(span);
%  log_dt = min(log_dt);
span = min(span);

% max and min value of mean HPs
mu_lb = 2*min_scale-max_scale;
mu_ub = 2*max_scale-min_scale;

if params.samemean
    % if all functions have same mean, then vary std
    mu = median(scale,2)'; % median values of scale

    % log std parameters
    log_tau = log(span/3)-(0:nFun-1)/2;
else
    % if  functions have different mean, then same std by default
    pctile = linspace(0,1,nFun+2);
    pctile = pctile(2:end-1);
    mu = quantile(scale',pctile); % percentile values for mean
    mu = mu(:)';

    % log std parameters
    tau = span/4/nFun;
    log_tau = repmat(log(tau),1,nFun);

    mu_lb= repmat(mu_lb,1,nFun);
    mu_ub= repmat(mu_ub,1,nFun);
end
if isfield(HP,'mu')
    mu = HP.mu;
end

% set up hyperparameter values
HH.HP = [mu log_tau]; % default values for mean and std term

% upper and lower bounds
HH.LB = [mu_lb -max_log_var/2*ones(1,nFun)]; % lower bound: to avoid exp(2*HP) = 0
HH.UB = [mu_ub max_log_var/2*ones(1,nFun)]; % lower bound: to avoid exp(2*HP) = Inf

% labels and types
HH.fit = true(1,n_HP);
if n_mean_HP ==1
    HH.label = "\mu";
elseif D==1 || nFun==1 || params.samemean
    HH.label = "\mu_"+(1:n_mean_HP);
else
    DD = repmat(1:D,1,nFun);
    bb = repelem(1:nFun,1,D);
    HH.label = ("\mu_"+DD)+bb;
end
if nFun==1
    HH.label(end+1) = "\log \tau";
else
    HH.label(end+1:end+nFun) = "\log \tau"+(1:nFun);
end
HH.type = repmat("basis",1,n_HP);
end