function [s,LLH] = compute_dispersion_parameter_neg_binomial(T, Y, w, s_ini)
%  compute_dispersion_parameter_neg_binomial() computes the dispersion
%  parameter for negative binomial regression using maximum likelihood
%  estimate.

% initial value of parameter
if nargin<4 || isnan(s_ini)
    s_ini = 0.1;
end

% function to minimize
f = @(s) negLLH(s, T, Y, w);

% find minimizing parameter
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','none');
LB = 1e-10;
s = fmincon(f, s_ini,[],[],[],[],LB, [],[],options);

LLH = -f(s);
end

% negative LLH function
function [nL, nGrad] = negLLH(s, T, Y, w)


% compute log-likelihood for each observation
one_sY = 1+s*Y;
log_one_sY = log(one_sY);
r = 1/s;
lh = T.*log(s*Y./one_sY) - r*log_one_sY + gammaln(T+r) - gammaln(T +1) - gammaln(r);

% sum over observations
if isempty(w)
    L = sum(lh);
else % observation weight
    L = sum(w .* lh); % LLH
end

nL = -L; % neg-LLH

if nargout>1
    grad = sum(log_one_sY + s*(T-Y)./one_sY + psi(r) - psi(T+r)) *r^2;
    nGrad = -grad;
end
end
