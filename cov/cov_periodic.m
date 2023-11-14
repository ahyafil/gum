function [K, grad] = cov_periodic(x, HP, B, period)
% Periodic covariance function
% [K, grad] = cov_periodic(x, HP, B, period)

if nargin<2
    K = 2; % number of hyperparameters
    return;
end

ell = exp(HP(1));
sf = exp(2*HP(2));

%adapted from GPML (see 4.31 from Rasmussen & Williams)
within = ~iscell(x);
if within
    x = {x,x};
end
T = pi/period*bsxfun(@plus,x{1},-x{2}');
S2 = (sin(T)/ell).^2;

K = sf*exp( -2*S2 );
K = force_definite_positive(K);


if nargout>1
    % turn dK into gradient tensor (weight x weight x HP)
    grad(:,:,1) = 4*S2.*K; % / ell; % grad w.r.t ell ( covPeriodic provides grad w.r.t log(ell))
    grad(:,:,2) = 2*K; % / sf;
end

end