function [K, grad] = cov_periodic(x, HP, period)
% Periodic covariance function
% [K, grad] = cov_periodic(x, HP, period)

if nargin<2
    K = 2; % number of hyperparameters
    return;
end
%HPfull = [period 1]; % default values
%HPfull(which_par) = HP; % active hyperparameters

ell = exp(HP(1));
sf = exp(2*HP(2));

% use function from GPML
%[cov, dK] = covPeriodic([log(ell) log(period) log(sf)], x);

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
    %gradK = gradient(dK, size(cov,1), 3);

    %  P = sin(2*T).*cov;

    grad(:,:,1) = 4*S2.*K; % / ell; % grad w.r.t ell ( covPeriodic provides grad w.r.t log(ell))
    %  gradK2(:,:,2) = 2/ell^2*P.*T; % grad w.r.t sf ( covPeriodic provides grad w.r.t log(sf))
    grad(:,:,2) = 2*K; % / sf;

    % dhyp = [4*(S2(:)'*Q(:)); 2/ell^2*(P(:)'*T(:)); 2*sum(Q(:))];

    %grad(:,:,1) = gradK(:,:,1) / ell;
    %grad(:,:,2) = gradK(:,:,3) / sf;
   
end

end