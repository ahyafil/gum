function [K, grad] = ard_covfun(scale, loglambda, B)
% Automatic Relevance Discrimination (ARD) covariance function
%  [K, grad] = ard_covfun(scale, loglambda, B)

nReg = length(scale);

if nargin<2
    K = nReg; % number of hyperparameters
    return;
end

lambda2 = exp(2*loglambda); % variance for each regressor
K = diag(lambda2);

% Jacobian
G = zeros(nReg,nReg,nReg);
for i=1:nReg
    G(i,i,i) = 2*lambda2(i);
end
grad.grad = G;

% M-step of EM to optimize hyperparameters given posterior on weights
grad.EM = @(m,V) log(m(:).^2 + diag(V))/2;

end