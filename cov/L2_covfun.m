function [K, grad] = L2_covfun(scale, loglambda, B)
% L2-regression covariance function
% [K, grad] = L2_covfun(scale, loglambda, B)
if nargin<2
    K = 1; % number of hyperparameters
    return;
end

nReg = length(scale);
lambda2 = exp(2*loglambda);
if isinf(lambda2)
    K = diag(lambda2 * ones(1,nReg));
else
    K = lambda2 * eye(nReg); % diagonal covariance
end

grad.grad = 2*K; %eye(nreg);
grad.EM = @(m,Sigma) log(mean(m(:).^2 + diag(Sigma)))/2; % M-step of EM to optimize L2-parameter given posterior on weights
end