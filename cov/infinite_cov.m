function [K, grad] = infinite_cov(scale, HP, B)
% infinite covariance
% [K, grad] = infinite_cov(scale, HP, B)

if nargin<2
    K = 0; % number of hyperparameters
    return;
end

nReg = length(scale);
K = diag(inf(1,nReg));

grad = zeros(nReg,nReg,0);
end