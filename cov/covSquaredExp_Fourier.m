function [K, grad] = covSquaredExp_Fourier(scale, HP, B)
% [K, grad] = covSquaredExp_Fourier(HP, scale, params)
%
% Squared Exponential RBF covariance matrix, in the Fourier domain:
%  K = RBF(X, tau, rho, sigma)
% K(i,j) = rho*exp(- [(x(i,1)-x(j,1))^2 + ...
% (x(i,d)-x(j,d))^2]/(2*tau^2)) + sigma*delta(i,j)
%
% tau can be scalar (same scale for each dimension) or a fecor or length D
%omit sigma if there is no innovation noise
%
% [K, grad] = RBF_Fourier(X, tau, rho, sigma, incl) get gradient over
% hyperparameters, incl is a vector of three booleans indicating which
% hyperparameter to optimize


tau = exp(HP(1:end-1));
rho = exp(HP(end));

Tcirc = B.params.Tcirc;
nDim = B.params.nDim;

% covariance matrix is diagonal in fourier domain
% !! check the formula !!
eexp = exp(-2*pi^2/Tcirc^2*tau^2*scale(1:nDim,:).^2);
kfdiag = sqrt(2*pi)*rho*tau*eexp;
K = diag(kfdiag);

% n_tau = length(tau);
% if n_tau ~=1 && n_tau ~= size(X,2)
%     error('tau should be scalar or a vector whose length matches the number of column in X');
% end
% tau = tau(:);
n = size(scale,2); % number of data points

% compute gradient
if nargout>1
    grad = zeros(n,n,2); % pre-allocate
    % if n_tau == 1
    grad_scale = sqrt(2*pi)*rho*  (1 - 4*pi^2/Tcirc^2*tau^2*scale(nDim,:).^2) .* eexp; % derivative w.r.t GP scale (tau)
    grad(:,:,1) = diag(grad_scale);
    % else
    %     for t=1:n_tau % gradient w.r.t scale for each dimension
    %             grad(:,:,t) = rho * dist(X(:,t)').^2 .* exp(-D.^2/2)/tau(t)^3; % derivative w.r.t GP scale
    %     end
    % end
    grad(:,:,2) = K/rho; % derivative w.r.t GP weight (rho)
end

end