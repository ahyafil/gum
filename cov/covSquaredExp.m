function [K, grad] = covSquaredExp(X, HP, B)
% radial basis function covariance matrix:
%  K = covSquaredExp(X, [log(tau), log(rho)])
% K(i,j) = rho^2*exp(- [(x(i,1)-x(j,1))^2 + ...
% (x(i,d)-x(j,d))^2]/(2*tau^2))
%
% tau can be scalar (same scale for each dimension) or a vector of length D
%
% [K, grad] = covSquaredExp(...) get gradient over
% hyperparameters, incl is a vector of two booleans indicating which
% hyperparameter to optimize

if nargin<2
    K = 2; % number of hyperparameters
    return;
end

within = ~iscell(X); % distance within set of points
if  within
    X = {X,X};
end
X{1} = X{1}';
X{2} = X{2}';

tau = exp(HP(1:end-1));
rho2 = exp(2*HP(end));

n_tau = length(tau);
if n_tau ~=1 && n_tau ~= size(X{1},2)
    error('tau should be scalar or a vector whose length matches the number of column in X');
end
tau = tau(:)';
if length(tau)==1
    tau = repmat(tau,1,size(X{1},2));
end

m = size(X{1},1); % number of data points
n = size(X{2},1); % number of data points

Xnorm = X{1}./tau; % normalize by scale
Ynorm = X{2}./tau;

% exclude scale-0 dimensions
nulltau = tau==0;
xx = Xnorm(:,~nulltau);
yy = Ynorm(:,~nulltau);

D = zeros(size(xx,1), size(yy,1));  % cartesian distance matrix between each column vectors of X
for i=1:size(xx,1)
    for j=1:size(yy,1)
        D(i,j) = sqrt(sum((xx(i,:)-yy(j,:)).^2));
        %  D(j,i) = D(i,j);
    end
end
if all(nulltau)
    D = zeros(m,n);
end

% treat separately for null scale
for tt=1:find(nulltau)
    Dinf = inf(m,n); % distance is infinite for all pairs
    SameValue = X(:,tt)==X(:,tt)';
    Dinf( SameValue ) = 0; % unless values coincide
    D = D + Dinf;
end

K = rho2* exp(-D.^2/2);
if within
    K = force_definite_positive(K);
end

% compute gradient
if nargout>1
    grad = zeros(m,n,n_tau+1); % pre-allocate
    if n_tau == 1
        grad(:,:,1) = rho2* D.^2 .* exp(-D.^2/2); % derivative w.r.t GP scale
    else
        for t=1:n_tau % gradient w.r.t scale for each dimension
            dd = bsxfun(@minus, X{1}(:,t),X{2}(:,t)').^2; % distance along dimension
            grad(:,:,t) = log(rho2) * dd .* exp(-D.^2/2)/tau(t)^2; % derivative w.r.t GP scale
        end
    end
    grad(:,:,n_tau+1) = 2*K; % derivative w.r.t GP log-variance (log-rho)
    % grad(:,:,n_tau+1) = exp(-D.^2/2); % derivative w.r.t GP variance (rho)
    % grad(:,:,n_tau+2) = eye(n); % derivative w.r.t. innovation noise
    % grad = grad(:,:,which_par);
end
end