function [I, x,w,grad] = GaussHermiteInt(fun, n, mu, sigma,varargin)
% GaussHermiteInt(FUN,n) computes the Gauss-Hermite approximation
% for the integral over reals of the product of function FUN by the standard
% normal distribution, using n sample points.
% GaussHermiteInt(FUN) uses 100 samples points.
%
% GaussHermiteInt(FUN,n,mu,sigma) computes the approximation for the
% integral over reals of the product of FUN by the normal distribution of
% mean mu and standard deviation sigma. mu and sigma can be scalar, or
% column vectors of the same length.
%
% GaussHermiteInt(FUN,n,mu,sigma,y,z,..) computes the integral of
% FUN(x,y,z,...) over x
%
% [I,x,w] = GaussHermiteInt(..) returns the sampling points (zeros of
% Hermitian polynomials) in x and corresponding weights in w.
%
%[I,x,w,grad] = GaussHermiteInt(FUN,n,mu,sigma,...,grad_mu,grad_sigma) computes the gradient of
% the integral, where grad_mu and grad_sigma are the gradient of mu and
% sigma w.r.t variables. Columns of grad_mu and grad_sigma represent the
% derivatives over the different variables. grad has the same number of
% rows as I, where each column stands for the derivative w.r.t one variable

%default number of samples
if nargin<2
    n=100;
end

% by default normal distribution is standard
if nargin<3
    mu=0;
    sigma =1;
else
    assert(iscolumn(mu)&& iscolumn(sigma),'mu and sigma should be scalars or column vectors');
    assert(isscalar(mu) || isscalar(sigma) || length(mu)==length(sigma),...
        'mu and sigma should have the same length, or at least one of them should be scalar');
end

with_grad = nargout>3;
if with_grad
    assert(length(varargin)>1,'missing arguments to compute gradient');
    grad_mu = varargin{end-1};
    grad_sigma = varargin{end};
    varargin(end-1:end)=[];
    assert(ismatrix(grad_mu)&&ismatrix(grad_sigma),'grad_mu and grad_sigma should be matrices');
    assert(size(grad_mu,2)==size(grad_sigma,2),'number of columns in grad_mu and grad_sigma should match');
end

%compute sample points (roots of Hermite polynomial)and corresponding
%weights
[x, w] = GaussHermiteRoots(n);

% rescale x (change of variable to get standard normal distribution)
xx = sqrt(2)*sigma.*x + mu;

% compute integral approximation
w_feval = w.*fun(xx,varargin{:});
I = sum( w_feval,2) / sqrt(pi);

if with_grad
    % derivative w.r.t mu = < f(x)(x-mu)>/sigma^2
    dI_dmu = sum( w_feval .* (xx-mu),2) / sqrt(pi);
    dI_dmu = dI_dmu./sigma.^2;

    % derivative w.r.t sigma = -< f(x)>/sigma +<f(x)*(x-mu)^2>/sigma^3
    dI_dsigma = -I./sigma+ sum(w_feval.*(xx-mu).^2,2) / sqrt(pi)./sigma.^3;

    % overall gradient
    grad = dI_dmu.*grad_mu + dI_dsigma.*grad_sigma;
end

end

function [x, w] = GaussHermiteRoots(n)
% computing roots of Hermite polynomials and corresponding weights, using
% Golub-Welsh formula

% 1. compute Jacobi matrix
a   = sqrt((1:n-1)/2);
Jacobi  = diag(a,1) + diag(a,-1);

% 2. compute eigenvalues and first components of Jacobi matrix
[V, D]   = eig(Jacobi);
[x, idx] = sort(diag(D)); % re-order
x = x';
V = V(:,idx);

% 3. compute corresponding weights
w  = sqrt(pi) * V(1,:).^2;
end

