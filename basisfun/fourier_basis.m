function [B,scale,params] = fourier_basis(X, HP, params)
% builds FFT transformation matrix concatenating for multiple processes
%  [B,scale] = basis_fft_cat(xx, len, n, condthresh, minl,addreg)

if nargin<3 
    params.condthresh = 1e12; % threshold on condition number of covariance matrix
    params.padding = true; % whether to add virtual padding
end


minl = exp(HP(1)); % length hyperparameter
assert(minl>=0,'length scale must be positive');

if params.padding
range = max(X) - min(X);  % range of values for given component
Tcirc = range + 3*minl; %   range + 3*minscale (for virtual padding to get periodic covariance)
else
Tcirc = params.Tcirc;
end

% set up Fourier frequencies
if ~isnan(params.condthresh)
     % the number of basis functions is pased on threshold
nFreq = floor((Tcirc/(pi*minl)) * sqrt(.5*log(params.condthresh))); 
else
    % number of basis functions is already provided
nFreq = params.nFreq;
end
nBasisFun = 2*nFreq + 1; % number of basis functions
nBasisFun = min(nBasisFun, length(X)); % no more basis functions than sample points

X = X - min(X); % start at 0

% compute transformation matrix and vector of spectral coordinates
% for just one process
[B,scale] = realnufftbasis(X,Tcirc,nBasisFun); % make Fourier basis

scale = scale'; % row vector
params.Tcirc = Tcirc;
params.nFreq = nFreq;