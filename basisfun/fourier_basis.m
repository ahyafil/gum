function [B,scale,params] = fourier_basis(X, HP, params)
% builds FFT transformation matrix concatenating for multiple processes
%  [B,scale] = basis_fft_cat(xx, len, n, condthresh, minl,addreg)

if nargin<4 % threshold on condition number of covariance matrix
    params.condthresh = 1e12;
end


minl = exp(HP(1)); % length hyperparameter
assert(minl>=0,'length scale must be positive');

range = max(X) - min(X);  % range of values for given component
Tcirc = range + 3*minl; %   range + 3*minscale (for virtual padding to get periodic covariance)

% set up Fourier frequencies
maxFreq = floor((Tcirc/(pi*minl)) * sqrt(.5*log(params.condthresh)));  % max frequency to use
nFreq = 2*maxFreq + 1; % number of fourier frequencies
nFreq = min(nFreq, length(X)); % no more fourier frequencies than sample points

X = X - min(X); % start at 0

% compute transformation matrix and vector of spectral coordinates
% for just one process
[B,scale] = realnufftbasis(X,Tcirc,nFreq); % make Fourier basis

scale = scale'; % row vector
params.Tcirc = Tcirc;