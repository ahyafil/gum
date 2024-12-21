function [B,scale,params, gradB] = basis_fourier(X, HP, params)
% Fourier basis functions
%  B = basis_fourier(X, [], params);
%
% [B,scale]= basis_fourier(...)
%
% [B,scale,params]= basis_fourier(...)

if nargin<3 || isnan(params.nFreq) 
    params.condthresh = 1e12; % threshold on condition number of covariance matrix
    params.padding = true; % whether to add virtual padding
end
if size(X,1)>1
    %% another scale: separate set of basis functions for each level
    levels = X(end,:);
    X(end,:) = [];
    unq = unique(levels);
    nLevel = length(unq);
    B = cell(1,nLevel);
    scale = cell(1,nLevel);
    gradB = cell(1,nLevel);

    for i=1:nLevel
        % compute basis functions separately for each level (recursive
        % call)
        this_level = levels==unq(i);
        if nargout>3
            [this_B, scale{i}, params, this_gradB] = basis_fourier(X(:,this_level),HP, params);
        else
            [this_B, scale{i}, params] = basis_fourier(X(:,this_level),HP, params);
        end

        if isstring(unq)
            scale{i} = string(scale{i});
        end
        scale{i}(end+1,:) = unq(i);
        B{i} = zeros(size(this_B,1),length(levels));
        B{i}(:,this_level) = this_B;

        if nargout>3
            gradB{i} = zeros(size(this_B,1),length(levels),length(HP));
            gradB{i}(:,this_level,:) = this_gradB;
        end
    end

    % now concatenate across all levels
    B = cat(1,B{:});
    scale = cat(2,scale{:});
    if nargout>3
        gradB = cat(1,gradB{:});
    end
    return;
end

minl = exp(HP(1)); % length hyperparameter
assert(minl>=0,'length scale must be positive');
X = double(X);

if params.padding
    range = max(X) - min(X);  % range of values for given component
    Tcirc = range + 3*minl; %   range + 3*minscale (for virtual padding to get periodic covariance)
else
    Tcirc = params.Tcirc;
end

% set up Fourier frequencies
if ~isempty(params.condthresh) && ~isnan(params.condthresh)
    % the number of basis functions is pased on threshold
    nFreq = floor((Tcirc/(pi*minl)) * sqrt(.5*log(params.condthresh)));
else
    % number of basis functions is already provided
    nFreq = params.nFreq;
end
nBasisFun = 2*nFreq + 1; % number of basis functions

% whether we actually use transform
params.noTransform = nBasisFun>=length(X);

if params.noTransform
    % in case that there are more basis functions than datapoints, then
    % forget about transform, we're better with the original space
    % (where the data can stay in sparse format)
    % note:perhaps should lower to a fraction of length(X)
    B = speye(length(X)); % basis functions: identity matrix
    scale = X;
else
    % use basis functions

    %nBasisFun = min(nBasisFun, length(X)); % no more basis functions than sample points

    X = X - min(X); % start at 0

    % compute transformation matrix and vector of spectral coordinates
    % for just one process
    [B,scale] = fftbasis(X,Tcirc,nBasisFun); % make Fourier basis

    scale = scale'; % row vector
    params.Tcirc = Tcirc;
    params.nFreq = nFreq;
end

% gradient is null as HP only used to define frequency cutoff
if nargout>3
    gradB = zeros([size(B) length(HP)]);
end
end

%% 
function [B,wvec] = fftbasis(tvec,T,N)
% Real basis for non-uniform discrete fourier transform
% (adapted from Pillowlab code)
%
% [B,wvec] = fftbasis(tvec,T,N)
%
% Makes basis of sines and cosines B for a function sampled at nonuniform
% time points in tvec, with a circular boundary on [0, T], and spacing of
% frequencies given by integers m*(2pi/T) for m \in [-T/2:T/2].
%
% If maxw input given, only returns frequencies with abs value < maxw
%
% INPUTS:
%  tvec [nt x 1] - column vector of non-uniform time points for signal
%     T  [1 x 1] - circular boundary for function in time
%     N  [1 x 1] - number of Fourier frequencies to use
%
% OUTPUTS:
%      B [ni x N] - basis matrix for mapping Fourier representation to real
%                   points on lattice of points in tvec.
%    wvec [N x 1] - DFT frequencies associated with columns of B

% make column vec if necessary
if size(tvec,1) == 1
    tvec = tvec';
end

if max(tvec+1e-6)>T
    error('max(tvec) greater than circular boundary T!');
end

% Make frequency vector
ncos =  ceil((N+1)/2); % number of cosine terms (positive freqs)
nsin = floor((N-1)/2); % number of sine terms (negative freqs)
wcos = (0:ncos-1)'; % cosine freqs
wsin = (-nsin:-1)'; % sine freqs
wvec = [wcos;wsin]; % complete frequency vector for realfft representation

if nsin > 0
    B = [cos((2*pi/T)*wcos*tvec'); sin((2*pi/T)*wsin*tvec')]/sqrt(T/2);
else
    B = cos((2*pi/T)*wcos*tvec')/sqrt(T/2);
end

% make DC term into a unit vector
B(1,:) = B(1,:)/sqrt(2);

% if N is even, make Nyquist (highest-freq cosine) term into unit vector
if (mod(N,2)==0) && (N==T)
    B(ncos,:) = B(ncos,:)/sqrt(2);
end
end
