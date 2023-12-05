function [B,scale,params, gradB] = basis_fourier(X, HP, params)
% Fourier basis functions
%  B = basis_fourier(X, [], params);
%
% [B,scale]= basis_fourier(...)
%
% [B,scale,params]= basis_fourier(...)

if nargin<3
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

if nargout>3
    gradB = zeros([size(B) length(HP)]); % gradient is null as HP only used to define frequency cutoff
end