function [B, scale, params, gradB] = basis_gauss(X,HP, params)
% compute basis functions with gaussian functions
% [B, scale, params] = basis_gauss(X,HP, params)

D = params.nDim; % space dimensionality

if size(X,1)>D
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
            [this_B, scale{i}, ~, this_gradB] = basis_gauss(X(:,this_level),HP, params);
        else
            [this_B, scale{i}] = basis_gauss(X(:,this_level),HP, params);
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

nBasisFun = params.nBasisFun;

% first parameters represent mean of gaussian
if params.samemean
    mu = repmat(HP(1:D)',1,nBasisFun); % same mean for all basis fun
    n_mean_HP = D;
else
    mu = reshape(HP(1:D*nBasisFun),D,nBasisFun);
    n_mean_HP = D*nBasisFun;
end
HP(1:n_mean_HP) = [];

v = exp(2*HP); %1/2 log-variance hyperparameters
assert(length(HP)==nBasisFun,'incorrect number of hyperparameters for gaussian basis functions');

B = zeros(nBasisFun, size(X,2));

% compose values of each polynomial for X
for p=1:nBasisFun
    B(p,:) = mvnpdf(X',mu(:,p)',v(p)*eye(D));
end

% scale (id of basis fun)
scale = 1:nBasisFun;

if nargout>3
    % gradient of matrix w.r.t hyperparameters
    gradB = zeros(nBasisFun, size(X,2),length(HP));

    for p=1:nBasisFun

        % gradient w.r.t mean hyperparameters
        mu_idx = 1:D; % index of corresponding mean hyperparameters
        if ~params.samemean
            mu_idx = mu_idx + (p-1)*D;
        end
        norm_dif = X-mu(:,p);
        gradB(p,:,mu_idx) = permute(B(p,:).*norm_dif/v(p),[3 2 1]);

        % gradient w.r.t mean hyperparameters
        gradB(p,:,n_mean_HP+p) = B(p,:).*(sum(norm_dif.^2,1)/v(p)-D);
    end
end

end
