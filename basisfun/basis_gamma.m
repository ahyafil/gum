function [B, scale, params, gradB] = basis_gamma(X,HP, params)
% compute basis functions as gamma distribution
% [B, scale, params, gradB] = basis_gamma(X,HP, params)

nFun = params.nFunctions;
assert(length(HP)==2*nFun, 'incorrect number of hyperparameters');
theta = HP(1:nFun); % scale parameter
k = HP(nFun+1:2*nFun); % shape parameter

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
            [this_B, scale{i}, ~, this_gradB] = basis_gamma(X(:,this_level),HP, params);
        else
            [this_B, scale{i}] = basis_gamma(X(:,this_level),HP, params);
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


X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)

B = zeros(nFun,length(X));
nz = (X>=0); % time domain with non-null value
for p=1:nFun
    B(p,nz) = X(nz).^(k(p)-1) .* exp(-X(nz)/theta(p)) / gamma(k(p)) / theta(p).^k(p);
    if k(p)==1
        B(p,X==0) = 0;
    end
end
scale = 1:nFun;

if nargout>3
    % compute gradient over each hyperparameter
    gradB = zeros(nFun, length(X),length(HP));
    for p=1:nFun

        % gradient over scale parameter
        gradB(p,nz,p) = B(p,nz).* (X(nz) - k(p)*theta(p)) / theta(p)^2;

        % gradient over shape parameter
        gradB(p,nz,p+nFun) = B(p,nz).* ( log(X(nz)) - psi(k(p)) - log(theta(p))   );

        if k(p)>=1
            gradB(p,X==0,p+[0 nFun])=0;
        end
    end
end
end