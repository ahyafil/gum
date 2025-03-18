function [B, scale, params, gradB] = basis_powerlaw(X,HP, params)
% compute basis functions as power law functions
% [B, scale, params, gradB] = basis_powerlaw(X,HP, params)

nFun = params.nFunctions;
assert(length(HP)==nFun, 'incorrect number of hyperparameters');
alpha = HP; % power parameter

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
            [this_B, scale{i}, ~, this_gradB] = basis_powerlaw(X(:,this_level),HP, params);
        else
            [this_B, scale{i}] = basis_powerlaw(X(:,this_level),HP, params);
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
for p=1:nFun
    B(p,:) = X.^alpha(p);
end
scale = 1:nFun;

if nargout>3
    % compute gradient over each hyperparameter
    gradB = zeros(nFun, length(X),length(HP));
    nz = ~(X==0);
    for p=1:nFun
        gradB(p,nz,p) = log(X(nz)).*B(p,nz);
    end
end
end