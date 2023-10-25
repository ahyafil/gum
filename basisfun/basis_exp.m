function [B, scale, params, gradB] = basis_exp(X,HP, params)
% compute basis functions as polynomial
% [B, scale, params, gradB] = basis_exp(X,HP, params)

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
            [this_B, scale{i}, ~, this_gradB] = basis_exp(X(:,this_level),HP, params);
        else
            [this_B, scale{i}] = basis_exp(X(:,this_level),HP, params);
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

nExp = length(HP);
B = zeros(nExp,length(X));
X = double(X(1,:)); % x is on first row (extra rows may be used e.g. if splitted)
for p=1:nExp
    tau = exp(HP(p));
    B(p,:) = exp(-X/tau);
end
scale = 1:nExp;

if nargout>3
    % gradient of matrix w.r.t hyperparameters
    gradB = zeros(nExp, length(X),length(HP));
    for p=1:nExp
        tau = exp(HP(p));
        gradB(p,:,p) = B(p,:) .* X/tau;
    end

end
end