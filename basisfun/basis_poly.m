function [B, powers, params, gradB] = basis_poly(X,HP, params)
% compute basis functions as polynomial
% [B, scale, params] = basis_poly(X,HP, params)

D = params.nDim; % space dimensionality

if size(X,1)>D
    %% another scale: separate set of basis functions for each level
    levels = X(end,:);
    X(end,:) = [];
    unq = unique(levels);
    nLevel = length(unq);
    B = cell(1,nLevel);
    powers = cell(1,nLevel);
    gradB = cell(1,nLevel);

    for i=1:nLevel
        % compute basis functions separately for each level (recursive
        % call)
        this_level = levels==unq(i);
        if nargout>3
            [this_B, powers{i}, ~, this_gradB] = basis_poly(X(:,this_level),HP, params);
        else
            [this_B, powers{i}] = basis_poly(X(:,this_level),HP, params);
        end

        powers{i}(end+1,:) = unq(i);
        B{i} = zeros(size(this_B,1),length(levels));
        B{i}(:,this_level) = this_B;

        if nargout>3
            gradB{i} = zeros(size(this_B,1),length(levels),length(HP));
            gradB{i}(:,this_level,:) = this_gradB;
        end
    end

    % now concatenate across all levels
    B = cat(1,B{:});
    powers = cat(2,powers{:});
    if nargout>3
        gradB = cat(1,gradB{:});
    end
    return;
end

order = params.order;

% get all possible powers for x1, x2... such that sum of powers is not
% larger than order
powers = nchoosek(0:order+D-1, D); % all combinations of D values (all integers in a simplex)
for d=2:D
    powers(:,d) = powers(:,d) - sum(powers(:,1:d-1),2) - d+1;
end
powers = powers';

if ~params.intercept %remove intercept
    powers(:,1) = [];
end

nBasisFun = size(powers,2);
B = zeros(nBasisFun, length(X));

% compose values of each polynomial for X
for p=1:nBasisFun
    B(p,:) = X(1,:).^powers(1,p);
    for d=2:D
        B(p,:) = B(p,:) .* X(d,:).^powers(d,p);
    end
end

if nargout>3
    % gradient of matrix w.r.t hyperparameters
    gradB = zeros(nBasisFun, length(X),length(HP));
end

end
