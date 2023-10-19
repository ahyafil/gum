function [B, scale, params, gradB] = basis_poly(X,HP, params)
% compute basis functions as polynomial
% [B, scale, params] = basis_poly(X,HP, params)

order = params.order;
B = zeros(order, length(X));
X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)

for p=1:order+1
    B(p,:) = X.^(p-1);
end
scale = 0:order;

if nargout>3
    % gradient of matrix w.r.t hyperparameters
    gradB = zeros(order, length(X),length(HP));
end

end
