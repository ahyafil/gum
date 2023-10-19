function [B, scale, params, gradB] = basis_exp(X,HP, params)
% compute basis functions as polynomial
% [B, scale, params, gradB] = basis_exp(X,HP, params)
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