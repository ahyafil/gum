function [B, scale, params, gradB] = basis_gamma(X,HP, params)
% compute basis functions as gamma distribution
% [B, scale, params, gradB] = basis_gamma(X,HP, params)

nFun = params.nFunctions;
theta = HP(1:nFun); % scale parameter
k = HP(nFun+1:2*nFun); % shape parameter
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