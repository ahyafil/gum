function [Kfree, gradKfree] = covariance_free_basis(K, ct,gradK)
% Kfree = covariance_free_basis(K,ct) transforms the covariance in full space K
% into the covariance in free space given constraint structure K.
% If K is size n-by-n and ct containes c constraint, then
% Kfree is of size (n-c)-by-(n-c).
%
% [K, gradKfree] = covariance_free_basis(K, ct,gradK) to also project
% gradient of covariance w.r.t hyperparameter

assert(size(K,1)==size(K,2),'K should be a square matrix');

if isequal(ct,"free") || ct.nConstraint==0
    Kfree=K;
    if nargout>1
        gradKfree = gradK;
    end
    return;
elseif ct.type =="fixed"
    Kfree = [];
    if nargout>1
    gradKfree = zeros(0,0,size(gradK,3));
    end
    return;
elseif all(isinf(diag(K)))
    nFree = size(ct.V,1) - ct.nConstraint; % size of Kfree
    Kfree = diag(inf(1,nFree));
    if nargout>1
    gradKfree = nan(nFree,nFree,size(gradK,3));
    end
    return;
end
n = size(K,1);
V = ct.V;
P = ct.P;
VKV = V'*K*V;
J = eye(n)- V/VKV*V'*K;

Kfree = P*K*J*P';
Kfree = (Kfree+Kfree')/2; % make symmetric to correct numerical errors

if nargout>1

    if isstruct(gradK)
        gradK = gradK.grad;
    end
    nHP = size(gradK,3);
    gradKfree = zeros(size(Kfree,1),size(Kfree,2),nHP);
    for p=1:nHP
        gradJ = V/VKV*V'*gradK(:,:,p)*(V/VKV*V'*K - eye(n));
        gradKfree(:,:,p) = P*(gradK(:,:,p)*J + K*gradJ)*P';
    end
end
end