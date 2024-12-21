function P = compute_orthonormal_basis_project(V)
% P = compute_orthonormal_basis_project(V), where V is a n-by-m matrix
% (with m<=n), computes an orthonormal basis in n-dimensional space for the subspace orthogonal to
% the m rows of V.
% P is a (m-n)-by-n projection matrix.

assert(ismatrix(V), 'V should be a matrix');
[n,m] = size(V);
assert(m<=n, 'the number of columns in V should not be larger than the rows of columns');

assert(all(any(V~=0,1)), 'at least one of columns is null');
% normalize rows of V to avoid numerical issues (that does not affect the
% subspace its rows form)
for i=1:m
    V(:,i) = V(:,i)/norm(V(:,i));
end

% first check that the columns of V are not colinear
[~,FLAG] = chol(V'*V);
assert(~FLAG, 'the rows in V are colinear');

% deal with special cases
if n==m
    P = zeros(0,n);
    return;
elseif m==0
    P = speye(n);
    return;
elseif all(sum(V~=0,1)==1) % if each column already maps onto a canonical direction
    subs = any(V~=0,2); % covert directions
    rows = 1:n-m;
    cols = find(~subs);
    P = sparse(rows, cols, ones(1,n-m),n-m,n);
    return;
elseif m==1 && all(V==V(1)) % one columns of ones
    P = zeros(n-1,n);
    for i=1:n-1
        P(i,:) = [ones(1,i) -i zeros(1,n-i-1)]/sqrt(i*(i+1))'; % coordinates for i-th basis vector of free space
    end
    return;
end

% now let's complete V to form a basis of R^n
% we'll find the first m-by-m subset of V that forms a basis of R^m
%Q = nchoosek(1:n,m); % all possible subsets of m dimensions
subs = [];
%i=1;
gotit = false;
while ~gotit
    subs = nextcombi(subs,n,m);
    % subs = Q(i,:); % subset of m dimensions
    W = V(subs,:); % square matrix of size m

    [~,flag] = chol(W'*W);
    gotit = ~flag; % if FLAG is 0, then vectors are not colinear

   % i = i+1;
end

% now if we reorder the n dimensions to place subs first and then the other
% n-m dimensions, such that V becomes [W;Z], then the determinant of matrix
% [W =; Z I] is equal to det(W)det(I) = det(W)>0, i.e. the matrix forms a
% basis of R^n. In other words our basis for the complementary subspace is
% formed by the indicator vectors for the complement of subs
comp = setdiff(1:n,subs);
B = zeros(n,n-m);
for j=1:n-m
    B(comp(j),j) = 1;
end
VB = [V B];
% now VB is a basis of R^n, i.e. B forms a basis of the complementary
% subspace.

% finally we want an orthonormal basis, so we'll use Gram-Schmidt method
U = VB; % orthogonal basis
for i=1:n
    u = U(:,i); % i-th column
    for j=i+1:n % for all next columns
        U(:,j) = U(:,j) - dot(u,VB(:,j))/dot(u,u) *u;
    end
end

% nor orthonormal
for i=1:n
    U(:,i) = U(:,i)/ norm(U(:,i));
end

P = U(:,m+1:n)';
end

function subs = nextcombi(subs,n,k)
% find next possible combination of k integers in 1...N
% code is adapted from NCHOOSEK
% We use this instead of NCHOOSEK to avoid issues for large k

if isempty(subs)
    subs = 1:k; % first combination: first k integers
    return;
end

% Find right-most index to increase
% j = find(ind < n-k+1:n, 1, 'last');
for j = k:-1:1
    if subs(j)<n-k+j
        break;
    end
end

% Increase index j, initialize all indices to j's right.
indj = subs(j) - j + 1;
subs(j:k) = indj + (j:k);

end