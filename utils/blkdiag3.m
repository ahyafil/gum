% block diagonal in 3rd dimension
% blkdiag3(M, N, ...) is a 3-dimensional array of size
% size(M,1)+size(N,1)+... by 
% size(M,2)+size(N,2)+...
% size(M,3)+size(N,3)+...
%
% See also BLKDIAG

function MM = blkdiag3(varargin)
C = varargin;
if any(cellfun(@ndims,C)>3) % dimension of the matrix
error('does not support arrays of dimension larger than 3')
end
n = 3;
for i=1:n
    dd(i,:) = cellfun(@(x) size(x,i), C); % size of each matrix in dimension i
end

MM = zeros(sum(dd,2)'); % initialize matrix
cnt = cumsum([zeros(n,1) dd]'); % starting and ending points in each dimension for each matrix
idx = cell(1,n);
for m=1:length(C)
    for i=1:n
        idx{i} = cnt(m,i)+1:cnt(m+1,i);
    end
    MM(idx{:}) = C{m};
end
end