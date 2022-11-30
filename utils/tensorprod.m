function X = tensorprod(X,U)
% P = tensorprod(X,U)
% X: n-dimensional array
% U: cell array of vectors

prod_dims= find(~cellfun(@isempty,U));
S = size(X);

for d= prod_dims
    
    d1 = prod(S(1:d-1)); % number of elements for lower dimensions
    d2 = prod(S(d+1:end)); % number of elements for higher dimensions
    
    u = U{d}; % weight we project on
    nrow = size(u,1);
    
    if d1<d2 %select based on computationally less expensive
        
        new_size = [d1*S(d) d2];
        
        
        if d1>1
            u = kron(u,speye(d1));
        end
        
        if ~isequal(new_size, size(X))
            X  = reshape(X, new_size);
        end
        
        X = u*X;
        
    else % d2>=1
        
        new_size = [d1 S(d)*d2];
        
        u = u';
        if d2>1
            u = kron(speye(d2),u);
        end
        
        if ~isequal(new_size, size(X))
            X  = reshape(X, new_size);
        end
        
        X = X*u;
        
    end
    S(d) = nrow;
    
end

% reshape
X = reshape(X,S);

end