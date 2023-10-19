function varargout = L2basis_covfun(scale, HP, B)
% L2 covariance function when using basis (L2 on last hyperparameter)
% L2basis_covfun(scale, HP, B)
if nargin<2
    varargout = {1}; % number of hyperparameters
else
    varargout = cell(1,nargout);
    loglambda = HP(end);
    [varargout{:}] = L2_covfun(scale,loglambda);
    if nargout>1
        % place gradient over variance HP as last matrix in 3-D array
        nR = size(varargout{1},1);
        grad = varargout{2}.grad;
        varargout{2} = cat(3,zeros(nR,nR,length(HP)-1),grad);
    end
end
end