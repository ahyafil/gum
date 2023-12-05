function       [M,c] = force_definite_positive(M, c)
% forces symmetric matrix to be definite positive (using Huang, Farewell & Pan, 2017)
% M = force_definite_positive(M)
% M = force_definite_positive(M, c) ensures that all eigenvalues are larger
% or equal to c (default: c=1e3*eps)

if any(isnan(M(:)))
    warning('cannot force matrix with nan to be definite positive');
    c = nan;
    return;
end

assert(isreal(M),'matrix should be real');

if nargin<2 || isempty(c)

    % it seems that we need to scale c with eigenvalues to make sure we really end up with definite positive matrix
    mean_eig = trace(M)/size(M,1); % mean eigenvalue (sum of eig is equal to trace)
    factor = max(mean_eig, 1e3); % can't be too small either

    c = factor*eps; % minimum eigenvalue
else
    assert(c>=0, 'c must be non-negative');
end

[~,isNPD] = chol(M);
while isNPD % if some eigenvalue is negative

    n = size(M,1);
    XX = M - c*eye(n);
    try
        [~, HH] = poldec(full(XX)); % Polar Decomposition from Matrix Computation Toolbox (http://www.maths.manchester.ac.uk/%7Ehigham/mctoolbox/)
    catch ME
        if strcmp(ME.identifier,'MATLAB:UndefinedFunction')
            error('The Matrix Computation Toolbox was not found on the path, download from http://www.maths.manchester.ac.uk/%7Ehigham/mctoolbox/ and add to path');
        else
            rethrow(ME)
        end
    end
    M = (XX+HH)/2 + c*eye(n);  % (equation 3 from Huang, Farewell & Pan, 2017, with c=eps)

    if c==0
        isNPD = 0;
    else
        [~,isNPD] = chol(M); % make sure that it did work (it may not, still for numerical reasons... not really sure why)
        if isNPD % otherwise, repeat again with larger c
            c = 10*c;

        end
    end
end
end