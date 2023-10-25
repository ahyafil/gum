function [B, scale, params, gradB] = basis_raisedcos(X,HP, params)
%computes basis functions as raised cosine
%[B, scale, params, gradB] = basis_raisedcos(X,[a,c,Phi_1], params)

if size(X,1)>1
    %% another scale: separate set of basis functions for each level
    levels = X(end,:);
    X(end,:) = [];
    unq = unique(levels);
    nLevel = length(unq);
    B = cell(1,nLevel);
    scale = cell(1,nLevel);
    gradB = cell(1,nLevel);

    for i=1:nLevel
        % compute basis functions separately for each level (recursive
        % call)
        this_level = levels==unq(i);
        if nargout>3
            [this_B, scale{i}, ~, this_gradB] = basis_raisedcos(X(:,this_level),HP, params);
        else
            [this_B, scale{i}] = basis_raisedcos(X(:,this_level),HP, params);
        end

        scale{i}(end+1,:) = unq(i);
        B{i} = zeros(size(this_B,1),length(levels));
        B{i}(:,this_level) = this_B;

        if nargout>3
            gradB{i} = zeros(size(this_B,1),length(levels),length(HP));
            gradB{i}(:,this_level,:) = this_gradB;
        end
    end

    % now concatenate across all levels
    B = cat(1,B{:});
    scale = cat(2,scale{:});
    if nargout>3
        gradB = cat(1,gradB{:});
    end
    return;
end

nCos = params.nFunctions;
a = HP(1);
c = HP(2);
Phi_1 = HP(3);
X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)

B = zeros(nCos,length(X));
Phi = Phi_1 + pi/2*(0:nCos-1); % Pi/2 spacing between each function
for p=1:nCos
    alog = a*log(X+c)-Phi(p);
    nz = (X>-c) & (alog>-pi) & (alog<pi); % time domain with non-null value
    B(p,nz) = cos(alog(nz))/2 + 1/2;
end
scale = 1:nCos;

if nargout>3
    gradB = zeros(nCos, length(X),3);
    for p=1:nCos
        alog = a*log(X+c)-Phi(p);
        nz = (X>-c) & (alog>-pi) & (alog<pi); % time domain with non-null value
        sin_alog = sin(alog(nz)) / 2;
        gradB(p,nz,1) = - sin_alog .* log(X(nz)+c); % gradient w.r.t a
        gradB(p,nz,2) = - sin_alog ./ (X(nz)+c) * a ; % gradient w.r.t c
        gradB(p,nz,3) = sin_alog; % gradient w.r.t Phi_1
    end
end
end