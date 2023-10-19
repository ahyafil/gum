function [B, scale, params, gradB] = basis_raisedcos(X,HP, params)
%computes basis functions as raised cosine
%[B, scale, params, gradB] = basis_raisedcos(X,[a,c,Phi_1], params)

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