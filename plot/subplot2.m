function [h, m1, m2, x1, x2] = subplot2(varargin)
%[h, n1 n2] = subplot2(n1, n2, x1, x2)
% or [h, n1, n2] = subplot2(n, x)
% creates subplot at position x out of n.
% h: axis handles
%
% See SUBPLOT

if nargin==2
    n2 = varargin{1};
    n1 = 1;
    x2 = varargin{2};
    x1 =1;
%    theAxis = subplot(n1, n2, x);
else
    n1 = varargin{1};
    n2 = varargin{2};
    x1 = varargin{3};
    x2 = varargin{4};
end


if (n1==1 || n2==1) && n1*n2>=4
    n = n1*n2;
    x = x1*x2;
    
    %try to get nice division
    ff = factor(n); 
    cbi = fullfact(2*ones(1,length(ff)))-1; %all combinations of 0s and 1s
    div = exp(cbi*log(ff)'); %all dividers of n
    div = unique(div);
    distsq = abs(div-sqrt(n))/n; %how far from square root
    [mindist, i] = min(distsq);
    
    if mindist < .2
        m1 = div(i);
        m2 = n/m1;
    else
    
    m1 = ceil( sqrt(n));
    m2 = ceil(n / m1);
    end
    
else
    x = x2 + n2*(x1-1);
    m1 = n1;
    m2 =n2;
end

m1 = round(m1);
m2 = round(m2);

h = subplot(m1,m2, x);
