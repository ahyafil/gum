%% example of regression with polynomial approximation of function
% rho = beta0 + beta1*x + beta2*x^2 ...

clear; close all;
n = 10000;

x = 2*rand(n,1)-1; % sample uniformly between -1 and 1
y = 2*rand(n,1)-1;
rho = exp(x) + 2*y; 
rho = 3*x.^2 - 4*x + 2*y;
p = 1./(1+exp(-rho)); % pass through logistic sigmoid
t = rand(n,1)<p; % draw from Bernoulli distribution

% build table
T = table(x,y,t);

% build model
params = struct('observations','binomial','constant','off');
M = gum(T, 't ~ poly2(x)+y', params);

% infer model
M = M.infer();

% plot model
figure;
h = M.plot_weights;