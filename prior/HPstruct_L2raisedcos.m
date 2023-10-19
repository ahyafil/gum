function HH = HPstruct_L2raisedcos(HH, scale, nCos)
% Hyperparameter structure for L2 prior with raised cosine basis functions
dt = min(diff(scale)); % time step
if isempty(dt)
dt = 1;
end
Ttot = scale(end)-scale(1); % total span
c = dt-scale(1); % time shift
if nCos<=3
    k_ini = -1; % first basis function starts on the rise
elseif nCos<=5
    k_ini = 0; % first basis function starts at peak
else
    k_ini = 1; % first basis function starts on decay
end
if nCos<=6
    k_end = 1; % last basis function end on decay
elseif nCos<=9
    k_end = 0; % last basis function end on peak
else
    k_end = -1; % last basis function end on rise
end
a = (nCos-1+k_end-k_ini)*pi/2/log(1+Ttot/dt);  % time power, this formula is to tile all time steps with

Phi_1 = a*log(dt) - k_ini*pi/2; % angle for first basis function
HH.HP = [a c Phi_1 0];
HH.LB = [a-2 -max(scale)  Phi_1-pi   -max_log_var];
HH.UB = [a+2 max(scale)-2*min(scale)  Phi_1+pi    max_log_var];
HH.fit = true(1,4);
HH.label = {'power','timeshift', '\Phi_1','\log \alpha'};
HH.type = ["basis","basis","basis","cov"];

end