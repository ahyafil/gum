function S = HPstruct_L2(d, HP, HPfit)
% S = HPstruct_L2(d, HP, HPfit)
% HP structure for L2 regularization

S = HPstruct();

% value of hyperparameter
if nargin>1
    S.HP = HPwithdefault(HP, 0); % default value if not specified
else
    S.HP = 0;
end
S.label = {['log \lambda' num2str(d)]};  % HP labels
if nargin>2
    S.fit = HPfit; % if HP is fittable
else
    S.fit = true;
end
S.LB = -max_log_var; % lower bound: to avoid exp(HP) = 0
S.UB = max_log_var;  % upper bound: to avoid exp(HP) = Inf
end