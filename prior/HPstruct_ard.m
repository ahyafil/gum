function S = HPstruct_ard(nReg,d, HP, HPfit)
%S = HPstruct_ard(nReg,d, HP, HPfit)
% HP structure for ARD regularization

S = HPstruct();

% value of hyperparameter
if nargin>1
    S.HP = HPwithdefault(HP, 0); % default value if not specified
else
    S.HP = zeros(1,nReg);
end
for i=1:nReg
    S.label = {['log \lambda' num2str(d) '_' num2str(i)]};  % HP labels
end
if nargin>2
    S.fit = HPfit; % if HP is fittable
else
    S.fit = true(1,nReg);
end
S.LB = -max_log_var*ones(1,nReg); % lower bound: to avoid exp(HP) = 0
S.UB = max_log_var*ones(1,nReg);  % upper bound: to avoid exp(HP) = Inf
end