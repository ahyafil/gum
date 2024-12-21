function S = HPstruct_L2(HP, HPfit)
% S = HPstruct_L2(HP, HPfit)
% HP structure for L2 regularization

S = HPstruct();

% value of hyperparameter
S.HP = 0;
if nargin>1 && isfield(HP, 'value') && ~isempty(HP.value)
    S.HP = HP.value(1); %if value is specified
elseif nargin>1 && isfield(HP, 'variance')
    S.HP = log(HP.variance)/2; % if variance is specified
end

S.label = "log \lambda";  % HP labels
if nargin>2 && ~isempty(HPfit)
    S.fit = logical(HPfit); % if HP is fittable
else
    S.fit = true;
end
S.LB = -max_log_var; % lower bound: to avoid exp(HP) = 0
S.UB = max_log_var;  % upper bound: to avoid exp(HP) = Inf

S.type = "cov";
end