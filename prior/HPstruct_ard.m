function H = HPstruct_ard(nReg, HP, HPfit)
%H = HPstruct_ard(nReg,HP, HPfit)
% HP structure for ARD regularization

H = HPstruct();

% set value of hyperparameter
if nargin>1 && isfield(HP, 'value')
    H.HP = HP.value; %if value is specified
    if isscalar(H.HP)
        H.HP = repmat(H.HP,1,nReg);
    else
        assert(length(H.HP)==nReg,'incorrect number of hyperparameters for ARD prior');
    end
elseif nargin>1 && isfield(HP, 'variance')
    H.HP = log(HP.variance)/2*ones(1,nReg); % if variance is specified
else
    H.HP = zeros(1,nReg); % default: variance 1
end

% labels
H.label = "\log \lambda_"+ (1:nReg);  % HP labels
if nargin>3 && ~isempty(HPfit)
    HPfit = logical(HPfit);
    H.fit = HPfit; % if HP is fittable
    if isscalar(HPfit)
        H.fit = repmat(HPfit,1,nReg);
    end
else
    H.fit = true(1,nReg);
end

% upper and lower bounds
H.LB = -max_log_var*ones(1,nReg); % lower bound: to avoid exp(HP) = 0
H.UB = max_log_var*ones(1,nReg);  % upper bound: to avoid exp(HP) = Inf

H.type = repmat("cov",1,nReg);

end