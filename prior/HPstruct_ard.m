function H = HPstruct_ard(nReg,d, HP, HPfit)
%S = HPstruct_ard(nReg,d, HP, HPfit)
% HP structure for ARD regularization

H = HPstruct();

% value of hyperparameter
if nargin>1
    H.HP = HPwithdefault(HP, zeros(1,nReg)); % default value if not specified
else
    H.HP = zeros(1,nReg);
end
    H.label = "log \lambda" +d+"_"+ (1:nReg);  % HP labels
if nargin>3 && ~isempty(HPfit)
    H.fit = HPfit; % if HP is fittable
    if isscalar(HPfit)
        H.fit = repmat(HPfit,1,nReg);
    end
else
    H.fit = true(1,nReg);
end
H.LB = -max_log_var*ones(1,nReg); % lower bound: to avoid exp(HP) = 0
H.UB = max_log_var*ones(1,nReg);  % upper bound: to avoid exp(HP) = Inf

H.type = repmat("cov",1,nReg);

end