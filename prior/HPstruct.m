function S = HPstruct()
% S = HPstruct();
% void HP structure
S.HP = []; % value of hyperparameter
S.label = {}; % labels
S.fit = []; % which ones are fittable
S.LB = []; % lower bound
S.UB = []; % upper bound
S.index = []; % index (for set of concatenated regressors)
end