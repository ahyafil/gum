function H = HPstruct()
% S = HPstruct();
% void HP structure
H.HP = []; % value of hyperparameter
H.label = {}; % labels
H.fit = []; % which ones are fittable
H.LB = []; % lower bound
H.UB = []; % upper bound
H.index = []; % index (for set of concatenated regressors)
H.type = string([]); % type "cov" or "basis", for covariance or basis function HP
end