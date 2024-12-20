function [B,new_scale, gradB] = compute_basis_functions(B, scale, HP)
% compute basis functions matrix and scale

isConcatenated = ~isscalar(B); % concatenated regressors

isBasisHP = contains(HP.type, "basis");

% compute projection matrix and levels in projected space
if ~isConcatenated % if no splitting or concatenating

    %if isrow(scale) && ~isConcatenated % if no splitting or concatenating
    [B.B,new_scale, B.params, gradB] = B.fun(scale, HP.HP(isBasisHP), B.params); % apply function (params is hyperparameter)
    B.nWeight = size(B.B,1); % not specified a priori for some cases (e.g. SE kernel w spectral trick)
    if nargout>2
    assert(size(gradB,3)==sum(isBasisHP), 'dimension 3 of gradient should match the number of basis functions hyperparameters');

    gradB(:,:,~HP.fit(isBasisHP)) = []; % remove fixed HPs from gradient
    end
else
    % more rows in scale means we fit different
    % functions for each level of splitting
    % variable
    assert(isempty(HP.index) || length(HP.index)==length(HP.type),'HP.index should be a vector of length equal to the number of hyperparameters');

    new_scale = zeros(size(scale,1),0);
    [id_list,~,split_id] = unique(scale(2:end,:)','rows'); % get id for each level
    scale(2:end,:) = [];

    B(1).B = zeros(0,size(scale,2)); % the matrix will be block-diagonal
    gradB = zeros(0,size(scale,2),sum(isBasisHP & HP.fit));
    nWeight = zeros(1,length(id_list));

    for g=1:length(id_list) % for each group of regressors
        subset = split_id==g; % subset of weights for this level of splitting variable
        if ~isempty(HP.index)
            iHP = HP.index==g & isBasisHP;
        else
            iHP = isBasisHP;
        end
        this_HP = HP.HP(iHP);

        gg = min(g,length(B)); % depending if B array with different structure or same B struct for all
        Bs = B(gg);

        if ~isequal(Bs.fun,'none')
            % evaluate function to compute basis function
            [this_B,this_new_scale,B(gg).params, this_gradB] = Bs.fun(scale(:,subset), this_HP, Bs.params);
            nWeight(g) = size(this_B.B,1); 
        else
            % when concatening regressor with and without basis function,
            nWeight(g) = sum(subset);
            this_B = eye(nWeight(g)); % no transform
            this_new_scale = scale(:,subset);
            
            this_gradB = zeros(nWeight(g),nWeight(g),sum(HP.index==g & HP.fit));
        end
        n_new = size(this_B,1);
        B(1).B(end+1 : end+n_new, subset) = this_B; %
        new_scale(1,end+1:end+n_new) = this_new_scale;
        new_scale(2:end,end-n_new+1:end) = repmat(id_list(g,:)',1,n_new);
        iBHP = HP.index(isBasisHP)==g; % corresponding indices for hyperparameters
        iBHP = iBHP(HP.fit); 
        gradB(end+1 : end+n_new, subset,iBHP) = this_gradB;
    end
    B(1).nWeight = sum(nWeight);

end

% if nargout>2
% % add gradient w.r.t no-basis function HPs
% tmp = gradB;
% gradB = zeros(size(gradB,1),size(gradB,2),length(HP.HP));
% gradB(:,:,isBasisHP) = tmp;
% end

end