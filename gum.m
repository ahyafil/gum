classdef gum
    
    % CONSTRUCTOR
    % function [M,S,grad] = gum(X, T, param)
    %
    % Generalized Unrestricted Model.  Design matrix X is (D+1)-dimensional, targets T, optional instance
    % weights W, optional sigma term sigma, optional parameters object PARAM.
    %
    %   p(T_i |X,U) = 1 ./ (1+exp(-sum_{k,l,...}
    %   U{1}(k)*U{2}(l}*...*X(i,k,l,..))
    %
    % Outputs is the cell array of weights U
    %
    % Possible fields in params are:
    % -'observations': 'binomial' (default) or 'poisson'
    %
    %-'w': is a weighting vector with length equal to the number of training examples
    %
    % -'constraint': how to constraint set of weights to avoid rescaling
    % indeterminacy. Value must be a character array of length D (or rank x D matrix) where each
    % character sets the type of constraint:
    %     * 'f': free, unconstrained (one and only one dimension)
    %     * 'b': sum of weights is set to 0
    %     * 'm': mean over weights in dimension d is set to 1
    %     * 's': sum over weights in dimension d is set to 1
    %     * '1': first element in weight vector is set to 1
    %     * 'n': fixed weights
    % Default value is 'f' (unconstrained) for first dimension, 'm' for other
    % dimensions.
    %
    % -'sigma': a cell array describing the covariance of the prior over
    % weights for each dimension. sigma{d} can be:
    %     * a scalar: covariance matrix is proportional to identity matrix
    %     sigma{d}*eye(m(d)), where m is the number of weights along dimension
    %     d (equivalent to ridge regression)
    %     * a vector of length m(d): covariance matrix is diagonal.
    %     * a square matrix of size m(d) x m(d): to describe the full
    %     covariance matrix
    % Covariance matrix for additional regressors (see below) can be set in
    % sigma{D+1}.
    % By default, covariance matrix is set to identity matrix in each
    % dimension.
    %
    % -'mu': a cell array describing the mean of the prior over weights for
    % each dimension. mu{d} must be a vector of length m(d).
    % By default mu{d} is set to 0 for free regressors ('f' or all weights in
    % '1' except first), to 1 for 'm' set of regressors, and /m(d) for 's' set
    % of regressors.
    %
    %
    % -'U': initial value for the weights
    %- 'constant': add constant bias (as extra component) (default: 'on')
    %- 'rank': rank of estimation matrix R (default 1). If larger than one, the
    %generative model changes to:
    %   p(T_i |X,U) = 1 ./ (1+exp(-sum_{k,l,...,c}
    %   U{c,1}(k)*U{c,2}(l}*...*X(i,k,l,..)) with c from 1 to R.
    %
    %
    %
    % ESTIMATE AND FIT
    % M = M.estimate(param) and M = M.fit(param);
    %- 'maxiter': maximum number of iterations of IRLS estimation algorithm
    %(default 50)
    %- 'miniter': minimum number of iterations of IRLS estimation algorithm
    %(default 4)
    %- 'TolFun': Tolerance value for LLH function convergence (default:1e-12)
    %
    
    %- 'ordercomponent': whether to order components by descending order of
    %average variance (default: true if all components have same constraints)
    %
    % - 'initialpoints': number of initial points for inference (default:10)
    %
    % - 'HPfit': which method is used to fit hyperparameters. Possible values
    % are:
    % * 'none': hyperparameters are not fitted
    % * 'basic': basic grid search for parameters that maximize marginalized
    % likelihood
    % * 'EM': Expectation-Maximization algorithm to find hyperparameters that
    % maximize marginalized likelihood
    % * 'cv': Cross-Validation. Cross-Validated Likelihood is maximized. For
    % this option, additional fields must be provided:
    % - 'crossvalidation': to use crossvalidation. Value of the field can be:
    %     * a 'cvpartition' object created using function cvpartition( stats
    %     toolbox)
    %     * a scalar determining the number of observations in the training set. Use
    %     either the raw number of observations; a number between  and 1 to indicate
    %     the proportion of observations; or -1 for leave-one-out. Use field 'nvalidation'
    %     to indicate number of observations in validation set (default: all observations not included
    %     in training set), and 'nperm' to indicate number of permutations
    %     (default:100).
    %     * a cell array of size nperm x 2, where the elements in the first column indicate
    % observations in the training sets, second column indicate observations in the test
    % sets, and rows indicate permutation
    % - 'gradient': whether to use gradients of CVLL to speed up search.
    % Gradients computation may suffer numerical estimation problems if
    % covariance are singular or close to singular. Possible values are: 'off'
    % [default], 'on' and 'check' (uses gradient but checks first that
    % numerical estimate is correct)
    %
    % -'gradient_hyperparameters': provide gradient of prior covariance matrix w.r.t hyperparameters
    % to compute gradient of MAP LLH over hyperparameters.
    % - 'maxiter_HP': integer, defining the maximum number of iterations for
    % hyperparameter optimization (default: 200)
    % - 'no_fitting': if true, does not fit parameters, simply provide as output variable a structure with LLH and
    % accuracy for given set of parameters (values must be provided in field
    % 'U')
    %
    %
    %
    %
    %
    % OUTPUT:
    % M is the Maximum A Posteriori (MAP) parameter set.
    %
    % - 'se': a cell array composed of the vector of standard errors of the mean (s.e.m.)
    % for each set of weights.
    % - 'T': Wald T-test of significance for each weight
    % - 'p': associated probability
    %
    % [M,S] = gum(...)
    % S is a structure with following fields:
    % - 'LLH': MAP log-likelihood
    % - 'rho': vector of predicted predictor at MAP parameters
    % - 'Y': vector of predicted probability of target at MAP parameters
    % - 'covb': covariance matrix of parameters weights (expressed in free
    % basis of the parameter, one parameter is removed per non-free dimension).
    % - 'testscore': LLH over test set of observations normalized by number of
    % observations, averaged across permutations (for cross-validation)
    % - 'testscore_all': a vector of LLH over test set of observations normalized by number of
    % observations, for each permutation
    % - 'LogEvidence': estimated log-evidence for the model (to improve...)
    % - 'BIC': Bayesian Information Criterion
    % - 'AIC': Akaike Information Criterion
    % - 'AICc': corrected Akaike Information Criterion
    %
    % [M S grad] = gum(...)
    % grad is the gradient of the LLH at MAP over hyperparameters
    % (field 'gradient_hyperparameters' must be provided in param)
    %
    % Methods:
    % - 'compute_rho_variance' provide the variance of predictor in output structure
    %
    % version 0.0. Bug/comments: send to alexandre.hyafil@gmail.com
    %
    
    properties
        regressor
        T
        obs
        nObs = 0
        nMod = 0
        ObservationWeight
        param = struct()
        grad
        score = struct()
        Predictions = struct('rho',[])
    end
    
    
    
    
    methods
        
        %% %%%%%% CONSTRUCTOR %%%%%
        function obj = gum(M, T, param)
            
            if nargin==0
                %% empty class
                return;
            end
            
            % optional parameters
            if (nargin < 3)
                param = struct;
            end
            
            if istable(M)
                [M, T, param] = parse_formula(M,T, param);
            end
            
            obj.regressor = M;
            
            % convert M to structure if needed
            if isnumeric(M)
                M = regressor(M,'linear');
            end
            nMod = length(M); % number of modules
            
            %  n = prod(n); % number of observations
            
            %% check dependent variable
            if isvector(T)
                n = length(T);
                T = T(:);
                BinaryCountCode = 0;
            elseif ~ismatrix(T) || size(T,2)~=2
                error('T should be a column vector or a matrix of two columns');
            else
                n = size(T,1);
                BinaryCountCode = 1;
            end
            obj.nObs = n;
            
            
            %% check regressors
            for m=1:nMod
                % M(m).val = tocell(M(m).val);
                M(m) = checkregressorsize(M(m),n);
                %                M(m).sparse = logical(M(m).sparse);
            end
            
            
            % observation weighting
            fn = fieldnames(param);
            bool = strcmpi(fn,'ObservationWeight') | strcmpi(fn,'ObservationWeights');
            if any(bool)
                obj.ObservationWeight = param.fn(find(bool,1));
            end
            
            %% transform two-column dependent variable into one-column
            if BinaryCountCode % if binary observations with one column for counts of value 1 and one column for total counts
                if any(T(:,1)<0)
                    error('counts should be non-negative values');
                end
                if any(T(:,1)>T(:,2))
                    error('values in the second column should be larger or equal to values in the first');
                end
                if ~isempty(obj.ObservationWeight)
                    error('two-column dependent variable is not compatible with observation weights');
                end
                
                w = [T(:,1) T(:,2)-T(:,1)]'; % first row: value 1, then value 0
                nRep = sum(w>0,1); % if there is observation both for 0 and/or for 1 (needs to replicate rows)
                
                T = repmat([1;0],1, n); % observation for each
                T = T(w>0); % only keep
                obj.ObservationWeight = w(w>0);
                
                n = sum(nRep);
                for m=1:nMod
                    RR = cell(1,ndims(M(m).val));
                    RR{1} = nRep;
                    for d=2:ndims(M(m).val)
                        RR{d} = 1;
                    end
                    M(m).val =repelem(M(m).val,RR{:});
                    M(m).nObs = n;
                end
                obj.nObs = n;
                
            end
            obj.T = T;
            
            %% parse parameters
            if isfield(param,'observations')
                obs = param.observations;
                obs = strrep(obs, 'poisson','count');
                obs = strrep(obs, 'binary','binomial');
                assert(any(strcmp(obs, {'binomial','count'})), 'incorrect observation type: possible types are ''binomial'' and ''count''');
            else
                obs = 'binomial';
            end
            if strcmp(obs,'binomial') && any(T~=0 & T~=1)
                error('for binomial observations, T values must be 0 or 1');
            elseif strcmp(obs, 'count') && any(T<0)
                error('for count observations, all values must be non-negative');
            end
            if all(T==0) || all(T==1)
                warning('T values are all 0 or 1, may cause problem while fitting');
            end
            obj.obs = obs;
            
            % add constant bias ('on' by default)
            if ~isfield(param,'constant') || strcmpi(param.constant, 'on')
                %  Mconst = struct('val',{{ones(n,1)}}, 'name','bias','ndim',1,'size',1, 'sparse',0);
                Mconst = regressor(ones(n,1),'linear','label','offset');
                % Mconst.val = tocell(Mconst.val);
                M = [M,Mconst]; %structcat(M, Mconst); % append this component
                nMod = nMod +1;
                %   rank(nMod) = 1;
                param.constant = 'off';
            end
            
            obj.nMod = nMod;
            
            if isfield(param,'rank')
                rank = param.rank;
                for m=1:obj.nMod
                    M(m).rank = rank(m);
                end
            end
            
            for m=1:obj.nMod
                rr = M(m).rank;
                dd = M(m).nDim; % dimension in this module
                
                % weight constraint
                if ~isempty(M(m).constraint)
                    if ~all(any(M(m).constraint(:)== ['fbms1n' 0],2))
                        error('Field constraint must be composed of the following characters: ''f'', ''b'',''m'',''s'',''1'',''n''');
                    end
                    if rr>1 && size(M(m).constraint,1)==1
                        M(m).constraint(2:rr,1:M(m).nDim) = repmat(M(m).constraint,rr-1,1);
                    end
                else % default: first dimension is free, for other dimensions average is one (add free for additional regressors)
                    M(m).constraint = repmat('f',rr,dd); % all free by default
                end
            end
            
            %initial value of weights
            for m =1:nMod
                if ~isempty(M(m).U)
                    if M(m).rank>1 && size(M(m).U,1)==1 % if multiple rank and only weight is provided for rank 1
                        M(m).U(2:M(m).rank,:) = cell(M(m).rank-1,M(m).nDim); % .. start the other ones from default values
                    end
                else
                    M(m).U = cell(M(m).rank,M(m).nDim);
                end
            end
            
            obj.param = param;
            obj.regressor = M;
            
            %% split model for population analysis
            if isfield(param, 'split')
                obj = split(obj, param.split);
            end
        end
        
        %% SELECT SUBSET OF OBSERVATIONS FROM MODEL
        function obj = extract_observations(obj,subset)
            obj.T = obj.T(subset);
            if ~isempty(obj.ObservationWeight)
                obj.ObservationWeight = obj.ObservationWeight(subset);
            end
            n_obs = length(obj.T);
            
            obj.nObs = n_obs;
            for m=1:length(obj.regressor) % for each module
                obj.regressor(m) = extract_observations(obj.regressor(m), subset);
                %  obj.regressor(m).val =   extract_observations(obj.regressor(m).val,subset); % extract for this module
                %  obj.regressor(m).nObs = n_obs;
            end
            
            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'rho') && ~isempty(obj.Predictions.rho)
                obj.Predictions.rho = obj.Predictions.rho(subset);
            end
            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'Expected') && ~isempty(obj.Predictions.rho)
                obj.Predictions.Expected = obj.Predictions.Expected(subset);
            end
            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'sample') && ~isempty(obj.Predictions.rho)
                obj.Predictions.sample = obj.Predictions.rho(sample);
            end
        end
        
        %% SPLIT OBSERVATIONS IN MODEL
        function  objS = split(obj, S)
            if ~isvector(S) || length(S)~=obj.nObs
                error('splitting variable must be a vector of the same length as number of observation in model');
            end
            
            V = unique(S);
            
            % preallocate as series of regressors
            objS = repmat(gum(),1, length(V));
            
            for v=1:length(V)
                % subset of observations
                subset = find(S == V(v));
                
                objS(v) = extract_observations(obj,subset);
            end
            
        end
        
        %% EVALUATE MODEL
        function obj = evaluate(obj)
            %%% !!!! THIS DOES NOT SEEM TO WORK
            
            % whether we skip fitting
            %compute score on testing set (mean log-likelihood per observation)
            [validationscore,~, accuracy] = loglike(obj.regressor,obj.T,obj.ObservationWeight,obj.obs);
            obj.score.testscore = validationscore;
            obj.score.accuracy = accuracy;
        end
        
        %% FIT HYPERPARMETERS (and estimate) %%%%%%%%%%%
        function obj = fit(obj, param)
            
            if nargin==1
                param = struct;
            end
            
            if length(obj)>1
                %% fitting various models at a time
                
                for i=1:numel(obj)
                    obj(i) = obj(i).fit(param);
                end
                return;
            end
            
            M = obj.regressor;
            
            %% which order for components
            for m=1:obj.nMod
                if isempty(M(m).ordercomponent) && M(m).nDim>1
                    dd = M(m).nDim; % dimension in this module
                    cc = M(m).constraint; % constraint for this module
                    M(m).ordercomponent = all(all(cc(:,1:dd)==cc(1,1:dd))); % default: reorder if all components have same constraints
                end
            end
            
            % level of verbosity
            if isfield(param, 'verbose')
                obj.param.verbose = param.verbose;
            else
                obj.param.verbose = 'on';
            end
            if isfield(param, 'display')
                display = obj.param.display;
            else
                display = 'iter';
            end
            
            %%  check fitting method
            if isfield(param, 'HPfit') % if specified as parameter
                HPfit = param.HPfit;
                assert(ischar(HPfit) && any(strcmpi(HPfit, {'em','cv','basic'})), 'incorrect value for field ''HPfit''');
            else
                HPfit = 'basic'; % by default, no hyperparameters optimization
            end
            
            if isfield(param, 'gradient')
                use_gradient = param.gradient;
                assert(ischar(use_gradient) && any(strcmpi(use_gradient, {'on','off','check'})), 'incorrect value for field ''gradient''');
            else
                use_gradient = 'off';
            end
            
            %% cross validation parameters
            if isfield(param, 'crossvalidation') && ~isempty(param.crossvalidation)
                CV = param.crossvalidation;
                if iscell(CV) % permutationas are already provided as nperm x 2 cell (first col: train set, 2nd: train set)
                    %  generateperm = 0;
                    
                    if any([CV{:}]>obj.nObs)
                        error('maximum crossvalidation index (%d) is larger than number of observations (%d)',max([CV{:}]),n);
                    end
                    
                    CV = struct;
                    CV.NumTestSets = size(CV,1); % number of permutation sets
                    CV.nTrain = length(CV{1,1});
                    CV.nTest = length(CV{1,2});
                    CV.training = CV(:,1)';
                    CV.test = CV(:,2)';
                elseif isa(CV,'cvpartition')
                    %  generateperm = 0;
                    assert(CV.NumObservations==obj.nObs, 'number of data points in cross-validation set does not match number of observations in dataset');
                    %  nPerm = CV.NumTestSets;
                    %  CV = cell(nPerm,2);
                    %  for p=1:nPerm
                    %      CV{p,1} = find(CV.training(p))'; % training set in first column
                    %      CV{p,2} = find(CV.test(p))'; % test set in second colun
                    %  end
                    % ntrain = length(allperm{1,1});
                    % ntest = length(allperm{1,2});
                    
                    if CV.NumObservations>obj.nObs
                        error('number of observations in CV set is larger than number of observations (%d)',CV.NumObservations,n);
                    end
                    
                else % draw the permutations now
                    nTrain = CV; % number of observations in training set
                    %  generateperm = 1;
                    
                    if nTrain == -1
                        nTrain = n-1; % leave one out
                    elseif nTrain<1 % defined as percentage
                        nTrain = round(nTrain*n);
                    end
                    if isfield(obj.param, 'ntest') % defined number of test observations
                        nTest = obj.param.ntest;
                    else % by default all observations not in train set
                        nTest = n-nTrain;
                    end
                    if isfield(obj.param, 'nperm')
                        NumTestSets = obj.param.nperm;
                    else
                        NumTestSets = 10; % default number of permutations
                    end
                    %  CV = cell(NumTestSets,2); % first col: training sets; second col: testing sets
                    CV = struct;
                    CV.training = cell(1,NumTestSets);
                    CV.test = cell(1,NumTestSets);
                    
                    for p=1:NumTestSets % for each permutation
                        trainset = randperm(n,nTrain); % training set
                        notrainset= setdiff(1:n, trainset); % observations not in training set
                        testset = notrainset(randperm(n-nTrain,nTest)); % test set
                        %CV(p,:) = {trainset, testset};
                        CV.training{p} = trainset;
                        CV.test{p} = testset;
                    end
                end
                obj.param.crossvalidation = CV;
            end
            if isfield(obj.param,'testset') && any(obj.param.testset>obj.nObs)
                error('maximum test set index (%d) is larger than number of observations (%d)',max(obj.param.testset),obj.nObs);
            end
            
            %% run optimization over hyperparameters
            
            %if HPfit %% if fitting hyperparameters
            HPini = cell(1,obj.nMod); % cell array of hyperparameters initial values
            HP_LB = cell(1,obj.nMod); % HP lower bounds
            HP_UB = cell(1,obj.nMod); % HP upper bounds
            HP_fittable =cell(1,obj.nMod); % which HP are fitted
            nHP  = cell(1,obj.nMod); % number of hyperparameters in each module
            idx = cell(1,obj.nMod);     % index of hyperparameters for each component
            cnt = 0;
            for m=1:obj.nMod
                ss = M(m).nRegressor;
                idx{m} = cell(size(M(m).HP,1),M(m).nDim);
                
                
                %% check initial values and bounds for HPs
                HPini{m} = [M(m).HP.HP]; % initial values for all HPs
                HP_LB{m} = [M(m).HP.LB];
                HP_UB{m} = [M(m).HP.UB];
                HP_fittable{m} = [M(m).HP.fit];
                
                %retrieve number of fittable hyperparameters for each function
                this_HPfit = reshape({M(m).HP.fit}, size(M(m).HP));
                nHP{m} = cellfun(@sum, this_HPfit); %cellfun(@length, HPini{m});
                for d=1:M(m).nDim
                    for r=1:size(nHP{m},1)
                        idx{m}{r,d} = cnt + (1:nHP{m}(r,d)); % indices of hyperparameters for this component
                        cnt = cnt + nHP{m}(r,d); % update counter
                    end
                end
                
                M(m).covfun = tocell(M(m).covfun);
                
                % value of scale along each dimensions (this is just for computing
                % covariance - at the moment for spectral trick only)
                M(m).scale = tocell(M(m).scale);
                for f=1:M(m).nDim
                    if length(M(m).scale)<f || isempty(M(m).scale{f})
                        M(m).scale{f} = 1:ss(f); % by default, just integers
                    end
                end
                
                % spectral trick
                
                % whether some components use spectral decomposition
                spc =  ~isempty(M(m).spectral);
                %                 if spc && ~iscell(M(m).spectral) && isempty(M(m).spectral)
                %                     spc= 0;
                %                 elseif spc && iscell(M(m).spectral) && all(cellfun(@isempty,M(m).spectral))
                %                     spc = 0;
                %                 end
                
                %spectral(mm) = spc;
                if spc
                    %                     if isnumeric(M(m).spectral)
                    %                         spect_comp = find(M(m).spectral); % component to be converted to spectral domain
                    %                         M(m).spectral = num2cell(M(m).spectral);
                    %                         M(m).spectral(~spect_comp) = {[]};
                    %                     end
                    
                    if rank(m)>1
                        for d=1:M(m).nDim
                            if ~isempty(M(m).spectral(d).fun) && ~all(cellfun(@isempty, M(m).covfun(2:end,d)))
                                error('Applying spectral analysis to component %d not possible if covariance function is provided for rank higher than one',d);
                            end
                        end
                    end
                    
                    %  else
                    %      M(m).spectral = {};
                end
                
            end
            
            
            if sum(cellfun(@(x) sum(x,'all'),nHP))==0 % no hyperparameter
                fprintf('required optimization of hyperparameters but the model has no hyperparameter!\n' );
                return;
            end
            
            
            HPini = [HPini{:}];
            HP_LB = [HP_LB{:}]; % concatenate over modules
            HP_UB = [HP_UB{:}];
            HP_fittable = [HP_fittable{:}];
            
            % select hyperparmaters to be fitted
            HPini = HPini(HP_fittable);
            HP_LB = HP_LB(HP_fittable);
            HP_UB = HP_UB(HP_fittable);
            
            %  n_hyper_tot = cellfun(@sum, n_hyper); % total number of parameters in each module
            
            %% optimization parameters
            
            if isfield(param, 'maxiter_HP')
                maxiter = param.maxiter_HP;
            else
                maxiter = 200;
            end
            if isfield(param, 'initialpoints')
                obj.param.initialpoints = param.initialpoints;
            end
            
            
            HP_TolFun = 1e-2; % stopping criterion for hyperparameter fitting
            
            %% optimize hyperparameters
            switch lower(HPfit)
                case 'basic' % grid search  hyperpameters that maximize marginal evidence
                    
                    % clear persisitent value for best-fitting parameters
                    gum_neg_marg();
                    
                    
                    % run gradient descent
                    errorscorefun = @(HP) gum_neg_marg(obj, HP, idx);
                    optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'display',display,'MaxIterations',maxiter);
                    HP = fmincon(errorscorefun, HPini,[],[],[],[],HP_LB,HP_UB,[],optimopt); % optimize
                    
                    %% run estimation again with the optimized hyperparameters to retrieve weights
                    [~, obj] = errorscorefun(HP);
                    
                    
                    %% expectation-maximization to find  hyperpameters that maximize marginal evidence
                case 'em'
                    
                    if any([M.rank]>1)
                        error('EM not coded yet for rank larger than one');
                    end
                    
                    old_logjoint = -Inf;
                    logjoint_iter = zeros(1,maxiter);
                    still =1;
                    iter = 0;
                    % UU = []; % initial values for weights
                    HP = HPini; % initial values for hyperparameters
                    param2 = obj.param;
                    param2.HPfit = 'none';
                    param2.crossvalidation = [];
                    param2.spectralback = 0;
                    
                    objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient
                    check_grad = strcmpi(use_gradient,'check'); % whether to use gradient
                    
                    param_tmp = obj.param;
                    obj.param = param2;
                    
                    while still % EM iteration
                        %% E-step (running inference with current hyperparameters)
                        
                        M = assign_hyperparameter(M, HP, idx);
                        
                        % evaluate covariances at for given hyperparameters
                        % [M2, spectral, spectc] = covfun_eval(M);
                        M2 = compute_prior_covariance(M);
                        
                        PP = projection_matrix(M2); % projection matrix for each dimension
                        
                        % estimate weights from GUM
                        %[M2, S] = gum(M2,T,param2);
                        
                        if iter==1
                            param2.initialpoints = 1;
                        end
                        
                        obj2 = obj.infer(param2);
                        
                        %% M-step (adjusting hyperparameters)
                        midx = 0;
                        allidx = [idx{:}]; % concatenate over components
                        ee = 1;
                        for m=1:obj.nMod
                            ss = M(m).nFreeParameters; % size of each dimension
                            
                            for d=1:M(m).nDim
                                for r=1:size(idx{m},1)
                                    id = idx{m}{r,d};
                                    if ~isempty(id)
                                        if ~any(ismember(id, [allidx{setdiff(1:obj2.nMod,ee)}])) %% if hyperparameters are not used in any other module
                                            if ~isa(M(m).covfun{1,d}, 'function_handle') % function handle
                                                error('hyperparameters with no function');
                                            end
                                            cc = idx{m}{r,d};  %index for corresponding module
                                            
                                            HPs = M(m).HP(r,d);
                                            
                                            
                                            % hyperparameter values for this component
                                            HP_fittable = HPs.fit;
                                            HPs.HP(HP_fittable) = HP(cc); % fittable values
                                            [~, gg] = obj2.regressor(m).covfun{r,d}(HPs.HP);
                                            
                                            % reg_idx = (1:ss(d)) + (r-1)*ss(d) + rank(m)*sum(ss(1:d-1)) + midx; % index of regressors in design matrix
                                            
                                            % posterior mean and covariance for associated weights
                                            this_mean =  obj2.regressor(m).U{r,d};
                                            reg_idx = (1:ss(r,d)) + sum(ss(:,1:d-1),'all') + sum(ss(1:r-1,d)) + midx; % index of regressors in design matrix
                                            
                                            % !!!  CHECK THIS !!!!!
                                            %     this_cov =   PP{m}{1,d}' * S.covb_free(reg_idx,reg_idx) * PP{m}{1,d}; % !! check this
                                            this_cov =  obj2.score.covb_free(reg_idx,reg_idx) ; % !! check this
                                            
                                            
                                            if isstruct(gg)  && isequal(PP{m}{r,d}, eye(ss(r,d)))  % function provided to optimize hyperparameters
                                                %  work on this to also take into account constraints (marginalize in the free base domain)
                                                
                                                HPnew = gg.EM(this_mean,this_cov); % find new set of hyperparameters
                                            else % find HP to maximize cross-entropy between prior and posterior
                                                
                                                optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                                                    'CheckGradients',check_grad,'display','off','MaxIterations',1000);
                                                
                                                assert(~isinf(mvn_negxent(M(m).covfun{1,d},obj2.regressor(m).mu{r,d}, this_mean, this_cov,PP{m}{r,d}, HP(id), HPs)), ...
                                                    'M step cannot be completed, covariance prior is not full rank');
                                                
                                                HPnew = fmincon(@(P) mvn_negxent(M(m).covfun{1,d}, obj2.regressor(m).mu{r,d}, this_mean, this_cov,PP{m}{r,d}, P, HPs),...
                                                    HP(id),[],[],[],[],M(m).HP(r,d).LB(HP_fittable),M(m).HP(r,d).UB(HP_fittable),[],optimopt);
                                            end
                                            HP(id) = HPnew;  % select values corresponding to fittable HPs
                                            
                                        else
                                            error('not coded: should optimize over various components at same time');
                                        end
                                    end
                                    ee = ee+1;
                                end
                                
                            end
                                                            midx = midx + sum(ss(1,:)) * rank(m); % jump index by number of components in module
                        end
                        
                        % for spectral decomposition, convert weights back from spectral to
                        % original domain
                        %M2 = weightprojection(M2,obj.score.covb,obj.nMod,spectral,spectc,rank);
                        M2 = project_from_spectral(M2);
                        
                        % has converged if improvement in LLH is smaller than epsilon
                        iter = iter + 1; % update iteration counter;
                        LogEvidence = obj2.score.LogEvidence;
                        fprintf('HP fitting: iter %d, log evidence %f\n',iter, LogEvidence);
                        % HP
                        converged = abs(old_logjoint-LogEvidence)<HP_TolFun;
                        old_logjoint = LogEvidence;
                        logjoint_iter(iter) = LogEvidence;
                        
                        still = (iter<maxiter) && ~converged;
                        
                    end
                    obj.regressor = M2;
                    obj.param = param_tmp;
                    
                case 'cv' % gradient search to minimize cross-validated log-likelihood
                    
                    % clear persisitent value for best-fitting parameters
                    gum_score();
                    
                    objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient
                    check_grad = strcmpi(use_gradient,'check'); % whether to use gradient
                    
                    % run gradient descent
                    errorscorefun = @(P) gum_score(obj, P, idx);
                    optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                        'CheckGradients',check_grad,'display',display,'MaxIterations',maxiter);
                    HP = fmincon(errorscorefun, HPini,[],[],[],[],HP_LB,HP_UB,[],optimopt); % optimize
                    
                    %% run estimation again with the optimized hyperparameters to retrieve weights
                    [~, ~, obj] = errorscorefun(HP);
            end
            
            
            % allocate fitted hyperparameters to each module
            for m=1:obj.nMod
                for d=1:M(m).nDim
                    for r=1:size(M(m).HP,1)
                        % M(m).HP{d} = HP(idx{m}{d}); % associate each parameter to corresponding module
                        this_HP_fittable = M(m).HP(r,d).fit;
                        M(m).HP(r,d).HP(this_HP_fittable) = HP(idx{m}{r,d}); % associate each parameter to corresponding module
                    end
                end
            end
            
            % fn = fieldnames(S);
            % for f=1:length(fn)
            %     obj.(fn{f}) = S.(fn{f});
            % end
            
        end
        
        
        
        %% %%%%% INFERENCE (ESTIMATE WEIGHTS) %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function   obj = infer(obj, param)
            
            if nargin==1
                param = struct();
            end
            
            if length(obj)>1
                %% fitting various models at a time
                for i=1:numel(obj)
                    obj(i) = obj(i).infer(param);
                end
                return;
            end
            
            if isempty(obj.regressor)
               error('the model has no regressor'); 
            end
            
            tic;
            
            %% parse parameters
            if ~isfield(param, 'maxiter')
                param.maxiter = 500; % maximum number of iterations
            end
            if ~isfield(param, 'miniter')
                param.miniter = 4; % minimum number of iterations
            end
            if ~isfield(param, 'TolFun')
                param.TolFun = 1e-9;
            end
            if ~isfield(param, 'initialpoints') % number of initial points for inference algorithm
                param.initialpoints = 10;
            end
            
            if ~isfield(param, 'verbose')
                param.verbose = 'on';
            else
                assert(ismember(param.verbose, {'on','off','little','full'}), 'incorrect value for field ''verbose'': possible values are ''on'', ''off'',''full'' and ''little''');
            end
            verbose =strcmp(param.verbose, 'on') || strcmp(param.verbose, 'full');
            
            obj.param = param;
            
            if isfield(param, 'gradient_hyperparameters') % whether we compute gradient for hyperparameters
                GHP = param.gradient_hyperparameters; % cell with how prior for each weight dimension depends on each hyperparameter
                do_grad_hyperparam = 1;
            else
                GHP = [];
                do_grad_hyperparam = 0;
            end
            
            nM = obj.nMod;
            M = obj.regressor;
            
            if isglm(obj)
                obj.param.initialpoints = 1; % if no multilinear term, problem is convex so no local minima
            end
            
            
            %% evaluate prior covariances for given hyperparameters
            if verbose
                fprintf('Evaluating prior covariance matrix...');
            end
            
            M = compute_prior_covariance(M);
            
            if verbose
                fprintf('done\n');
            end
            %M = numberofparameters(M, rank); % number of parameters and free parameters per component
            
            %% compute prior mean and covariance and initialize weight
            for m=1:nM
                M(m) = M(m).check_prior_covariance();
                M(m) = M(m).compute_prior_mean();
                M(m) = M(m).initialize_weights();
            end
            obj.regressor = M;
            
            singular_warn = warning('off','MATLAB:nearlySingularMatrix');
            
            % check hyper prior gradient matrix has good shape
            if do_grad_hyperparam
                nParamTot = 0;  % total number of weights
                for m=1:nM
                    nParamTot = nParamTot + nParametersTot(obj); % total number of weights
                end
                if size(GHP,1)~=nParamTot || size(GHP,2)~=nParamTot
                    error('The number of rows and columns in field ''gradient_hyperparameters'' (%d) must match the total number of weights (%d)',size(GHP,1),nParamTot);
                end
                
            end
            
            %% fit weights on full dataset implement the optimization (modified IRLS) algorithm
            [obj, Sfit] = IRLS(obj);
            M = obj.regressor;
            
            % score on test set
            if isfield(obj.param, 'testset') && ~isempty(obj.param.testset)
                testset = obj.param.testset;
                
                % [testscore,~, accuracy_test] = loglike(extract_observations(M,testset), obj.T(testset),obj.ObservationWeight(testset),obj.obs) ;
                [testscore,~, accuracy_test] = loglike(extract_observations(obj,testset)) ;
                
                if isempty(obj.ObservationWeight)
                    n_Test = length(testset);
                else
                    n_Test = sum(obj.ObservationWeight(testset));
                end
                testscore= testscore/ n_Test; % normalize by number of observations
            end
            
            
            %% cross validation
            crossvalid = isfield(param, 'crossvalidation') && ~isempty(param.crossvalidation);
            if crossvalid
                CV = param.crossvalidation;
                nSet = CV.NumTestSets;
                variance = cell(1,nM);
                for m=1:nM
                    obj.regressor(m).U_CV = cell(rank(m),M(m).nDim); % fitted parameters for each permutation
                    for d=1:obj.regressor(m).nDim
                        for r=1:rank(m)
                            obj.regressor(m).U_CV{r,d} = zeros(nSet,obj.regressor(m).nRegressor(d));
                        end
                    end
                    variance{m} = zeros(nSet,obj.regressor(m).rank); % variance across observations for each component
                    
                end
                % U_CV{1,D+1} = zeros(nperm,m(D+1));
                validationscore = zeros(1,nSet); % score for each (LLH per observation)
                accuracy = zeros(1,nSet); % proportion correct
                exitflag_CV = zeros(1,nSet); % exit flag (converged or not)
                
                if do_grad_hyperparam
                    grad_hp = zeros(size(GHP,3),nSet); % gradient matrix (hyperparameter x permutation)
                    
                    PP = projection_matrix(M,'all'); % global transformation matrix from full parameter set to free basis
                    %    PP = [PP{:}]; % concatenate over modules
                    %    PP = blkdiag(PP{:});
                else
                    PP = []; % just for parfor
                end
                
                spmd
                    warning('off','MATLAB:nearlySingularMatrix');
                end
                
                % global covariance
                %K = {M.sigma};
                
                
                
                % parfor p=1:nperm % for each permutation
                for p=1:nSet % for each permutation
                    
                    if verbose
                        fprintf('Cross-validation set %d/%d', p, nSet);
                    end
                    
                    % if generateperm % generate permutation
                    %     this_ntrain = nTrain;
                    %     trainset = randperm(n,this_ntrain); % training set
                    %     notrainset= setdiff(1:n, trainset); % observations not in training set
                    %     validationset = notrainset(randperm(n-nTrain,nTest)); % test set
                    % else
                    trainset = CV.training(p);
                    validationset = CV.test(p);
                    if iscell(trainset) % CV structure
                        trainset = trainset{1};
                        validationset = validationset{1};
                        % else % CVpartition object: convert boolean to indices
                        %     trainset = find(trainset);
                        %     validationset = find(validationset);
                    end
                    
                    %    traintest = allPerm(p,:);
                    %    trainset = traintest{1}; %allperm{p,1};
                    %    validationset = traintest{2};%allperm{p,2};
                    %     this_ntrain = length(trainset);
                    %  end
                    
                    %fit on training set
                    obj_train = IRLS(extract_observations(obj,trainset));
                    
                    exitflag_CV(p) = obj_train.score.exitflag;
                    
                    for m=1:nM
                        for d=1:M(m).nDim
                            for r=1:M(m).rank
                                obj.regressor(m).U_CV{r,d}(p,:) = obj_train.regressor(m).U{r,d};
                            end
                        end
                    end
                    
                    
                    % Ua = {obj_train.U};
                    %Ucat = [Ua{:}]; % concatenate weights
                    %U_all(p,:) = Ua;
                    
                    % Ua = cell(1,nMod);
                    % for m=1:nMod
                    %     Ua{m} = this_M(m).U;
                    % end
                    % concatenate weights
                    
                    Ucat = concatenate_weights(obj_train);
                    % U_all(p,:) = Ucat;
                    %  U_all{p} = this_M;
                    
                    s = 1; % warning: change for normal observations
                    
                    %compute score on testing set (mean log-likelihood per observation)
                    obj_v = extract_observations(obj,validationset); % model with validat
                    for m=1:nM
                        obj_v.regressor(m).U = obj_train.regressor(m).U;
                    end
                    [validationscore(p),grad_validationscore, accuracy(p)] = loglike(obj_v) ;
                    
                    if isempty(obj.ObservationWeight)
                        n_Validation = length(validationset);
                    else
                        n_Validation = sum(obj.ObservationWeight(validationset));
                        
                    end
                    validationscore(p) = validationscore(p)/ n_Validation; % normalize by number of observations
                    grad_validationscore = grad_validationscore / n_Validation; % gradient w.r.t each weight
                    if do_grad_hyperparam
                        
                        H = PosteriorCov(obj_train);
                        this_gradhp = zeros(1,size(GHP,3));
                        for q=1:size(GHP,3) % for each hyperparameter
                            gradgrad = PP*GHP(:,:,q)'*Ucat'; % LLH derived w.r.t to U (free basis) and hyperparameter
                            gradU = - PP' * H * gradgrad;% derivative of inferred parameter U w.r.t hyperparameter (full parametrization)
                            this_gradhp(q) = grad_validationscore * gradU/s; % derivate of score w.r.t hyperparameter
                        end
                        grad_hp(:,p) = this_gradhp;
                    end
                    
                    if verbose
                        fprintf('done\n');
                    end
                    
                end
                
                warning(singular_warn.state,'MATLAB:nearlySingularMatrix');
                
                %  spmd
                %      warning(singular_warn.state,'MATLAB:nearlySingularMatrix');
                %  end
                
                n_nonconverged = sum(exitflag_CV<=0);
                if  n_nonconverged>0
                    warning('gum:notconverged', 'Failed to converge for %d/%d permutations', n_nonconverged, nSet);
                end
                
                
                %                 for m=1:nM
                %                     for p=1:nSet
                %                         for d=1:rank(m)*M(m).nDim
                %                             obj.regressor(m).U_CV{d}(p,:) = U_all{p,m}{d};
                %                         end
                %                     end
                %
                %                     %         % weight as means over cross validation
                %                     %         M(m).U = cell(rank(m),M(m).nDim);
                %                     %         for d=1:rank(m)*M(m).nDim
                %                     %             M(m).U{d} = mean(M(m).U_CV{d},1); % mean value of parameters over permutation
                %                     %         end
                %                 end
                
                %   S.U_CV = U_CV; %cat(2,U_CV{:}); %group all parameter estimates
                
                S.validationscore = mean(validationscore);
                S.validationscore_all = validationscore;
                S.accuracy_validation = mean(accuracy);
                S.accuracy_all = accuracy;
                S.exitflag_CV = exitflag_CV;
                S.converged_CV = sum(exitflag_CV>0); % number of permutations with convergence achieved
                % S.variance_CV = variance_CV;
                if isfield(obj.param, 'testset')
                    S.testscore = testscore;
                    S.accuracy_test = accuracy_test;
                end
                if do_grad_hyperparam
                    S.grad = mean(grad_hp,2); % gradient is mean of gradient over permutations
                end
                
            elseif do_grad_hyperparam % optimize parameters directly on whole dataset
                
                PP = projection_matrix(M,'all'); % projection matrix for each dimension
                % PP = cat(2,PP{:}); % concatenate over modules
                % PP = blkdiag(PP{:}); % global transformation matrix from full parameter set to free basis
                
                %compute score on whole dataset (mean log-likelihood per observation)
                [validationscore,grad_validationscore, S.accuracy] = loglike(obj) ;
                
                if isempty(obj.ObservationWeight)
                    nWeightedObs = obj.nObs;
                else
                    nWeightedObs = sum(obj.ObservationWeight);
                end
                S.validationscore = validationscore/ nWeightedObs; % normalize by number of observations
                grad_validationscore = grad_validationscore / nWeightedObs; % gradient w.r.t each weight
                
                Ua = {M.U};
                Ucat = [Ua{:}]; % concatenate weights
                
                [~,H] = PosteriorCov(obj); % Hessian w.r.t weights
                grd = zeros(size(GHP,3),1);
                for q=1:size(GHP,3) % for each hyperparameter
                    % Ucat = cellfun(@(x) x.U, M, 'uniformoutput',0);
                    % Ucat = [Ucat{:}];
                    warning('looks like should be derivative for inverse of GHP here');
                    gradgrad = [Ucat{:}]*GHP(:,:,q)*PP'; % LLH derived w.r.t to U (free basis) and hyperparameter
                    gradU = - PP' * H * gradgrad';% derivative of inferred parameter U w.r.t hyperparameter (full parametrization)
                    grd(q) = grad_validationscore * gradU; % derivate of score w.r.t hyperparameter
                end
                obj.grad = grd;
            end
            
            if  Sfit.exitflag<=0 && (~crossvalid || n_nonconverged==0)
                warning('gum:notconverged', 'Failed to converge');
            end
            
            
            % Posterior Covariance
            if verbose
                fprintf('Computing posterior covariance...');
            end
            [obj, S.covb_free, B]= PosteriorCov(obj);
            if verbose
                fprintf('done\n');
            end
            
            %Covariance
            P = projection_matrix(M,'all'); % projection matrix for each dimension
            %   P = [PP{:}]; % concatenate over modules
            %   P = blkdiag(P{:}); % global transformation matrix from full parameter set to free basis
            S.covb = P'*S.covb_free* P; % see covariance under constraint Seber & Wild Appendix E
            
            
            % standard error of estimates
            all_se = sqrt(diag(S.covb))';
            
            % T-statistic for the weights
            Ucat = cellfun(@(x) [x{:}], {M.U}, 'uniformoutput',0);
            % Ucat = [Ucat{:}];
            allU = cat(2,Ucat{:});
            all_T = allU ./ all_se;
            
            % p-value for significance of each coefficient
            %all_p = 1-chi2cdf(all_T.^2,1);
            all_p = 2*normcdf(-abs(all_T));
            
            % distribute values of se, T, p and V to different regressors
            midx = 0;
            for m=1:obj.nMod
                rr = rank(m);
                ss = M(m).nRegressor;
                dd = M(m).nDim;
                
                M(m).se = cell(1,dd);
                M(m).T = cell(1,dd);
                M(m).p = cell(1,dd);
                M(m).V = cell(1,dd);
                
                for d=1:dd
                    for r=1:M(m).rank
                        idx = (1:ss(d)) + (r-1)*ss(d) + rr*sum(ss(1:d-1)) + midx; % index of regressors in design matrix
                        M(m).se{d}(r,:) = all_se(idx);
                        M(m).T{d}(r,:) = all_T(idx);
                        M(m).p{d}(r,:) = all_p(idx);
                        M(m).V{d}(:,:,r) = S.covb(idx,idx);
                        
                    end
                end
                
                
                %                 idx = midx + (1:rr*sum(ss)); % index of regressors for this module
                %                 % convert to cells
                %                 M(m).se = num2dimcell(all_se(idx),ss,rr);
                %                 M(m).T = num2dimcell(all_T(idx),ss,rr);
                %                 M(m).p = num2dimcell(all_p(idx),ss,rr);
                %
                midx = midx + sum(ss) * rr; % jump index by number of components in module
                
            end
            
            S.exitflag = Sfit.exitflag;
            S.exitflag_allstarting = Sfit.exitflag_allstarting;
            
            %% log-likelihood and approximation to log-evidence
            
            % LLH at inferred params
            [obj,S.LogLikelihood] = LogLikelihood(obj);
            obj = predictor_variance(obj);
            
            %if strcmp(obj.obs, 'binomial')
            %    S.LogLikelihood = sum(obj.ObservationWeight .* log(Y.*obj.T + (1-Y).*(1-obj.T)));
            %else % poisson
            %    S.LogLikelihood = sum(obj.ObservationWeight .* (obj.T.*log(Y)-Y-logfact(obj.T)));
            %end
            
            % number of free parameters
            nFreePar = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));
            
            %  nFreePar = [M.nFreeParameters]; %cellfun(@(x) sum(x(:)), M);% number of free parameters for each component
            %  nFreePar = sum(nFreePar);
            obj.score.nFreeParameters = nFreePar;
            
            % model evidence using Laplace approximation (Bishop - eq 4.137)  -
            % requires that a prior has been defined
            %LD = - 2*sum(log(diag(chol(B)))); % log-determinant of posterior covariance (fast reliable way)
            LD = logdet(B);
            %S.LogEvidence = S.LogLikelihood + nfreepar/2*log(2*pi) - logdet/2;
            S.LogEvidence = S.LogLikelihood - LD/2;
            
            PP = projection_matrix(M);
            for m=1:nM % add part from prior
                for d=1:M(m).nDim
                    for r=1:M(m).rank % assign new set of weight to each component
                        dif =  (M(m).U{r,d} - M(m).mu{r,d})*PP{m}{r,d}'; % distance from prior mean (projected)
                        this_cov = PP{m}{r,d}*M(m).sigma{r,d}*PP{m}{r,d}'; % corresponding covariance prior
                        % logdet = 2*sum(log(diag(chol(PP{m}{d}*this_cov*PP{m}{d}')))); % log-determinant of prior covariance (fast reliable way)
                        % S.LogEvidence = S.LogEvidence -  (dif/this_cov)*dif'/2  - logdet/2 -M(m).nfreepar(1,d)/2*log(2*pi); % log-prior for this weight
                        S.LogEvidence = S.LogEvidence -  (dif/this_cov)*dif'/2; % log-prior for this weight
                    end
                end
            end
            
            S.BIC = nFreePar*log(obj.nObs) -2*S.LogLikelihood; % Bayes Information Criterion
            S.AIC = 2*nFreePar - 2*S.LogLikelihood; % Akaike Information Criterior
            S.AICc = S.AIC + 2*nFreePar*(nFreePar+1)/(obj.nObs-nFreePar-1); % AIC corrected for sample size
            S.logjoint_allstarting = Sfit.logjoint_allstarting; % values for all starting points
            
            
            % model evidence using Laplace approximation (Bishop - eq 4.137)
            % S.LLH_model = S.LLH - U'*sigma1*U/2 - V'*sigma2*V/2 + (m1+m2)/2*log(2*pi) - log(det(Inff))/2;
            
            %  S.variance = variance;
            
            %     % for spectral decomposition, convert weights back from spectral to
            %     % original domain
            if ~isfield(param, 'spectralback') || ~param.spectralback
                %  M = weightprojection(M,S.covb,obj.nMod,spectral,spectc);
                M = project_from_spectral(M);
            end
            
            obj.regressor = M;
            
            %  obj = obj.clear_data;
            
            score_fields = fieldnames(S);
            for f=1:length(score_fields)
                obj.score.(score_fields{f}) = S.(score_fields{f});
            end
            
            obj.score.FittingTime = toc;
        end
        
        %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% IMPLEMENT THE LOG-JOINT MAXIMIZATION ALGORITHM (modified IRLS)
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function  [obj, S] = IRLS(obj)
            
            nM = obj.nMod; % number of modules
            n = obj.nObs; % number of observations
            M = obj.regressor;
            nD = [M.nDim]; % number of dimension for each module
            D = max([M.nFreeDimensions]);
            initialpoints = obj.param.initialpoints;
            maxiter = obj.param.maxiter;
            miniter = obj.param.miniter;
            TolFun = obj.param.TolFun;
            
            U_allstarting = cell(nM, initialpoints); % estimated weights for all starting points
            logjoint_allstarting = zeros(1,initialpoints); % log-joint for all starting points
            exitflag_allstarting = zeros(1,initialpoints); % exitflag for all starting points
            logjoint_hist_allstarting = cell(1,initialpoints); % history log-joint for all starting points
            
            %stepsize = 1; % should always stay 1 unless LLH decreases after some step(s)
            
            logprior = cell(1,nM);
            rank = zeros(1,nM);
            for m=1:nM
                rank(m) = M(m).rank;
                logprior{m} = zeros(rank(m),nD(m));
            end
            
            % order of updates for each dimensions
            UpdOrder = UpdateOrder(M);
            
            PP = projection_matrix(M); % free-to-full matrix conversion
            P = catadditional(PP, UpdOrder); % concatenate projection matrix corresponding to different modules for each dimension
            K = catadditional({M.sigma}, UpdOrder); % group covariance corresponding to different modules for each dimension (full parameter space)
            
            mu = cell(1,D);
            for d=1:D
                this_d = UpdOrder(:,d); % over which dimension we work for each model
                for m=1:nM
                    mu{d} = [mu{d} M(m).mu{:,this_d(m)}]; % add mean values of prior
                end
            end
            
            
            spcode = any(cellfun(@issparse, {M.val}));
            
            
            % number of regressors for each dimension
            nReg = zeros(1,D);
            nFree = zeros(1,D); % number of free parameters
            size_mod = zeros(nM,D); % size over each dimension for each module
            precision = cell(1,D); % for infinite variance
            KP = cell(1,D);
            Pmu = cell(1,D);
            for d=1:D
                nFree(d) = size(P{d},1);
                this_d = UpdOrder(:,d)'; %min(d,nDim); % over which dimension we work for each model
                for m=1:nM
                    size_mod(m,d) = M(m).nRegressor(this_d(m));
                    nReg(d) = nReg(d) + rank(m)*size_mod(m,d); % add number of regressors for this module
                end
                
                K{d} = P{d}*K{d}*P{d}'; % project onto free basis
                
                if any(isinf(K{d}(:))) % use precision only if there is infinite covariance (e.g. no regularization)
                    wrn = warning('off', 'MATLAB:singularMatrix');
                    precision{d} = inv(K{d});
                    warning(wrn.state, 'MATLAB:singularMatrix');
                elseif spcode
                    KP{d} = K{d}* P{d}; % we're going to need it several times
                end
                
                Pmu{d} = P{d}*mu{d}';
            end
            
            
            % dispersion parameter from exponential family
            s = 1;
            
            if D>1 % in case we use Newton update on full dataset
                % we create dummy 'regressor' variables to store the gradients and constant weights
                Udc_dummy = clear_data(M);
                %  Udc_dummy = B_dummy;
                B_dummy = [];
                
                % full prior covariance matrix
                Kall = global_prior_covariance(M);
                if any(isinf(Kall(:))) % use precision only if there is infinite covariance (e.g. no regularization)
                    wrn = warning('off', 'MATLAB:singularMatrix');
                    precision_all = inv(Kall);
                    warning(wrn.state, 'MATLAB:singularMatrix');
                end
                
                Pall = projection_matrix(M,'all'); % full-to-free basis
                Kall =  Pall*Kall*Pall'; % project onto free basis
                                nFree_all = size(Kall,1);
            end
            
            
            %% repeat fitting procedure with initial points
            for ip=1:initialpoints
                
                if initialpoints>1 && strcmp(obj.param.verbose, 'on')
                    fprintf('Fitting weights, starting point %d/%d\n',ip, initialpoints);
                    %  elseif initialpoints>1 && strcmp(obj.param.verbose,'little')
                    %      fprintf('*');
                end
                
                
                
                %% generate random starting points
                if ip>1
                    M = sample_weights_from_prior(M);
                    obj.regressor = M;
                end
                
                % initial value of log-joint
                obj = Predictor(obj);
                obj = LogJoint(obj);
                logjoint = obj.score.LogJoint;
                % old_logjoint = nan;
                % rho = Predictor(M);
                % logjoint = LogJoint(rho, T, w, obs,prior_score);
                % logjoint = -Inf;
                iter = 1;
                not_converged = true;
                logjoint_hist = zeros(1,maxiter);
                Uall = nan(size(concatenate_weights(M)));
                dLJ = nan;
                prev_dLJ = nan;
                dU = nan(size(concatenate_weights(M)));
                prev_dU = dU;
                
                % weighting by observations weights
                if isempty(obj.ObservationWeight)
                    weighted_fun = @(x) x;
                else
                    weighted_fun = @(x) x.*obj.ObservationWeight;
                end
                
                FullHessianUpdate = false;
                
                %% loop until convergence
                while not_converged
                    
                    old_logjoint = logjoint;
                    
                    
                    for d=1:D % for each dimension
                        this_d = UpdOrder(:,d)'; % min(d,nDim); % over which dimension we work for each model
                        
                        % concatenate set of weights for this dimension for each component and each module
                        UU = concatenate_weights(M, this_d);
                        
                        % design matrix (project tensors in all expcept
                        % this dimension)
                        Phi = design_matrix(M,[], this_d, 0);
                        
                        if iter==1 && d==1 && any(isnan(UU))
                            % make sure we stay in "safe" region of weight space
                            switch obj.obs
                                case 'count'
                                    obj.Predictions.rho = log(obj.T+.25);
                                case 'binomial'
                                    obj.Predictions.rho = log((obj.T+.5)./(1.5-obj.T));
                            end
                        elseif d==1 || ~FullHessianUpdate
                            obj.Predictions.rho = Phi*UU'; % predictor
                        end
                        
                        
                        if strcmp(obj.obs, 'binomial')
                            Y = 1 ./ (1 + exp(-obj.Predictions.rho)); % expected mean
                            R = Y .* (1-Y) ; % derivative wr.t. predictor
                        else % poisson
                            Y = exp(obj.Predictions.rho); % rate
                            R = Y;
                            R = min(R,1e10); % to avoid badly scaled hessian
                        end
                        
                        % remove constant parts from projected activation
                        [rho_proj, UconstU] = remove_constrained_from_predictor(M, this_d, obj.Predictions.rho, Phi, UU);
                        
                        % compute gradient
                        if ~FullHessianUpdate
                            G = weighted_fun(R .* rho_proj + (obj.T-Y));
                        else
                            G =  weighted_fun(obj.T-Y);
                        end
                        weights = spdiags(weighted_fun(R), 0, n, n);
                        
                        if ~spcode
                            PhiP = Phi*P{d}';
                        end
                        
                        if ~any(isinf(K{d}(:))) %finite covariance matrix
                            if spcode
                                B = KP{d}*(Phi'*G) + s*Pmu{d};
                            else
                                B = K{d}*(PhiP'*G) + s*Pmu{d};
                            end
                        else % any infinite covariance matrix (e.g. no prior on a weight)
                            if spcode
                                B = P{d}*(Phi'*G) + s*precision{d}*Pmu{d};
                            else
                                B = (PhiP'*G) + s*precision{d}*Pmu{d};
                            end
                        end
                        
                        
                        if ~FullHessianUpdate %% update weights just along that dimension
                            
                            % Hessian matrix on the free basis
                            if ~any(isinf(K{d}(:))) %finite covariance matrix
                                
                                if spcode
                                    HH = KP{d}*(Phi'*weights*Phi)*P{d}' + s*eye(nFree(d)); % should be faster this way
                                else
                                    HH = K{d}* PhiP'*weights*PhiP + s*eye(nFree(d)); % Hessian matrix on the free basis
                                end
                                
                            else % any infinite covariance matrix (e.g. no prior on a weight)
                                if spcode
                                    HH = P{d}*(Phi'*weights*Phi)*P{d} + s*precision{d}; % Hessian matrix on the free basis
                                else
                                    HH = PhiP'*weights*PhiP + s*precision{d}; % Hessian matrix on the free basis
                                end
                            end
                            
                            
                            Unu = (HH\B)' * P{d}; % new set of weights (projected back to full basis)
                            
                            
                            %  while strcmp(obs,'poisson') && any(Phi*(Unu+[repelem(U_const(1:rank),m(d)) zeros(1,m(D+1))])'>500) % avoid jump in parameters that lead to Inf predicted rate
                            
                            % step halving if required (see glm2: Fitting Generalized Linear Models
                            %    with Convergence Problems - Ian Marschner)
                            obj.Predictions.rho = Phi*(Unu'+UconstU);
                            %  while LogJoint(rho, T, w, obs,prior_score)<old_logjoint %% avoid jumps that decrease the log-joint
                            %      Unu = (UU+Unu)/2;  %reduce step by half
                            %      rho = Phi*(Unu'+UconstU);
                            %  end
                            while iter<4 && strcmp(obj.obs,'count') && any(abs(obj.Predictions.rho)>500) % avoid jump in parameters that lead to Inf predicted rate
                                Unu = (UU+Unu)/2;  %reduce step by half
                                obj.Predictions.rho = Phi*(Unu'+UconstU);
                            end
                            
                            
                            compute_logjoint = true;
                            cc = 0;
                            diverged = false;
                            
                            while compute_logjoint
                                obj.regressor = set_weights(M,Unu+UconstU', this_d);
                                
                                for m=1:nM
                                    d2 = this_d(m);
                                    logprior{m}(:,d2) = LogPrior(obj.regressor(m),d2); % log-prior for this weight
                                end
                                
                                % compute log-joint
                                [obj,logjoint] = LogJoint(obj,logprior);
                                
                                
                                compute_logjoint = (logjoint<old_logjoint-1e-3);
                                if compute_logjoint % if log-joint decreases,
                                    Unu = (UU-UconstU'+Unu)/2;  %reduce step by half
                                    obj.Predictions.rho = Phi*(Unu'+UconstU);
                                    
                                    cc = cc+1;
                                    if cc>100 %% if we have halved 100 times step to new weights, we probably have diverged
                                        diverged = true;
                                        break;
                                    end
                                end
                            end
                            M = obj.regressor;
                        else
                            %% prepare for Newton step in while parameter space
                            % B_dummy = set_weights(B_dummy, B, this_d);
                            B_dummy = set_free_weights(Udc_dummy,B', B_dummy, this_d);
                            Udc_dummy = set_weights(Udc_dummy, UconstU', this_d);
                            
                        end
                    end
                    
                    
                    %% DIRECT NEWTON STEP ON WHOLE PARAMETER SPACE
                    if FullHessianUpdate
                        
                        UU = concatenate_weights(M); % old set of weights
                        UconstU = concatenate_weights(Udc_dummy); % fixed weights
                        % B = concatenate_weights(B_dummy); % gradient over all varialves
                        B = cellfun(@(x) [x{:}], B_dummy,'unif',0);
                        B = [B{:}]';
                        
                        
                        Hess = Hessian(obj);
                        B = B + Kall*Hess*Pall*UU';  %% !!! check if projections are well done!!!
                        
                        
                        % Hessian matrix on the free basis
                        if ~any(isinf(Kall(:))) %finite covariance matrix
                            %   if spcode
                            %       HH = KPfull*(Phi'*weights*Phi)*Pfull' + s*eye(nFree(d)); % should be faster this way
                            %   else
                            HH = Kall* Hess + s*eye(nFree_all); % Hessian matrix on the free basis
                            %   end
                            
                        else % any infinite covariance matrix (e.g. no prior on a weight)
                            %  if spcode
                            %      HH = Pfull*(Phi'*weights*Phi)*Pfull + s*precision{d}; % Hessian matrix on the free basis
                            %  else
                            HH = Hess + s*precision_all; % Hessian matrix on the free basis
                            %  end
                        end
                        
                        
                        Unu = (HH\B)' * Pall; % new set of weights (projected back to full basis)
                        
                        
                        obj.regressor = set_weights(M,Unu+UconstU);
                        obj = Predictor(obj); % compute rho
                        [obj,logprior] = LogPrior(obj); % compute log-prior
                        [obj,logjoint] = LogJoint(obj);
                        
                        
                        if (logjoint<old_logjoint-1e-3) % full step didn't work: go back to previous weights and run
                            FullHessianUpdate = 0;
                            obj.regressor = M;
                            obj = Predictor(obj); % compute rho
                            [obj,logprior] = LogPrior(obj); % compute log-prior
                            [obj,logjoint] = LogJoint(obj);
                            
                            
                            old_logjoint = logjoint - 2*TolFun; % just to make sure it's not flagged as 'converged'
                        else
                            M = obj.regressor;
                        end
                    end
                    
                    
                    %  update scales (to avoid slow convergence) (only where there is
                    % more than one component without constraint
                    % inforce constraint during optimization )
                    % if iter<5
                    recompute = false;
                    
                    for m=find(nD>1)
                        for r=1:rank(m)
                            alpha = zeros(1,nD(m));
                            free_weights = M(m).constraint(r,:)=='f';
                            n_freeweights = sum(free_weights);
                            if n_freeweights>1
                                this_nLP = -logprior{m}(r,free_weights);
                                mult_prior_score = prod(this_nLP )^(1/n_freeweights);
                                if mult_prior_score>0
                                    alpha(free_weights) = sqrt( mult_prior_score./ this_nLP); % optimal scaling factor
                                    
                                    for d=find(free_weights)
                                        obj.regressor(m).U{r,d} = obj.regressor(m).U{r,d}*alpha(d);
                                        logprior{m}(r,d) = logprior{m}(r,d)*alpha(d)^2;
                                    end
                                    recompute = true;
                                end
                            end
                        end
                    end
                    
                    %% compute predictor
                    % obj.regressor = M;
                    if recompute
                        obj = Predictor(obj);
                        
                        % compute log-joint
                        [obj,logjoint] = LogJoint(obj,logprior);
                        
                        M = obj.regressor;
                    end
                    
                    
                    prev_Uall = Uall;
                    prev_dU = dU;
                    Uall = concatenate_weights(M);
                    dU = Uall - prev_Uall; % weight updates
                    cos_successive_updates(iter) = dot(prev_dU,dU)/(norm(prev_dU)*norm(dU));
                    
                    rat_successive_updates(iter) = norm(dU)^2/norm(prev_dU)^2;
                    
                    dU_hist(:,iter) = dU;
                    
                    if any(~isreal(Uall))
                        error('imaginary weights...something went wrong')
                    end
                    
                    prev_dLJ = dLJ;
                    dLJ = logjoint - old_logjoint;
                    
                    rat_successive_logjoint(iter) = dLJ / prev_dLJ;
                    consistency = rat_successive_logjoint(iter) /  rat_successive_updates(iter);
                    
                    % move to full hessian update if there are signs that
                    % has fallen into convex region of parameter space
                    if D>1 && ~FullHessianUpdate && cos_successive_updates(iter)>.9 && consistency>.8 && consistency<1.2
                         FullHessianUpdate = true;
                    end
                    
                    if strcmp(obj.param.verbose, 'on')
                        fprintf(DisplayChar(iter, FullHessianUpdate));
                    elseif strcmp(obj.param.verbose, 'full')
                        fprintf('iter %d, log-joint: %f\n',iter,logjoint);
                    end
                    
                    
                    logjoint_hist(iter) = logjoint;
                    iter = iter+1;
                    tolfun_negiter = 1e-4; % how much we can tolerate log-joint to decrease due to numerical inaccuracies
                    not_converged = ~diverged && (iter<=maxiter) && (iter<miniter || (dLJ>TolFun) || (dLJ<-tolfun_negiter) ); % || (LLH<oldLLH))
                end
                
                
                logjoint_hist(iter:end) = [];
                
                %% compute variance of each order components and sort
                obj = predictor_variance(obj);
                % var_c = cell(1,nM);
                for m=1:nM
                    %                     var_c{m} = zeros(1,rank(m));
                    %                     if all(M(m).nRegressor>0)
                    %                         for r=1:rank(m)
                    %                             %    var_c{mm}(r) = var(projdim(X,U,r,spd,zeros(1,0))); % variance of overall activation for each component
                    %                             var_c{m}(r) = var(ProjectDimension(M(m),r,zeros(1,0))); % variance of overall activation for each component
                    %                         end
                    %                         if M(m).ordercomponent && rank(m)>1
                    %                             [var_c{m},order] = sort(var_c{m},'descend'); % sort variances by descending order
                    %                             M(m).U = M(m).U(order,:); % reorder
                    %                         end
                    %                     end
                    
                    
                    if all(M(m).nRegressor>0) && M(m).ordercomponent && rank(m)>1
                        [~,order] = sort(obj.score.PredictorVariance(:,m),'descend'); % sort variances by descending order
                        M(m).U = M(m).U(order,:); % reorder
                        obj.score.PredictorVariance(:,m) = obj.score.PredictorVariance(order,m);
                    end
                end
                
                
                extflg = (logjoint>old_logjoint-tolfun_negiter) && (logjoint - old_logjoint < TolFun); %)
                %exitflag = (LLH>=oldLLH-1e-10) && abs(LLH-oldLLH<param.TolFun);
                % if stepsize<1
                %    2;
                % end
                if  ~extflg
                    
                    % warning('gum:notconverged', 'Failed to converge');
                    if iter>maxiter
                        extflg = -2; % maximum number of iterations
                        msg = '\nreached maximum number of iterations (%d), log-joint:%f\n';
                        % elseif LLH<oldLLH
                        %     exitflag = -3; % LLH lower than previous step
                    elseif diverged
                        extflg = -3;
                        msg = '\n incorrect step after %d iterations, log-joint:%f\n';
                    else
                        extflg = -4;
                        msg = '\n diverged after %d iterations, log-joint:%f\n';
                    end
                else
                    msg = '\nconverged after %d iterations, log-joint:%f\n';
                end
                switch obj.param.verbose
                    case 'on'
                        fprintf(msg, iter, logjoint);
                    case 'little'
                        fprintf(DisplayChar(ip));
                end
                
                %% store weights, log-joint and exitflag for this starting point
                U_allstarting(:,ip) = {M.U};
                logjoint_allstarting(ip) = logjoint;
                logjoint_hist_allstarting{ip} = logjoint_hist;
                exitflag_allstarting(ip) = extflg;
                S.exitflag = extflg;
            end
            
            
            if strcmp(obj.param.verbose, 'little')
                fprintf('\n');
            end
            
            %% find starting point with highest log-joint
            if initialpoints>1
                [logjoint, ip] = max( logjoint_allstarting);
                obj.score.LogJoint = logjoint;
                S.exitflag = exitflag_allstarting(ip);
                for m=1:nM
                    M(m).U =  U_allstarting{m,ip};
                    M(m).U_allstarting = U_allstarting(m,:);
                end
            end
            
            S.logjoint = logjoint;
            S.logjoint_allstarting = logjoint_allstarting;
            S.logjoint_hist_allstarting = logjoint_hist_allstarting;
            S.exitflag_allstarting = exitflag_allstarting;
            
            obj.regressor = M;
            obj.score = S;
        end
        
        %% sample weights from gaussian prior
        function M = sample_weights_from_prior(M)
            for m=1:length(M)
                M(m).sample_weights_from_prior;
            end
        end
        
        
        
        %% COMPUTE PREDICTOR RHO
        function  [obj,rho] = Predictor(obj)
            rho = zeros(obj.nObs,1);
            for m=1:obj.nMod
                rho = rho + Predictor(obj.regressor(m));
            end
            obj.Predictions.rho = rho;
        end
        
        %% GET TOTAL NUMBER OF REGRESSORS
        function nReg = number_of_regressors(obj,D)
            %nReg = number_of_regressors(obj)
            % nReg = number_of_regressors(obj,D) to specify which dimensions
            % to project to on each dimension
            
            
            M = obj.regressor;
            
            if nargin<3 % by default, project on dimension 1
                D = ones(1,obj.nMod);
            end
            
            nReg = 0;
            
            for m=1:obj.nMod
                nReg = nReg + M(m).rank*M(m).nRegressor(D(m)); % add number of regressors for this module
            end
        end
        
        %% TEST IF MODEL IS A GML (dimension=1)
        function bool = isglm(obj)
            bool = all([obj.regressor.nFreeDimensions]<=1);
        end
        
        %% REMOVE DATA (to make it lighter after fitting)
        function   obj = clear_data(obj)
            
            obj.T = [];
            obj.ObservationWeight = [];
            
            for m=1:obj.nMod
                
                % remove data field from output
                obj.regressor(m) = clear_data(obj.regressor(m));
            end
        end
        
        %% SAVE MODEL TO FILE
        function save(obj, filename)
            if ~ischar(filename)
                error('filename should be a string array');
            end
            try
                save(filename, 'obj');
            catch % sometimes it fails using standard format, so need to do it using Version 7.3 format
                save(filename, 'obj', '-v7.3');
            end
        end
        
        %% COMPUTE EXPECTED VALUE
        function [obj,Y,R] = ExpectedValue(obj)
            if isempty(obj.Predictions.rho)
                [obj,rho] = Predictor(obj);
            else
                rho = obj.Predictions.rho;
            end
            
            if strcmp(obj.obs, 'binomial')
                Y = 1 ./ (1 + exp(-rho)); % predicted probability
                R = Y .* (1-Y) ; % derivative wr.t. predictor
            else % poisson
                Y = exp(rho);
                R = Y;
            end
            
            obj.Predictions.Expected = Y;
        end
        
        %% GENERATE SAMPLE FROM MODEL
        function [obj, smp] = Sample(obj)
            % retrieve expected value
            if isempty(obj.Predictions.Expected)
                [obj,Y] = ExpectedValue(obj);
            else
                Y = obj.Predictions.Expected;
            end
            
            % sample
            switch obj.obs
                case 'binomial'
                    smp = Y > rand(obj.nObs,1); % generate from Bernouilli distribution
                case {'poisson','count'}
                    smp = poissrnd(Y);
            end
            obj.Predictions.sample = smp;
            
        end
        
        %% COMPUTE LOG-LIKELIHOOD
        function  [obj,LLH] = LogLikelihood(obj)
            % compute predictor if not provided
            if isempty(obj.Predictions.rho)
                obj = Predictor(obj);
            end
            
            % compute log-likelihood
            if strcmp(obj.obs, 'binomial')
                Y = 1 ./ (1 + exp(-obj.Predictions.rho)); % predicted probability
                lh = log(Y.*obj.T + (1-Y).*(1-obj.T));
            else % poisson
                Y = exp(obj.Predictions.rho);
                lh = obj.T.*log(Y) - Y -logfact(obj.T);
            end
            if isempty(obj.ObservationWeight)
                LLH = sum(lh);
            else
                LLH = sum(obj.ObservationWeight .* lh);
            end
            
            obj.score.LogLikelihood = LLH;
        end
        
        %% computes log-likelihood for set of parameters
        %% !! should merge with function before !!
        function [LLH, gd, accuracy] = loglike(obj)
            M = obj.regressor;
            [obj,Y] = ExpectedValue(obj); % compute expected value
            [obj,LLH] = LogLikelihood(obj);
            
            if strcmp(obj.obs, 'binomial')
                %  Y = 1 ./ (1 + exp(-obj.Predictions.rho)); % predicted probability
                %  LLH = sum(obj.ObservationWeight .* log(Y.*obj.T + (1-Y).*(1-obj.T)));
                correct = obj.T==(Y>.5); % whether each observation is correct (using greedy policy)
            else % poisson
                %  Y = exp(obj.Predictions.rho);
                %  LLH = sum(obj.ObservationWeight .* (obj.T.*log(Y)-Y-logfact(obj.T)));
                correct = (obj.T==Y);
            end
            
            if isempty(obj.ObservationWeight)
                accuracy = mean(correct);
            else
                accuracy = sum(obj.ObservationWeight.*correct) / sum(obj.ObservationWeight); % proportion of correct
            end
            
            if nargout>1 % also compute gradient
                gd = [];
                err = (obj.T-Y)'; % difference between target and predictor
                
                if ~isempty(obj.ObservationWeight)
                    err = obj.ObservationWeight' .* err;
                end
                
                for m=1:obj.nMod % for each module
                    
                    for d=1:M(m).nDim % for each dimension
                        for r=1:M(m).rank
                            %    Phi = projdim(M{mm}.val,M{mm}.U,r,M{mm}.sparse,d);
                            Phi =  ProjectDimension(M(m),r,d);
                            this_gd = err*Phi; % gradient of LLH w.r.t each weight in U{r,d}
                            gd = [gd this_gd];
                        end
                    end
                end
            end
        end
        
        %% COMPUTE LOG-PRIOR
        function  [obj,LP] = LogPrior(obj)
            LP = cell(1,obj.nMod);
            % compute log-prior for each weight in each regressor
            for m=1:obj.nMod
                LP{m} = LogPrior(obj.regressor(m));
            end
            
            % sum over all
            obj.score.LogPrior = sum(cellfun(@(x) sum(x(:)), LP));
        end
        
        
        
        %% COMPUTE LOG-POSTERIOR
        function  [obj,logjoint] = LogJoint(obj, prior_score)
            
            % compute log-likelihood first
            obj = LogLikelihood(obj);
            
            if nargin>1             % log-prior provided by input
                logprior = 0;
                for m=1:length(prior_score)
                    logprior = logprior + sum(prior_score{m}(:));
                end
                obj.score.LogPrior = logprior;
            else
                obj = LogPrior(obj);
            end
            
            logjoint =  obj.score.LogLikelihood + obj.score.LogPrior;
            obj.score.LogJoint = logjoint;
            
        end
        
        
        
        %% COMPUTE HESSIAN OF LIKELIHOOD (IN UNCONSTRAINED SPACE)
        function [H,P] = Hessian(obj)
            %  nM = obj.nMod;
            M = obj.regressor;
            n = obj.nObs;
            
            % free_idx = [0 cumsum([M.nFreeParameters])]; % position for free parameters in each dimension
            full_idx = [0 cumsum([M.nParameters])]; % position for full parameter in each dimension
            % P = zeros(free_idx(end),full_idx(end)); % conversion matrix from free basis to full parametrization across all dimensions
            
            %% compute predictor, prediction values
            [obj,Y,R] = ExpectedValue(obj);
            
            
            if ~isempty(obj.ObservationWeight)
                R = R .* obj.ObservationWeight;
            end
            
            if all( cellfun(@(x) all(x>0),{M.nRegressor})>0) % unless there is an empty model
                weights = spdiags(R, 0, n, n);
            elseif ~isempty(obj.ObservationWeight)
                weights = spdiags(obj.ObservationWeight,0,n,n);
            else
                weights = speye(n);
            end
            
            
            %%  projection matrix from full to unconstrained space
            P = projection_matrix(M,'all');
            % P = [P{:}];
            % P = blkdiag(P{:});
            
            
            %% full design matrix
            [Phi,nReg] = design_matrix(M,[], 0, 0);
            % spcode = any(cellfun(@issparse, {M.val})); % use sparse matrices
            
            % nRegtot = full_idx(end);
            
            
            
            %             if spcode
            %                 Phi = sparse(n,nRegtot);
            %             else
            %                 Phi = zeros(n,nRegtot); % 2-D design matrix with all weighted regressors
            %             end
            %             midx = 0;
            %             for m=1:nM
            %                 rr = M(m).rank;
            %                 for d=1:M(m).nDim
            %
            %                     for r=1:rr
            %                         gdx = free_idx( midx+(d-1)*rr+r ) + 1 : free_idx( midx+(d-1)*rr+r+1 ); % index for free basis
            %                         fdx = full_idx( midx+(d-1)*rr+r ) + 1 : full_idx( midx+(d-1)*rr+r+1 ); % index for full parameter set
            %                         % BigPhi(:,fdx) = projdim(X,U,r,spd,d);
            %                         Phi(:,fdx) = ProjectDimension(M(m),r,d);
            %                         %conversion matrix from free to full
            %                         %   P(gdx, fdx) = PP{m}{r,d};
            %                     end
            %                 end
            %                 midx = midx + M(m).nDim * rr; % jump index by number of components in module
            %             end
            
            %% Hessian for full parameter set
            H_full = Phi'*weights*Phi;
            
            %% correct for non-diagonal blocks (i.e. different set of weights) for same component
            
            % negative prediction error
            nPE = (Y-obj.T)';
            
            midx = 0;
            for m=1:obj.nMod
                rr = M(m).rank;
                for d=1:M(m).nDim
                    
                    for f=d+1:M(m).nDim % non-diagonal blocks (i.e. different set of weights) for same component
                        
                        for r=1:rr
                            
                            fdx = full_idx( midx+(d-1)*rr+r ) + 1 : full_idx( midx+(d-1)*rr+r+1 ); % index for full parameter set
                            fdx2 = full_idx( midx+(f-1)*rr+r ) + 1 : full_idx( midx+(f-1)*rr+r+1 ); % index for full parameter set
                            
                           % H_across = Phi(:,fdx)'*weights*Phi(:,fdx2);
                            Ha2 = ProjectDimension(M(m),r,[d f],1,nPE); % add collapsing over observable dimension (Y-T),
                           % H_across = H_across + Ha2;
                            
                          %  H_full(fdx,fdx2 ) = H_across;
                             H_full(fdx,fdx2 ) = H_full(fdx,fdx2 ) + Ha2;
                            H_full(fdx2,fdx ) =  H_full(fdx,fdx2)';
                        end
                    end
                end
                midx = midx + M(m).nDim * rr; % jump index by number of components in module
            end
            

            
            
            %% project from full parameter space to free basis
            H = P*H_full*P';
            
            H = (H+H')/2; % ensure that it's symmetric (may lose symmetry due to numerical problems)
            
            
        end
        
        %% COMPUTE POSTERIOR COVARIANCE
        function [obj, V, B]= PosteriorCov(obj)
            
            % compute Hessian of likelihood
            [H,P] = Hessian(obj);
            
            M = obj.regressor;
            
            %precision = cellfun(@(x) x.precision, M, 'uniformoutput',0); % precision matrix from all modules
            %precision = [precision{:}];
            K = {M.sigma};
            K = cellfun(@(x) x(:)', K, 'unif',0);
            K = [K{:}]; % group prior covariance from all modules
            K = blkdiag(K{:});
            Kfree = P*K*P'; % prior covariance in free basis
            
            %sqW = sqrtm(W);
            %B = eye(free_idx(end)) + sqW*Kfree*sqW; % formula to computed marginalized evidence (eq 3.32 from Rasmussen & Williams 2006)
            nFreeParameters = sum(cellfun(@(x) sum(x(:)), {M.nFreeParameters}));%nFreeParameters = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));
            %   nFreeParameters = sum([M.nFreeParameters]);
            B = Kfree*H + eye(nFreeParameters); % formula to computed marginalized evidence
            
            [~,isNPD] = chol(B);
            if isNPD % if some eigenvalue is negative due to numerical issues
                c = 1e3*eps; % force minimum eigenvalue
                XX = B - c*eye(nFreeParameters);
                try
                [~, HH] = poldec(full(XX)); % Polar Decomposition from Matrix Computation Toolbox (http://www.maths.manchester.ac.uk/%7Ehigham/mctoolbox/)
                catch ME
                    if strcmp(ME.identifier,'MATLAB:UndefinedFunction')
                        error('The Matrix Computation Toolbox was not found on the path, download from http://www.maths.manchester.ac.uk/%7Ehigham/mctoolbox/ and add to path');
                    else
                        rethrow(ME)
                    end
                end
                B = (XX+HH)/2 + c*eye(nFreeParameters);  % (equation 3 from Huang, Farewell & Pan, 2017, with c=eps)
            end
            
            
            if any(isinf(Kfree(:))) % no prior on some variable
                wrn = warning('off', 'MATLAB:singularMatrix');
                V = inv(H + inv(Kfree));
                warning(wrn.state, 'MATLAB:singularMatrix');
            else
                V = B\Kfree; %inv(W + inv(Kfree));
            end
            
            % check that covariance is symmetric
            if norm(V-V')>1e-3*norm(V)
                if any(cellfun(@(x) any(sum(x=='f',2)>1), {M.constraint}))
                    warning('posterior covariance is not symmetric - this is likely due to having two free dimensions in our regressor - try adding constraints');
                else
                    warning('posterior covariance is not symmetric - dont know why');
                end
            end
            
            V = (V+V')/2; % may be not symmetric due to numerical reasons
            
        end
        
        %% CONCATENATE ALL WEIGHTS
        function U = concatenate_weights(obj)
            U = concatenate_weights(obj.regressor);
            % U = [obj.regressor.U];
            % U = [U{:}];
        end
        
        %% CONCATENATE WEIGHTS OVER POPULATION
        function obj = concatenate_over_models(obj, place_first)
            if nargin==1 % whether we put model index as first dimension in weights
                place_first = false;
            end
            
            nObj = length(obj);
            
            %% concatenate weights
            n_Mod = [obj.nMod];
            if ~all(n_Mod == n_Mod(1))
                error('all models should have the same number of modules');
            end
            for m=1:n_Mod(1)
                nD = cellfun(@(x) x(m).nDim, {obj.regressor}); %[obj.regressor(m).nDim];
                if ~all(nD==nD(1))
                    error('dimensionality of module %d differs between modules');
                end
                rank = cellfun(@(x) x(m).rank, {obj.regressor});
                if ~all(rank==rank(1))
                    error('rank of module %d differs between models', m);
                end
                %                 nReg = cellfun(@(x) x(m).nRegressor, {obj.regressor},'unif',0);
                %                 nReg = cat(1,nReg{:});
                %                 if ~all(nReg==nReg(1,:),'all')
                %                     error('number of regressors in module %d differs between models', m);
                %                 end
                
                U = cellfun(@(x) x(m).U, {obj.regressor},'unif',0);
                U = cat(1,U{:});
                se = cellfun(@(x) x(m).se, {obj.regressor},'unif',0);
                se = cat(1,se{:});
                TT = cellfun(@(x) x(m).T, {obj.regressor},'unif',0);
                TT = cat(1,TT{:});
                p = cellfun(@(x) x(m).p, {obj.regressor},'unif',0);
                p = cat(1,p{:});
                scale = cellfun(@(x) x(m).scale, {obj.regressor},'unif',0);
                scale = cat(1,scale{:});
                
                % in case scale is not specified, use default values
                NoScale = cellfun(@isempty, scale);
                DefaultScale = cellfun(@(x) 1:length(x), U, 'unif',0);
                scale(NoScale) = DefaultScale(NoScale);
                
                %U = cat(1,obj.regressor(m).U); % all weights for this module per model (% model x dimension cell array)
                % se = cat(1,obj.regressor(m).se); % all weights for this module per model
                
                % check that scale is the same, and if not add nans where
                % appropriate
                for d=1:nD(1)
                    if ~all(cellfun(@(x) isequal(x,scale{1,d}) ,scale(2:end,d)))
                        sc = unique([scale{:,d}]); % all values across all models
                        for i=1:nObj
                            ss = scale{i,d};
                            
                            % find the indices in sc corresponding to values
                            % for this model
                            idx = zeros(1,length(ss));
                            for v=1:length(ss)
                                
                                idx(v) = find(ss(v)==sc);
                            end
                            
                            % replace
                            tmp = nan(size(U{i,d},1),length(sc));
                            tmp(:,idx) = U{i,d};
                            U{i,d} = tmp;
                            
                            tmp = nan(size(se{i,d},1),length(sc));
                            tmp(:,idx) = se{i,d};
                            se{i,d} = tmp;
                            
                            tmp = nan(size(TT{i,d},1),length(sc));
                            tmp(:,idx) = TT{i,d};
                            TT{i,d} = tmp;
                            
                            tmp = nan(size(p{i,d},1),length(sc));
                            tmp(:,idx) = p{i,d};
                            p{i,d} = tmp;
                            
                        end
                        obj(1).regressor(m).scale{d} = sc;
                    end
                end
                
                if place_first && rank(1)>1
                    %move rank from dim 1 to dim 3
                    U = cellfun(@(x) permute(x,[3 2 1]), U, 'unif',0);
                    se = cellfun(@(x) permute(x,[3 2 1]), se, 'unif',0);
                    TT = cellfun(@(x) permute(x,[3 2 1]), TT, 'unif',0);
                    p = cellfun(@(x) permute(x,[3 2 1]), p, 'unif',0);
                    
                    
                end
                
                % over which dimension we concatenate
                if place_first
                    dd =1; % put model along first dimension (one row per model)
                else
                    dd = 3; % rank x weight x model
                    % elseif rank(1)>1
                    %     dd = 3; % weight x rank x model
                    % else
                    %     dd = 2; % weight x model
                end
                
                U_all = cell(1,nD(1));
                se_all = cell(1,nD(1));
                T_all = cell(1,nD(1));
                p_all = cell(1,nD(1));
                for d=1:nD(1)
                    U_all{d} = cat(dd, U{:,d});
                    se_all{d} = cat(dd, se{:,d});
                    T_all{d} = cat(dd, TT{:,d});
                    p_all{d} = cat(dd, p{:,d});
                end
                
                obj(1).regressor(m).U = U_all;
                obj(1).regressor(m).se = se_all;
                obj(1).regressor(m).T = T_all;
                obj(1).regressor(m).p = p_all;
                
                % concatenate hyperparameters
                HP = cellfun(@(x) x(m).HP, {obj.regressor},'unif',0);
                HP = cat(1,HP{:});
                for d=1:size(HP,2)
                    
                    if  ~isempty( HP(1,d).HP)
                        obj(1).regressor(m).HP(d).HP = cat(1,HP(:,d).HP);
                    end
                end
            end
            
            %% concatenate scores
            Sc = concatenate_score(obj);
            obj(1).score = Sc;
            
            %% concatenate predictions
            obj(1).Predictions = [obj.Predictions];
            
            %% keep only first model
            obj(2:end) = [];
            
            
            
            
            
        end
        
        %% CONCATENATE SCORES OVER MODELS(FOR MODEL SELECTION/COMPARISON)
        function Sc = concatenate_score(obj)
            n =  numel(obj);
            
            all_score = {obj.score};
            
            with_score = ~cellfun(@isempty, all_score); % models with scores
            
            all_fieldnames = cellfun(@fieldnames, all_score(with_score),'unif',0);
            all_fieldnames = unique(cat(1,all_fieldnames{:})); % all field names
            
            % all possible metrics
            metrics = {'LogPrior','Loglikelihood','LogJoint','AIC','AICc','BIC','LogEvidence', 'accuracy','testscore','FittingTime'};
            
            % select metrics that are present in at least one model
            metrics = metrics(ismember(metrics, all_fieldnames));
            
            Sc = struct;
            for i=1:length(metrics)
                mt = metrics{i};
                
                % pre-allocate values for each model
                X = nan(size(obj));
                
                % add values from each model where value is present
                for m=1:n
                    if ~isempty(all_score{m}) && isfield(all_score{m}, mt)
                        X(m) =  all_score{m}.(mt);
                    end
                end
                
                Sc.(mt) = X;
                
                
            end
            
        end
        
        %% AVERAGE WEIGHTS OVER POPULATION
        function obj = population_average(obj)
            n = length(obj); % size of population (one model per member)
            
            % first concatenate
            obj = concatenate_over_models(obj);
            
            % now compute average and standard deviation over
            % population
            for m=1:obj.nMod
                
                
                if obj.regressor(m).rank(1)>1
                    dd = 3; % weight x rank x model
                else
                    dd = 2; % weight x model
                end
                
                for d=1:obj.regressor(m).nDim
                    X = obj.regressor(m).U{d};
                    
                    obj.regressor(m).U{d} = mean(X,d); % population average
                    obj.regressor(m).se{d} = std(X,[],d)/sqrt(n); % standard error of the mean
                end
                
            end
            
        end
        
        %% COMPUTE EXPLAINED PREDICTOR VARIANCE FROM EACH MODULE
        function [obj, PV] = predictor_variance(obj)
            
            if ~isscalar(obj)
                PV = cell(size(obj));
                for m=1:numel(obj)
                    [obj(m),PV{m}] = predictor_variance(obj(m));
                end
                return;
            end
            
            rank = [obj.regressor.rank];
            PV = zeros(max(rank),obj.nMod);
            
            for m=1:obj.nMod
                for r=1:obj.regressor(m).rank
                    PV(r,m) = var(Predictor(obj.regressor(m),r));
                end
            end
            
            obj.score.PredictorVariance = PV;
        end
        
        %% COMPUTE ESTIMATED VARIANCE OF PREDICTOR
        function obj = compute_rho_variance(obj)
            
            n = obj.nObs;
            M = obj.regressor;
            sigma = obj.score.covb;
            
            if any(isinf(sigma(:))) % if no prior defined on any dimension, cannot compute it
                obj.Predictions.rhoVar = nan(n,1);
                return;
            end
            
            nSample = 1000; % number of sample to compute variance
            
            full_idx = [0 cumsum([M.npar])]; % position for full parameter in each dimension
            
            % concatenate weights mean
            U = concatenate_weights(obj);
            % U = [M.U];
            % U = [U{:}];
            
            rho_sample = zeros(n,nSample);
            
            for i=1:nSample
                % draw sample for weights
                
                try
                    Us = mvnrnd(U,sigma);
                catch
                    [~,isnodefpos] = chol(sigma);
                    if isnodefpos
                        warning('posterior covariance is not definite positive');
                        obj.Predictions.rhoVar = nan(n,1);
                        return;
                    else
                        error('unknown error');
                    end
                end
                
                %place it in model
                midx = 0;
                for m=1:obj.nMod
                    rr = M(m).rank;
                    for d=1:obj.regressor(m).nDim
                        for r=1:rr
                            fdx = full_idx( midx+(d-1)*rr+r ) + 1 : full_idx( midx+(d-1)*rr+r+1 ); % index for full parameter set
                            obj.regressor(m).U{r,d} = Us(fdx);
                        end
                    end
                    midx = midx + obj.regressor(m).nDim * rr; % jump index by number of components in module
                end
                
                % compute predictor and store it
                obj = Predictor(obj);
                rho_sample(:,i) = obj.Predictions.rho;
            end
            
            % compute variance over samples
            obj.Predictions.rhoVar = var(rho_sample,0,2);
        end
        
        
        %% COMPUTE VIF (VARIANCE INFLATION FACTOR)
        function [V, V_free] = vif(obj,D)
            if nargin<2 % by default, project on dimension 1
                D = ones(1,obj.nMod);
            end
            
            Phi = design_matrix(obj.regressor,[],D);
            
            PP = projection_matrix(obj.regressor); % free-to-full matrix conversion
            P = catadditional(PP, D(:)); % projection matrix
            P = P{1};
            
            Phi = Phi*P';
            
            R = corrcoef(Phi); % correlation matrix
            V_free = diag(inv(R))'; % VIF in free basis
            
            % project back to full basis and normalize
            V = (V_free*P) ./ (V_free*ones(size(P)));
        end
        
        %% PLOT DESIGN MATRIX
        function h = plot_design_matrix(obj, varargin)
            
            h.Axes = [];
            h.Objects = {};
            
            [Phi, nReg] = design_matrix(obj.regressor, varargin{:});
            
            M = obj.regressor;
            
            %             if nargin>1 % only plot for subset of trials
            %                 obj = extract_observations(obj,subset);
            %             end
            %
            %             M = obj.regressor;
            %
            %             if nargin<3 % by default, project on dimension 1
            %                 D = ones(1,obj.nMod);
            %             end
            %
            %
            %             %  rank = ones(1,obj.nMod); % all rank one
            %             nReg = zeros(1,obj.nMod);
            %
            %             for m=1:obj.nMod
            %                 nReg(m) = M(m).rank*M(m).nRegressor(D(m)); % add number of regressors for this module
            %
            %                 if isempty(M(m).U)
            %                     for d=1:M(m).nDim
            %                         M(m).U{d} = zeros(M(m).rank,M(m).nRegressor(d));
            %                     end
            %                 end
            %
            %                 %M(m).val = tocell(M(m).val);
            %
            %                 % initialize weight to default value
            %                 M(m) = M(m).initialize_weights();
            %
            %             end
            %
            %             n = obj.nObs;
            %
            %             ii = 0; % index for regressors in design matrix
            %             Phi = zeros(n,sum(nReg));
            %             for m=1:obj.nMod
            %
            %                 % project on all dimensions except the dimension to optimize
            %
            %                 for r=1:rank(m)
            %                     idx = ii + (1:M(m).nRegressor(D(m))); % index of regressors
            %                     Phi(:,idx) = ProjectDimension(M(m),r,D(m)); % tensor product, and squeeze into observation x covariate matrix
            %                     ii = idx(end); %
            %                 end
            %             end
            
            hold on;
            colormap(gray);
            h_im = imagesc(Phi);
            h.Objects = [h.Objects h_im];
            axis tight;
            
            nRegCum = cumsum([0 nReg]);
            for m=1:obj.nMod-1
                plot((nRegCum(m+1)+.5)*[1 1], ylim, 'b','linewidth',2);
            end
            
            set(gca, 'ydir','reverse');
            if ~all(cellfun(@isempty,{M.label}))
                for m=1:obj.nMod
                    h_txt(m) = text( mean(nRegCum(m:m+1))+1, 0.5, M(m).label,'verticalalignment','bottom','horizontalalignment','center');
                    if nReg(m)<.2*sum(nReg)
                        set(h_txt(m),'Rotation',90,'horizontalalignment','left');
                    end
                end
                h.Objects = [h.Objects h_txt];
                
            end
            
            axis off;
            
        end
        
        %% PLOT POSTERIOR COVARIANCE
        function h = plot_posterior_covariance(obj)
            
            
            h.Axes = [];
            h.Objects = {};
            
            
            M = obj.regressor;
            covb = obj.score.covb; % posterior covariance matrix
            
            
            hold on;
            colormap(flipud(gray));
            h_im = imagesc(covb);
            h.Objects = [h.Objects h_im];
            axis tight;
            
            nReg = zeros(1,obj.nMod);
            
            for m=1:obj.nMod
                nReg(m) = M(m).nParameters; % add number of regressors for this module
            end
            
            nRegCum = cumsum([0 nReg]);
            for m=1:obj.nMod-1
                plot((nRegCum(m+1)+.5)*[1 1], ylim, 'b','linewidth',2);
                plot(xlim,(nRegCum(m+1)+.5)*[1 1], 'b','linewidth',2);
            end
            
            set(gca, 'ydir','reverse');
            if ~all(cellfun(@isempty,{M.label}))
                for m=1:obj.nMod
                    h_txt(m,1) = text( mean(nRegCum(m:m+1))+1, 0.5, M(m).label,'verticalalignment','bottom','horizontalalignment','center');
                    h_txt(m,2) = text(0.5, mean(nRegCum(m:m+1))+1,  M(m).label,'verticalalignment','middle','horizontalalignment','right');
                    
                    if nReg(m)<.2*sum(nReg)
                        set(h_txt(m,1),'Rotation',90,'horizontalalignment','left');
                    else
                        set(h_txt(m,2),'Rotation',90);
                    end
                end
                h.Objects = [h.Objects h_txt(:)'];
                
            end
            
            axis off;
            
        end
        
        
        
        %% PLOT FITTED WEIGHTS
        function h = plot_weights(obj, U2)
            % M.plot_weights()
            % M.plot_weights(1:3)
            % M.plot_weights('regressor1')
            % M.plot_weights({'regressor1','regressor2')
            
            if nargin<2
                U2 = 1:obj.nMod;
            end
            M = SelectRegressors(obj.regressor, U2);
            
            cols = defcolor;
            colmaps = {'jet','hsv','winter','automn'};
            
            i = 1; % subplot counter
            c = -1; % color counter
            cm = 1; % colormap counter
            h.Axes = [];
            h.Objects = {};
            
            NoFixedWeights = cell(1,obj.nMod);
            for m=1:numel(M)%obj.nMod
                NoFixedWeights{m} = any(M(m).constraint~='n',1); % regressors with constant
            end
            nNoFixedWeights  = cellfun(@sum,NoFixedWeights); % number of non-fixedd dimensions in each module
            nNoFixedWeightsTot = sum(nNoFixedWeights);
            
            
            for m=1:numel(M)%obj.nMod
                if isempty(M(m).label)
                    M(m).label = cell(1,M(m).nDim );
                else
                    M(m).label = tocell(M(m).label);
                end
                
                M(m).scale = tocell(M(m).scale);
                
                for d=find(NoFixedWeights{m})
                    h.Axes(end+1) = subplot2(nNoFixedWeightsTot,i);
                    
                    if length(M(m).label)<d || isempty(M(m).label{d})
                        M(m).label{d} = sprintf('U%d_%d',m,d); % default label
                    end
                    title(M(m).label{d});
                    
                    if isempty(M(m).scale{d})
                        M(m).scale{d} = 1:M(m).nRegressor(d);
                    end
                    
                    U = cat(1,M(m).U{:,d})'; % concatenate over ranks
                    if isempty(M(m).se)
                        se = nan(size(U));
                    else
                        se = cat(1,M(m).se{:,d})';
                    end
                    scale = M(m).scale{d}';
                    nU = size(U,1);
                    nScale = length(scale);
                    if size(U,2)>1
                        nCurve = size(U,2);
                    else
                        nCurve = nU/nScale;
                        if nCurve>1
                            U = reshape(U, nScale,nCurve);
                            se = reshape(se, nScale,nCurve);
                        end
                    end
                    
                    %% plotting options
                    plot_opt = {};
                    
                    % fopts = fieldnames(M(m).plot);
                    % for ff=1:length(fopts)
                    %    fopts{end+1:end+2} = {fops{f}, M(m).plot.(fopts{f})};
                    % end
                    %  else
                    
                    if M(m).nRegressor(d) < 8
                        plot_opt = {'bar'};
                    end
                    
                    % default color
                    plot_opt{end+1} = 'Color';
                    
                    if nCurve<5
                        plot_opt{end+1} = cols(mod(c+(1:nCurve),length(cols))+1);
                        %   end
                        if M(m).rank ==1
                            c = c + nCurve;
                        end
                    else
                        plot_opt{end+1} = colmaps{cm};
                        %   end
                        cm = cm + 1;
                        
                    end
                    
                    if ~isempty(M(m).plot) && ~isempty(M(m).plot{d})
                        plot_opt = [plot_opt M(m).plot{d}];
                    end
                    
                    %% plot
                    if iscell(scale)
                        [~,~,h_nu] = wu(U,se,{scale},plot_opt{:});
                        twodplot = 0;
                    elseif isvector(scale)
                        
                        [~,~,h_nu] = wu(scale, U,se,plot_opt{:});
                        twodplot = 0;
                    else
                        %2d plot for scale
                        twodplot = 1;
                        x_unq = unique(scale(:,1));
                        y_unq = unique(scale(:,2));
                        nX = length(x_unq);
                        nY = length(y_unq);
                        U2 = nan(nX,nY);
                        for iX=1:nX
                            for iY=1:nY
                                bool = (scale(:,1) == x_unq(iX)) & (scale(:,2)==y_unq(iY));
                                if any(bool)
                                    U2(iX,iY) = U(find(bool,1));
                                end
                            end
                        end
                        h_nu = imagesc(y_unq,x_unq, U2);
                        
                    end
                    if ~twodplot && M(m).nRegressor(d) >= 8
                        hold on;
                        if M(m).constraint(d) == '1'
                            plot(xlim,[1 1], 'Color',.7*[1 1 1]);
                        elseif M(m).constraint(d) == 'm'
                            plot(xlim,[1 1]/M(m).nRegressor(d), 'Color',.7*[1 1 1]);
                        else
                            plot(xlim,[0 0], 'Color',.7*[1 1 1]);
                        end
                    end
                    
                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end
        end
        
        %% PLOT HYPERPARAMETERS
        function h = plot_hyperparameters(obj)
            
            M = obj.regressor;
            
            i = 1; % subplot counter
            h.Axes = [];
            h.Objects = {};
            
            % which dimensions have hyperparameters
            with_HP = cell(1,obj.nMod);
            ndim = zeros(1,obj.nMod);
            for m=1:obj.nMod
                with_HP{m} = find(~cellfun(@isempty, {M(m).HP.HP}));
                ndim(m) = length(with_HP{m});
            end
            nDim_tot = sum(ndim);
            
            %% plot hyperparmeters
            for m=find(ndim)
                if isempty(M(m).label)
                    M(m).label = cell(1,M(m).nDim );
                else
                    M(m).label = tocell(M(m).label);
                end
                % if isempty(M(m).HP.label)
                %     M(m).HP_labels = cell(1,M(m).nDim );
                % else
                % M(m).HP_label = {M(m).HP.label};
                %if ~isempty(M(m).HP_label) && ~iscell(M(m).HP_label{1})
                %    M(m).HP_label = {M(m).HP_label};
                %end
                % end
                for d=ndim(m)
                    h.Axes(end+1) = subplot2(nDim_tot,i);
                    if isempty(M(m).label{d})
                        M(m).label{d} = sprintf('U%d_%d',m,d); % default label
                    end
                    title(M(m).label{d});
                    
                    [~,~,h_nu] = wu(M(m).HP(d).HP',[],{M(m).HP(d).label},'bar');
                    
                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end
            
        end
        
        %% PLOT DATA GROUPED BY VALUES OF PREDICTOR
        function h = plot_data_vs_predictor(obj, Q)
            [~,rho] = Predictor(obj);
            
            if nargin<2 % number of quantiles: by default scales with square root of number of observations
                Q = ceil(sqrt(obj.nObs)/2);
            end
            
            H = quantile(rho,Q);
            
            H = [-inf H inf];
            
            if isempty(obj.ObservationWeight)
                w = ones(obj.nObs,1);
            else
                w = obj.ObservationWeight;
            end
            
            for q=1:Q+1
                idx = rho>H(q) & rho<=H(q+1);
                nobs = sum(w(idx));
                M(q) = sum(obj.T(idx) .* w(idx)) /  nobs;
                if strcmp(obj.obs,'binomial')
                    sem(q) = sqrt( M(q)*(1-M(q))/nobs);
                else
                    sem(q) = std( obj.T(idx), w(idx)) / sqrt(nobs);
                end
                
            end
            
            H = [H(2:end-1) max(rho)];
            
            
            h =  wu(H, M, sem);
            axis tight; hold on;
            
            xx = linspace(min(xlim), max(xlim), 200);
            switch obj.obs
                case {'binary','binomial'}
                    yy = 1./(1+exp(-xx));
                    plot(xlim,.5*[1 1], 'color',.7*[1 1 1]);
                    plot([0 0],ylim, 'color',.7*[1 1 1]);
                case {'count','poisson'}
                    yy = exp(xx);
                    set(gca, 'yscale','log');
            end
            plot(xx,yy,'b');
            xlabel('\rho');
            ylabel('dependent variable');
            
            
        end
        
    end
    %%%% END OF METHODS
    
end

%% FUNCTIONS

%% order of IRLS updates in estimation
function UpOrder = UpdateOrder(M)
nM = length(M);
D = max([M.nFreeDimensions]);
UpOrder = zeros(nM, D);

for m=1:nM
    if all(M(m).constraint(:)=='n') %all(M(m).constraint=='n','all')
        UpOrder(m,:) = ones(1,D);
    else
        fir = first_update_dimension(M(m));
        free_dims = find(any(M(m).constraint ~= 'n',1));
        fir = find(free_dims== fir);
        UpOrder(m,:) = 1+mod(fir-1+(0:D-1),length(free_dims)); % update dimension 'fir', then 'fir'+1, .. and loop until D
        UpOrder(m,:) = free_dims(UpOrder(m,:));
    end
end
end

%% for projection or covariance matrix, group matrices from all modules that are used in each update
function NN = catadditional(PP, update_o)
rank = cellfun(@(x) size(x,1),PP); % rank for each module
nMod = length(PP);
%siz = cellfun(@length,PP); % number of dimension for each module
NN = cell(1,size(update_o,2) ); %cell(1,max(siz));

for d=1:size(update_o,2) %siz
    TT = cell(1,sum(rank));
    id = update_o(:,d)'; %min(d,siz); % for which dimension we pick from in each module
    cc = 0;
    for m=1:nMod
        TT(cc + (1:rank(m))) = PP{m}(:,id(m))';
        cc = cc + rank(m);
    end
    %  TT = [TT{:}]; % group over dimensions
    
    %% if some projection matrix is sparse and some other is not, resolve...
    sp = cellfun(@issparse, TT);
    if any(sp) && any(~sp)
        nnz_all = sum(cellfun(@nnz, TT));
        numel_all = sum(cellfun(@numel, TT));
        if nnz_all/numel_all > .2 % if overall sparsity is more than 0.2, convert to full (otherwise will be converted to sparse by default)
            TT = cellfun(@full, TT, 'unif',0);
        end
    end
    
    NN{d} = blkdiag(TT{:}); %
end
end

% %% convert vectors for all weights to cell array for each dimension
% function C = num2dimcell(V,siz,rank)
% full_idx = [0 cumsum(repelem(siz,rank)) ]; % position for full parameter in each dimension
% C = cell(rank,length(siz));
% for i=1:rank*(length(siz))
%     C{i} = V(full_idx(i)+1:full_idx(i+1));
% end
% end

% %% extract design matrix for subset of observations
% function X = extract_observations(X,subset) %,dim)
%
% S = size(X);
% X = reshape(X, S(1), prod(S(2:end))); % reshape as matrix to make subindex easier to call
% X = X(subset,:);
% X = reshape(X, [size(X,1), S(2:end)]); % back to nd array
%
% %if nargin<3
% %    dim=1; % by default dimension 1 (observations)
% %end
% %D = length(m)-1;
% %for i=1:length(X)
% %    if i==1 || (~isempty(X{i}) && size(X{i},dim)>1)
% %        Xidx = cell(1,ndims(X{i})); % index within X for observations subset
% %        Xidx{dim} = subset;
% %        for d=setdiff(1:ndims(X{i}),dim) % for non sparse matrices
% %            Xidx{d} = 1:size(X{i},d); % take all data along that direction
% %        end
% %        X{i} = X{i}(Xidx{:});
% %    end
% %end
% end

%% select regressors by index or label  (e.g. for plotting)
function M = SelectRegressors(M, idx)
if isstring(idx)
    idx = {idx};
end

% if regressor labels are provided, convert to labels
jdx = cell(1,length(idx));
if iscell(idx)
    for m=1:length(idx)
        ii = find(strcmp(idx{m},{M.label}));
        if isempty(ii)
            error('label ''%s'' does not match any regressor label', idx{m});
        end
        jdx{m} = ii;
    end
    idx = jdx;
end

% select set of regressors;
M = M(idx);
end

%% projection matrix from free set of parameters to complete set of
% parameters
function PP = projection_matrix(M, do_all)
nMod = length(M); % number of modules
PP = cell(1,nMod);

for m=1:nMod
    PP{m} = ProjectionMatrix(M(m));
end

if nargin>1 && isequal(do_all,'all')
    PP = cellfun(@(x) x(:)', PP,'unif',0);
    PP = [PP{:}]; % concatenate over modules
    PP = blkdiag(PP{:}); % global transformation matrix from full parameter set to free basis
end

end

%% check size of regressors and resize if multiple dimension observations
function M = checkregressorsize(M,n)
if M.nObs ~=n
    error('number of observations do not match between observable and regressors');
end
% obsDim = length(n);
% ismat = obsDim>1; % multiple dimensino observations
% for i=1:length(M.val)
%     if ~isempty(M.val{i})
%         siz = size(M.val{i});
%         if ~all( (siz(1:obsDim)==1) | (siz(1:obsDim)==n))
%             error('number of regressor lines do not equate number of observations');
%         end
%
%         % singleton observations dimension, should be expanded over these
%         singledim = (siz(1:obsDim)==1);
%         if any(singledim)
%             repvec = ones(size(siz));
%             repvec(singledim) = n(singledim);
%             M.val{i} = repmat(M.val{i}, repvec);
%         end
%
%         % group observation into dimension one
%         if ismat
%             error('not coded yet');
%             %M.val{i} = reshape(M.val{i}, [prod(n) siz(obsDim+1:end)]);
%         end
%     end
% end

end



%% assign hyperparameter values to regressors
function M = assign_hyperparameter(M, P, idx)
for m=1:length(M) % for each regressor
    
    for d=1:M(m).nDim % for each dimension
        for r=1:size(M(m).HP,1) %M(m).rank
            cc = idx{m}{r,d};  %index for corresponding module
            % hyperparameter values for this component
            HP_fittable = M(m).HP(r,d).fit;
            M(m).HP(r,d).HP(HP_fittable) = P(cc); % fittable values
        end
    end
    
end
end


% %% evaluate covariances for given values of hyperparameters
% function [M,GHP] = covfun_eval(M)
% %function [M,GHP,spectral,spectc] = covfun_eval(M,P,UU,n_hyper, idx, rank)
%
% nMod = length(M);
% grad_sf  = cell(1,nMod);
% with_grad = (nargout>3); % compute gradient
% for m=1:nMod
%     if with_grad
%         [M(m),grad_sf{m}] = compute_prior_covariance(M(m));
%     else
%         M(m) = compute_prior_covariance(M(m));
%     end
% end
%
% if with_grad
%     grad_sf = [grad_sf{:}]; % concatenate over modules
%     GHP = blkdiagn(grad_sf{:}); % overall gradient of covariance matrices w.r.t hyperparameters
% end
% end

% %% CONVERT WEIGHTS BACK FROM SPECTRAL TO ORIGINAL SPACE
% function M = weightprojection(M,covb, nMod,spectral,spectc)
%
% midx = 0;
% for m=1:nMod
%     ss = M(m).nRegressor; % size of each dimension
%     if spectral(m) % regression in spectral domain
%         M(m).U_Fourier =  M(m).U; % save weight in fourier domain
%         M(m).Fourier_se =  M(m).se;
%
%         for f= 1:length(spectc{m}) % for each component to be converted
%             d = spectc{m}(f);
%
%             for r=1:M(m).rank
%                 M(m).U{r,d} = M(m).U{r,d} * M(m).Bfft{d}; % compute coefficient back in original domain
%                 reg_idx = (1:ss(d)) + (r-1)*ss(d) + M(m).rank*sum(ss(1:d-1)) + midx; % index of regressors in design matrix
%                 this_covb = M(m).Bfft{d}' * covb(reg_idx,reg_idx) * M(m).Bfft{d}; % posterior covariance in original domain
%                 M(m).se{r,d} = sqrt(diag(this_covb))'; % standard error in original domain
%             end
%         end
%     end
%     midx = midx + sum(M(m).nRegressor) * M(m).rank; % jump index by number of components in module
% end
% end

%% negative marginalized evidence (for hyperparameter fitting)
function [negME, obj] = gum_neg_marg(obj, HP, idx)

persistent UU fval nfval;
%% first call with no input: clear persistent value for best-fitting parameters
if nargin==0
    fval = [];
    UU = [];
    nfval = 0;
    return;
end
if isempty(fval)
    fval = Inf;
end

M = obj.regressor;
nMod = obj.nMod;
if ~isempty(UU)
    for m=1:nMod
        M(m).U = UU{m};
    end
end

param = obj.param;
%param.crossvalidation = [];
param.spectralback = 0;
if strcmp(param.verbose, 'full')
    param.verbose = 'on';
elseif strcmp(param.verbose, 'on')
    param.verbose = 'little';
else
    param.verbose = 'off';
end

if nfval>0
    param.initialpoints = 1; % just one initial point after first eval
end

% assign hyperparameter values to regressors
M = assign_hyperparameter(M, HP, idx);

% evaluate covariances at for given hyperparameters
M = compute_prior_covariance(M);
%[M, spectral, spectc] = covfun_eval(M,HP);

% check that there is no nan value in covariance matrix
all_sigma = [M.sigma];
for c=1:length(all_sigma)
    if any(isnan(all_sigma{c}(:)))
        negME = nan;
        return;
    end
end

obj.regressor = M;

% estimate weights from GUM

obj = obj.infer(param);
%[M, S] = gum(M,T,param);

% for spectral decomposition, convert weights back from spectral to
% original domain
%M = weightprojection(M,S.covb,nMod,spectral,spectc,rank);
obj.regressor = project_from_spectral(obj.regressor);

% negative marginal evidence
negME = -obj.score.LogEvidence;

% if best parameter so far, update the value for initial parameters
if negME < fval
    fval = negME;
    UU = cell(1,nMod); % group all weights into a cell array
    for m=1:nMod
        UU{m} = M(m).U;
    end
end

nfval = nfval+1;

end


%% LLH score and gradient for a given value of hyperparameters
function [errorscore, grad, obj] = gum_score(obj, HP, idx)

persistent UU fval nfval;
%% first call with no input: clear persisitent value for best-fitting parameters
if nargin==0
    fval = [];
    UU = [];
    nfval = 0;
    return;
end
if isempty(fval)
    fval = Inf;
end

nMod = obj.nMod;
M = obj.regressor;

if ~isempty(UU)
    for m=1:nMod
        M(m).U = UU{m};
    end
end

% assign hyperparameter values to regressors
M = assign_hyperparameter(M, HP, idx);

param = obj.param;
param.spectralback = 0;
if strcmp(param.verbose, 'full')
    param.verbose = 'on';
elseif strcmp(param.verbose, 'on')
    param.verbose = 'little';
else
    param.verbose = 'off';
end

if nfval>0
    param.initialpoints = 1; % just one initial point after first eval
end

% evaluate covariances at for given hyperparameters
if nargout>1
    % [M, spectral, spectc,param.gradient_hyperparameters] = covfun_eval(M,P);
    [obj.regressor,param.gradient_hyperparameters] = compute_prior_covariance(M);
else % same without gradient
    %  [M, spectral, spectc] = covfun_eval(M,P);
    obj.regressor = compute_prior_covariance(M);
end

%% estimate GUM weights

obj = obj.infer(param);

if nargout>1
    grad = obj.score.grad;
end

n = obj.nObs;
errorscore = -obj.score.validationscore * n; % neg-LLH (i.e. cross-entropy)
if nargout>1
    grad = -grad * n; % gradient
end

% for spectral decomposition, convert weights back from spectral to
% original domain
%M = weightprojection(M,S.covb, nMod,spectral,spectc,rank);
obj.regressor = project_from_spectral(obj.regressor);

% if best parameter so far, update the value for initial parameters
if errorscore < fval
    fval = errorscore;
    UU = cell(1,nMod); % group all weights into a cell array
    for m=1:nMod
        UU{m} = obj.regressor(m).U;
    end
end

nfval = nfval+1;

end

%% converts to cell if not already a cell
function x= tocell(x)
if ~iscell(x)
    x = {x};
end
end

%% negative cross-entropy of prior and posterior
% multivariate gaussians (for M-step of EM in hyperparameter optimization)
function [Q, grad] = mvn_negxent(covfun, mu, m, V, P, HP, HPs)
k = size(P,1); % dimensionality of free space
nP = length(HPs.HP); % number of hyperparameters

HPs.HP(HPs.fit) = HP;

msgid1 = warning('off','MATLAB:nearlySingularMatrix');
msgid2 = warning('off','MATLAB:SingularMatrix');

mdif = (m-mu)*P'; % difference between prior and posterior means
[S, gS] = covfun(HPs.HP); % covariance matrix and gradient w.r.t hyperparameters
if isstruct(gS), gS = gS.grad; end
S = P*S*P'; % project on free base
SV = S \V;

% compute log-determinant (faster and more stable than log(det(X)),
% from Pillow lab code), since matrix is positive definite
%[C,noisPD] = chol(S);
[C,noisPD] = chol(SV); % instead of log det(S), we compute log det(S^-1 V) which may deal better with non-full rank covariance prior
if noisPD % if not semi-definite (must be a zero eigenvalue, so logdet is -inf)
    %    warning('covariance is not semi-definite positive, hyperparameter fitting through EM may not work properly, you may change the covariance kernels or try fitting by cross-validation instead');
    LD = logdet(S);
    LD = logdet(SV);
    %Q = inf;
    %grad = nan(1,nP);
    %return;
else
    LD = 2*sum(log(diag(C)));
end

% cross-entropy
%Q = -(trace(SV) +LD + (mdif/S)*mdif' + k*log(2*pi)   )/2;
Q = -(trace(SV) -LD + (mdif/S)*mdif' + k*log(2*pi)   )/2;


% gradient
grad= zeros(1,nP);
for p=1:nP
    this_gS = P*gS(:,:,p)*P';
    SgS = S\this_gS;
    grad(p) = (  trace( SgS*(SV-eye(k)))   + mdif*SgS*(S\mdif')  )/2;
end

warning(msgid1.state,'MATLAB:nearlySingularMatrix');
warning(msgid2.state,'MATLAB:SingularMatrix');


Q = -Q;
grad = -grad;
grad = grad(HPs.fit);
end


%% compute log-determinant (copied from GPML)
function ldB = logdet(A)
n = size(A,1);
[L,U,P] = lu(A); u = diag(U);         % compute LU decomposition, A = P'*L*U
signU = prod(sign(u));                                           % sign of U
detP = 1;               % compute sign (and det) of the permutation matrix P
p = P*(1:n)';
for i=1:n                                                     % swap entries
    if i~=p(i), detP = -detP; j = find(p==i); p([i,j]) = p([j,i]); end
end
if signU~=detP     % log becomes complex for negative values, encoded by inf
    ldB = Inf;
else          % det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
    ldB = sum(log(abs(u)));
end
end

%% compute log-factorial
function L = logfact(T)
L = zeros(length(T),1);
for i=1:length(T)
    if T(i)>100 % Stirling approximation to log n!
        L(i) = (T(i) + 0.5)*log(T(i) + 0.5) - 0.434294*(T(i) + 0.5)*log(10) + 0.399090*log(10);
    else
        L(i) = log(factorial(T(i)));
    end
end
end

%% parse formula
function [M,T, param] = parse_formula(Tbl,fmla, param)
if ~ischar(fmla)
    error('if first argument is a table, the second must be a formula string');
end
VarNames = Tbl.Properties.VariableNames; % list of all columns

%% first let's do the easy: find the target variable
i = find(fmla=='~');
if isempty(i)
    error('formula must contain the character ''~~''');
elseif length(i)>1
    error('Character ''~~'' must be present only once in formula');
end

T_fmla = trimspace(fmla(1:i-1)); % characters before ~

[target, T_fmla] = starts_with_word(T_fmla, VarNames);

if isempty(target)
    error('formula should start with dependent variable from the list of table variables');
end


T = Tbl.(target);
nObs = length(T);

% check if splitting model
if ~isempty(T_fmla)
    if ~T_fmla(1) == '|'
        error('dependent variable in formula should be followed either by ''~'' or ''|'' symbol');
    end
    
    SplitVar = trimspace(T_fmla(2:end));
    if ~ismember(SplitVar, VarNames)
        error('Variable ''%s'' is not present in the table', SplitVar);
    end
    
    param.split = Tbl.(SplitVar);
end

fmla(1:i) = []; % remove left part of formula
fmla = trimspace(fmla);

%% now let's get serious and parse the predictor formula
V = {}; % list of variables
O = ''; % list of operators

option_list = {'sum','mean','tau','variance','binning','constraint'};


while ~isempty(fmla)
    [transfo, fmla] = starts_with_word(fmla, {'f(','cat('});
    if ~isempty(transfo)
        if strcmp(transfo, 'f(')
            type = 'continuous';
        else
            type = 'categorical';
        end
        fmla = trimspace(fmla);
        [v, fmla] = starts_with_word(fmla, VarNames);
        if isempty(v)
            error('''%s'' in formula must be followed by variable name',transfo);
        end
        
        
        
        %% process regressor options
        opts = struct();
        
        % check for split variable first
        [opts, fmla] = process_split_variable(fmla, opts, Tbl);
        
        while fmla(1)~=')'
            if fmla(1) ~= ','
                error('incorrect character in formula (should be comma or closing parenthesis) at ''%s''', fmla);
            end
            fmla(1) = [];
            fmla = trimspace(fmla);
            
            i = find(fmla=='=',1);
            if isempty(i)
                error('missing ''='' sign for option in formula at ''%s''', fmla);
            end
            
            option = trimspace(fmla(1:i-1));
            if ~any(strcmpi(option, option_list))
                error('incorrect option type: %s',option);
            end
            fmla(1:i) = [];
            fmla = trimspace(fmla);
            
            switch lower(option)
                case {'tau','variance','binning'}
                    [Num, fmla] = starts_with_number(fmla);
                    if isnan(Num)
                        error('Value for option ''%s'' should be numeric',option);
                    end
                    opts.(option) = Num;
                case {'sum','mean'}
                    [Num, fmla] = starts_with_number(fmla);
                    if isnan(Num)
                        error('Value for option ''%s'' should be numeric',option);
                    end
                    if Num==0
                        opts.constraint='b';
                    elseif Num==1 && strcmpi(option, 'sum')
                        opts.constraint='s';
                    elseif Num==1 && strcmpi(option, 'mean')
                        opts.constraint='m';
                    else
                        error('%s=%f: not coded yet', option, Num);
                    end
                case 'constraint'
                    i = 1;
                    opts.constraint = fmla(1);
                    fmla = trimspace(fmla(2:end));
            end
            
            
        end
        fmla = trimspace(fmla(2:end));
        
        
        % i = find(fmla==')',1);
        % if isempty(i)
        %    error('missing closing parenthesis in formula');
        % end
        % opts = trimspace(fmla(1:i-1));
        
        V{end+1} = struct('variable',v, 'type',type, 'opts', opts);
        
        
    else
        [v, fmla] = starts_with_word(fmla, VarNames);
        
        if ~isempty(v) %% linear variable
            
            opts = struct;
            
            % check for split variable first
            [opts, fmla] = process_split_variable(fmla, opts, Tbl);
            
            if iscategorical(Tbl.(v))
                type = 'categorical';
            else
                type = 'linear';
            end
            
            V{end+1} = struct('variable',v, 'type',type, 'opts', opts);
            
        elseif ~isnan(str2double(fmla(1))) %% numeric constant
            
            [Num, fmla] = starts_with_number(fmla);
            V{end+1} = struct('variable',Num, 'type','constant','opts',struct());
        else
            error('could not parse formula at point ''%s''', fmla);
            
        end
    end
    
    if isempty(fmla)
        break;
    end
    
    %% Now let's look for operation
    if ~any(fmla(1)=='+*')
        error('Was expecting an operator in formula at point ''%s''', fmla);
    end
    O(end+1) = fmla(1);
    
    fmla(1) = [];
    fmla = trimspace(fmla);
    
    fmla = trimspace(fmla);
end

%% build the regressors
for v=1:length(V)
    w = V{v}.variable;
    if ischar(w)
        x = Tbl.(w);
        label = w;
    else
        x = w*ones(nObs,1);
        label = num2str(w);
    end
    
    opts_fields = fieldnames(V{v}.opts);
    opts_values = struct2cell(V{v}.opts);
    opts_combined = [opts_fields opts_values]';
    
    V{v} = regressor(x, V{v}.type, 'name',label, opts_combined{:});
end

%% build predictor from operations between predictors
while length(V)>1
    
    % first perform products
    if any(O=='*')
        v = find(O=='*',1);
        
        V{v} = V{v} * V{v+1}; % compose product
        
        % remove regressor and operation
        V(v+1) = [];
        O(v) = [];
        
    else % now we're left with additions
        
        V{1} = V{1} + V{2}; % compose addition
        
        % remove regressor and operation
        V(2) = [];
        O(1) = [];
        
    end
    
end

M = V{1};

end

%% check if string starts with any of possible character strings
function [word, str] = starts_with_word(str, WordList)
word = '';
for w=1:length(WordList)
    wd = WordList{w};
    if length(str)>=length(wd) && strcmp(str(1:length(wd)),wd)
        %we've got a match
        word = wd;
        str(1:length(wd)) = []; % remove corresponding characters from string
        
        str = trimspace(str);
        
        return;
    end
end
end

%% check if string starts with a number
function  [Num, fmla] = starts_with_number(fmla)

% find longest string that corresponds to numeric value
Num = nan(1,length(fmla));
for i=1:length(fmla)
    Num(i) = str2double(fmla(1:i));
end
i = find(~isnan(Num) & ~(fmla==','),1,'last');

if isempty(i)
    Num = nan;
else
    Num = Num(i);
    
    
    fmla(1:i) = [];
    fmla = trimspace(fmla);
end
end

%% process split variable
function [opts, fmla] = process_split_variable(fmla, opts, T)



if ~isempty(fmla) && fmla(1)=='|'
    fmla = trimspace(fmla(2:end));
    
    VarNames = T.Properties.VariableNames;
    [v_split, fmla] = starts_with_word(fmla, VarNames);
    if isempty(v_split)
        error('| in formula must be followed by variable name');
    end
    
    opts.split = T.(v_split);
end
end

%% trim space in character string
function str = trimspace(str)
j=find(str~=' ',1);
if isempty(j)
    str = '';
    return;
elseif j>1
    str(1:j-1) = [];
end

j=find(str~=' ',1,'last');
str(j+1:end) = [];


end

%% display character to indicate fitting progress
function char = DisplayChar(i,bool)
if ~mod(i,100)
    char = 'O\n';
elseif ~mod(i,20)
    char = 'O';
elseif ~mod(i,10)
    char = 'o';
elseif ~mod(i,5)
    char = 'x';
elseif nargin>1 && bool
    char = '>';
else
    char = '*';
end
end