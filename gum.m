classdef gum

    % Defines a Generalized Unrestricted Model (GUM). Two possible
    % syntaxes:
    % * function M= gum(X, T, param)
    %
    % where X is a regressor, a vector of regressors (type "help regressor" for more information) or a numerical array.
    % Param is a structure with optional parameters (see below).
    %
    % * function M= gum(Tbl, fmla, param) where Tbl is a table and fmla is a
    % string array describing the formula of the GUM using table entries
    % (i.e. fmla = ' y ~ f(x)*f(z) + t + 1', where 'x', 'y', 'z' and 't' are
    % variables in the table. See below for further information on how to
    % define the formula.
    %
    % Possible fields in params are:
    % -'observations': 'binomial' (default), 'normal' or 'poisson'
    %
    % -'w': is a weighting vector with length equal to the number of training examples
    % - 'split': add
    % - 'constant': whether a regressor of ones is included (to capture the
    % intercept). Possible values: 'on' [default], 'off'
    %
    % - 'mixture': for mixture of GUMS: 'multinomialK', 'hmmK', 'regressionK' where K is the number of models.
    % (e.g. 'hmm3' for 3-state HMM). The value can also be a user-defined mixture model (defined with mixturemodel). Other fields may be provided:
    % - 'indexComponent': to which component each regressor object is assigned to. If not provided, all regressors are duplicated
    % (i.e. each state corresponds to the same model but possibly with different weights). indexComponent must be a vector of indices of length K
    % - 'mixture_regressors' (for 'regressionK' only): matrix of regressors defining which model is used for each observation.
    % - 'mixture_category' (for 'multinomialK' only): vector of indices of length nObs defining which mixture model is used for each observation
    %
    % Principal methods to run on a GUM (see full list at the end):
    % - inference using M = M.infer();
    % - fitting hyperparameters using M = M.fit();
    % - plotting weights using M = M.plot_weights();
    % Type 'help gum.infer', 'help gum.fit' for further help on these
    % methods.
    %
    % Defining GUM formulas:
    % Formulas are defined using the classic R-style syntax, e.g. to define
    % a simple GLM with y as dependent variable and x1, x2, x3 as regressors use:
    % fmla = 'y ~ x1 + x2 + x3'
    % Use simple regressor name for linear mapping (as above), use 'f(x1)'
    % for nonlinear mapping, e.g.:
    % fmla = 'y ~ f(x1) + f(x2) + x3' will define a GUM with nonlinear
    % mappings for variables x1 and x2 and linear mapping for x2.
    % Use * for multiplication of regressors, e.g.
    %'y ~ f(x1) * f(x2) + f(x3)' or 'y ~ f(x1) * (f(x2) + f(x3))'
    % Use [] for fixed regressors (no inferred mapping), e.g.
    % 'y ~ f(x)*[z] + f(t)
    % By default, a constant offset regressor is added. Add '0' in the
    % formula to exclude it (i.e. 'y~f(x1)+x2+0').
    % You can include numerical values (except 0) as fixed multipliers, e.g.
    % ' y~f(x) + 2*f(z)'
    % The minus sign can be used, e.g. 'f(x) - f(y)' is equivalent to
    % 'f(x)+ f(y)*(-1)
    %
    % Use 'cat(x)' instead of 'f(x)' if x is a categorical variable
    % Use | to define different mappings depending on another variable
    % (e.g. a random effect), e.g. 'x|z', '1|z' or 'f(x|z)'
    % Use 'f(x, options1=V1, options2=V2,...) to specify further options for the
    % nonlinear mapping. Possible options are:
    % - f(x, type=V) for specific regressor type (possible values:
    % 'linear','categorical', 'continous', 'periodic','none')
    % flin(x) is equivalent to f(x, type=linear)
    % fper(x) is equivalent to f(x, type=periodic)
    % - f(x;sum=V) to enforce the sum of regressor values to be equal to
    % V (either 1 or 0)
    % - f(x;mean=V) to enforce the mean of regressor values to be equal
    % to V (either 1 or 0)
    % - f(x;tau=V) to set an (initial) value V for the scale hyperparameter
    % - f(x; variance= V) to set an (initial) value V for the variance hyperparameter
    % - f(x; type=periodic; period=V) to define period of periodic
    % regressor
    % - f(x; fit=V) to define which hyperparameters can be fitted (others
    % are fixed). Possible values: 'all' (default), 'none', 'variance',
    % 'tau'
    % - f(x;binning=V) to bin values in variable in V bins (binned by quantiles)
    % Use ' y|x ~ ...' if you want to define separate GUM for each level in
    % variable 'x' (the output will be a vector of GUM objects).
    %


    % TODO:
    % - boostrapping (need to work on sparsearray to allow for X(ind, ...) - TEST
    % - no prior, ARD (TEST) - Matern prior
    % - add anova type formulas x1:x2 (FINISH, TEST), x1*x2
    % - spectral trick (marginal likelihood in EM sometimes decreases - because of change of basis?)
    % - spectral trick for periodic: test; change prior (do not use Squared
    % Exp)
    % - negative binomial, probit (finish)
    % - raised cosine, correct default HPs
    % - 'fourier' for periodic (TEST)
    % - link functions as in glmfit
    % - CV gradient for dispersion parameter
    % - CV gradient for parameterized basis functions
    % - crossvalidation compatibility with matlab native (done?)
    % - constraint structure (e.g. when concat weights or split)
    % - prior mean function (mixed effect; fixed effect: mean with 0 covar)
    % - use fitglme/fitlme if glmm model
    % - work on labels
    % - add label on models (or just dataset label?)
    % - reset: weights for gum, clearing all scores and predictions
    % - plot_vif
    % - add number of (free) HPs to metrics
    % - plot with 2D scale
    % - move plot methods to regressors
    % - allow parallel processing for crossvalid & bootstrap
    % - add mixture object (lapses; built-in: dirichlet, multiple dirichlet, HMM, multinomial log reg)
    % - add history regressors (with exp basis)
    % - knock-out model
    % - print summary, weights, hyperparameters
    % - allow EM for infinite covariance:  we should remove these from computing logdet, i.e. treat them as hyperparameters (if no HP attached)

    %
    % Complete list of methods:
    % OPERATIONS AND TESTS ON MODEL
    % -'extract_observations': extract model for only a subset of
    % observations
    % - 'isgam': whether model corresponds to a GAM
    % - 'isglm': whether model corresponds to a GLM
    % - 'isestimated': whether model weights have been estimated
    % - 'is_infinite_covariance': whether any regressor has infinite
    % covariance (i.e. no prior)
    % - 'number_of_regressors': number of regressors
    % - 'concatenate_weights': concatenates model weights into vector
    % - 'clear_data'
    % - 'save': save model into .mat file
    %
    % DEALING WITH MULTIPLE MODELS
    % - 'split': split model depending on value of a vector
    % - 'concatenate_over_models': concatenate weights over models
    % - 'concatenate_score': concatenate socre from array of models
    % - 'population_average': averages weights over models
    %
    % NUMERICAL METHODS
    % - 'compute_rho_variance' provide the variance of predictor in output structure
    % - 'vif': Variance Inflation Factor
    % - 'sample_weights_from_prior': assign weight values sampling from
    % prior
    % - 'Predictor': compute the predictor for each observation
    % - 'ExpectedValue': computes the expected value for each observation
    % - 'LogPrior': computes the LogPrior of the model
    % - 'LogLikelihood': computes the LogLikelihood of the model
    % - 'Accuracy': accuracy of the model predictions at MAP weights
    % - 'LogJoint': computes the Log-Joint of the model
    % - 'Hessian': computes the Hessian of the negative LogLihelihood
    % - 'PosteriorCov': computes the posterior covariance
    % - 'IRLS': core step in inference
    % - 'sample': generate observations from model
    % - 'ExplainedVariance': computes model explained variance
    % - 'predictor_variance': computes the variance of the predictors across
    % datapoints
    % - 'compute_rho_variance': computes variance of predictor for each
    % datapoint
    % - 'inverse_link_function': output inverse link function
    %
    % PLOTTING:
    % - 'plot_design_matrix': plot design matrix
    % - 'plot_posterior_covariance': plot posterior covariance for each
    % regressor
    % - 'plot_weights': plot estimated weights and functions
    % - 'plot_data_vs_predictor': plots individual observed data points vs
    % predictor
    % - 'plot_hyperparameters': plot hyperparameter values
    % - 'plot_score': plot scores of different models (model comparison)
    % - 'plot_basis_functions': plot regressor basis functions
    %
    % version 0.0. Bug/comments: send to alexandre.hyafil@gmail.com
    %

    properties
        regressor
        T
        obs
        link
        nObs = 0
        nMod = 0
        ObservationWeight
        param = struct()
        grad
        score = struct()
        Predictions = struct('rho',[])
        mixture = []
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

            % check regressors
            for m=1:nMod
                M(m) = checkregressorsize(M(m),n);
            end

            assert(isstruct(param), 'param should be a structure');

            % observation weighting
            if isfield(param,'ObservationWeights')
                param.ObservationWeight = param.ObservationWeights;
            end
            if isfield(param,'ObservationWeight')
                obj.ObservationWeight = param.ObservationWeight;
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
                    RR = cell(1,ndims(M(m).Data));
                    RR{1} = nRep;
                    for d=2:ndims(M(m).Data)
                        RR{d} = 1;
                    end
                    M(m).Data =repelem(M(m).Data,RR{:});
                    M(m).nObs = n;
                end
                obj.nObs = n;

            end
            obj.T = T;

            %% parse parameters
            if isfield(param,'observations')
                obs = param.observations;
                obs = strrep(obs, 'count','poisson');
                obs = strrep(obs, 'binary','binomial');
                obs = strrep(obs, 'gaussian','normal');
                assert(any(strcmp(obs, {'normal','binomial','poisson'})), 'incorrect observation type: possible types are ''normal'',''binomial'' and ''poisson''');
            else
                obs = 'binomial';
            end
            if strcmp(obs,'binomial') && any(T~=0 & T~=1)
                error('for binomial observations, T values must be 0 or 1');
            elseif strcmp(obs, 'poisson') && any(T<0)
                error('for count observations, all values must be non-negative');
            end
            if all(T==0) || all(T==1)
                warning('T values are all 0 or 1, may cause problem while fitting');
            end
            obj.obs = obs;

            % link function
            switch obs
                case 'normal'
                    obj.link = 'identity';
                case 'binomial'
                    obj.link = 'logit';
                case 'poisson'
                    obj.link = 'log';
            end

            %% add constant bias ('on' by default)
            if ~isfield(param,'constant') || strcmpi(param.constant, 'on')
                % let's see whether we should add a prior or not on this
                % extra weight
                GCov = global_prior_covariance(compute_prior_covariance(M));
                if  isempty(GCov) || any(isinf(diag(GCov))) % if any other regressor has infinite variance
                    % Mconst.HP.HP = Inf;
                    % Mconst.HP.fit = false;
                    const_prior_type = 'none';
                else
                    const_prior_type = 'L2';
                end
                clear GCov;

                % create regressor
                Mconst = regressor(ones(n,1),'linear','label','offset', 'prior',const_prior_type);

                M = [M,Mconst]; %structcat(M, Mconst); % append this component
                nMod = nMod +1;
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

                % weight constraint
                cc = [M(m).Weights.constraint];
                %  if ~isempty(cc)
                if ~all(any(cc(:)== ['fbms1n' 0],2))
                    error('Field constraint must be composed of the following characters: ''f'', ''b'',''m'',''s'',''1'',''n''');
                end
            end

            obj.param = param;
            obj.regressor = M;

            obj = obj.computer_n_parameters_df; % compute number of parameters and degrees of freedom

            %% mixture model
            if isfield(param, 'mixture') && ~isempty(param.mixture)
                if isa(param.mixture, 'mixturemodel') % user-defined mixture object
                    obj.mixture = param.mixture;
                else
                    assert(isstr(param.mixture), 'incorrect value for parameter mixture');
                    [mixture_type, str] = starts_with_word(param.mixture, {'multinomial', 'hmm', 'regression'});
                    assert(~isempty(mixture_type),['incorrect mixture type: ' param.mixture])

                    K = str2num(str); % number of components

                    % assign regressors to each model
                    if isfield(param, 'indexComponent') && ~isempty(param.indexComponent)
                        assert(length(param.indexComponent)==obj.nMod, 'the length of indexComponent must match the number of regressors in model')
                    else
                        % no index component: replicate model
                        assert(~isempty(K), 'missing value for K at the end of mixture parameter');
                        param.indexComponent = repelem(1:K, obj.nMod);
                        obj.nMod = K* obj.nMod;
                        obj.regressor = repmat(obj.regressor,1,K); % replicate regressors
                    end

                    MxtVarargin = {};
                    if isfield(param.mixture_regressors)
                        assert(size(param.mixture_regressors,1)==obj.nObs)
                        MxtVarargin = {param.mixture_regressors};
                    elseif isfield(param.mixture_category)
                        MxtVarargin = {param.mixture_category};
                    end
                    obj.mixture = mixturemodel(K, obj.nObs,mixture_type, MxtVarargin{:});
                end
            end

            %% split model for population analysis
            if isfield(param, 'split')
                obj = split(obj, param.split);
            end


        end

        %% SELECT SUBSET OF OBSERVATIONS FROM MODEL
        function obj = extract_observations(obj,subset)
            % M = extract_observations(M, subset) generates a model with
            % only observations provided by vector subset. subset is either
            % a vector of indices or a boolean vector indicating
            % observations to be included.

            obj.T = obj.T(subset);
            if ~isempty(obj.ObservationWeight)
                obj.ObservationWeight = obj.ObservationWeight(subset);
            end
            n_obs = length(obj.T);

            obj.nObs = n_obs;
            for m=1:length(obj.regressor) % for each module
                obj.regressor(m) = extract_observations(obj.regressor(m), subset);
                %  obj.regressor(m).Data =   extract_observations(obj.regressor(m).Data,subset); % extract for this module
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

            % update number of parameters and degrees of freedom
            obj = computer_n_parameters_df(obj);

        end

        %% SPLIT OBSERVATIONS IN MODEL
        function  objS = split(obj, S)
            % M = split(M,S) splits model M into array of models M with one
            % model for each value of S. S should be a vector of the same
            % length as the number of observations in M.

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
                objS(v).param.split = (S==V(v)); % know the index of selected datapoints is useful for stratified cross-validation

                objS(v).score.Dataset = V(v); % give label to dataset
            end

        end


        %% FIT HYPERPARMETERS (and estimate) %%%%%%%%%%%
        function obj = fit(obj, param)
            % M = M.fit(); or M = M.fit(param);
            % FITS THE HYPERPARAMETERS OF A GUM.
            % Optional parameters are passed in structure param with optional
            % fields:
            % - 'HPfit': which method is used to fit hyperparameters. Possible values
            % are:
            % * 'basic': basic grid search for parameters that maximize marginalized
            % likelihood
            % * 'EM': Expectation-Maximization algorithm to find hyperparameters that
            % maximize marginalized likelihood [default]
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
            % - 'gradient' (for crossvalidation fitting): whether to use gradients of CVLL to speed up search.
            % Gradients computation may suffer numerical estimation problems if
            % covariance are singular or close to singular. Possible values are: 'off'
            % [default], 'on' and 'check' (uses gradient but checks first that
            % numerical estimate is correct)
            %
            % -'CovPriorJacobian': provide gradient of prior covariance matrix w.r.t hyperparameters
            % to compute gradient of MAP LLH over hyperparameters.
            % - 'maxiter_HP': integer, defining the maximum number of iterations for
            % hyperparameter optimization (default: 200)
            % - 'no_fitting': if true, does not fit parameters, simply provide as output variable a structure with LLH and
            % accuracy for given set of parameters (values must be provided in field
            % 'U')

            % default values
            HPfit = 'em'; % default algorithm
            display = 'iter'; % default verbosity
            use_gradient = 'on'; % use gradient for CV fitting
            HP_TolFun = 1e-3; % stopping criterion for hyperparameter fitting

            if nargin==1
                param = struct;
            end

            if length(obj)>1
                %% fitting various models

                for i=1:numel(obj) % use recursive call
                    fprintf('fitting model %d/%d...\n', i, numel(obj));
                    obj(i) = obj(i).fit(param); % fit individual model
                    fprintf('\n');
                end
                return;
            end

            M = obj.regressor;

            %% which order for components
            for m=1:obj.nMod
                if isempty(M(m).ordercomponent) && M(m).nDim>1
                    dd = M(m).nDim; % dimension in this module
                    cc = [M(m).Weights.constraint]; % constraint for this module
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
            end

            %%  check fitting method
            if isfield(param, 'HPfit') % if specified as parameter
                HPfit = param.HPfit;
                assert(ischar(HPfit) && any(strcmpi(HPfit, {'em','cv','basic'})), 'incorrect value for field ''HPfit''');
            end

            if isfield(param, 'gradient')
                use_gradient = param.gradient;
                assert(ischar(use_gradient) && any(strcmpi(use_gradient, {'on','off','check'})), 'incorrect value for field ''gradient''');
            end

            %%  check cross validation parameters
            if isfield(param, 'crossvalidation')
                obj.param.crossvalidation =  param.crossvalidation;
            end
            obj = check_crossvalidation(obj);

            %% run optimization over hyperparameters

            %if HPfit %% if fitting hyperparameters
            HPall = [M.HP]; % concatenate HP structures across regressors
            HPini = [HPall.HP]; %hyperparameters initial values
            HP_LB = [HPall.LB]; % HP lower bounds
            HP_UB = [HPall.UB]; % HP upper bounds
            HP_fittable = logical([HPall.fit]);  % which HP are fitted

            % retrieve number of fittable hyperparameters and their indices
            % for each regressor set
            [HPidx, nHP] = get_hyperparameters_indices(M);

            if sum(cellfun(@(x) sum(x,'all'),nHP))==0
                % if requested to fit but there are no hyperparameter in
                % the model...
                fprintf('required optimization of hyperparameters but the model has no hyperparameter, inferring instead!\n' );
                obj = obj.infer();
                return;
            end

            % select hyperparmaters to be fitted
            HPini = HPini(HP_fittable);
            HP_LB = HP_LB(HP_fittable);
            HP_UB = HP_UB(HP_fittable);

            %% optimization parameters
            if isfield(param, 'maxiter_HP')
                maxiter = param.maxiter_HP;
            else
                maxiter = 200;
            end
            if isfield(param, 'initialpoints')
                obj.param.initialpoints = param.initialpoints;
            end

            %% optimize hyperparameters
            switch lower(HPfit)
                case 'basic' % grid search  hyperpameters that maximize marginal evidence

                    % clear persisitent value for best-fitting parameters
                    gum_neg_marg();

                    % run gradient descent on negative marginal evidence
                    errorscorefun = @(HP) gum_neg_marg(obj, HP, HPidx);
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
                    HP = HPini; % initial values for hyperparameters
                    param2 = obj.param;
                    param2.HPfit = 'none';
                    param2.crossvalidation = [];
                    param2.originalspace = false;

                    objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient
                    check_grad = strcmpi(use_gradient,'check'); % whether to use gradient

                    param_tmp = obj.param;
                    obj.param = param2;

                    while still % EM iteration
                        %% E-step (running inference with current hyperparameters)

                        M = set_hyperparameters(M, HP, HPidx);

                        % evaluate covariances at for given hyperparameters
                        %M2 = compute_prior_covariance(M);
                        obj.regressor = compute_prior_covariance(M);

                        % make sure that all weigths have real-valued priors (not infinite)
                        if iter==0

                            if is_infinite_covariance(obj)
                                error('infinite covariance (probably some regressor have no prior), cannot use the ''em'' algorithm for fitting');
                                %% instead we should remove these from computing logdet, i.e. treat them as hyperparameters (if no HP attached)
                            end

                            param2.verbose = 'little';
                        else
                            param2.verbose = 'off';
                        end

                        if iter==1
                            param2.initialpoints = 1;
                        end

                        % inference
                        obj = obj.infer(param2);

                        PP = projection_matrix(M); % projection matrix for each dimension


                        %% M-step (adjusting hyperparameters)
                        midx = 0;
                        HPidx_cat = [HPidx{:}]; % concatenate over components
                        ee = 1;
                        for m=1:obj.nMod % for each regressor object
                            ss = obj.regressor(m).nFreeParameters; % size of each dimension

                            for d=1:M(m).nDim % for each dimension
                                for r=1:size(HPidx{m},1)
                                    this_HPidx = HPidx{m}{r,d}; % indices of hyperparameters for this set of weight
                                    if ~isempty(this_HPidx)
                                        if ~any(ismember(this_HPidx, [HPidx_cat{setdiff(1:obj.nMod,ee)}])) %
                                            % if hyperparameters are not used in any other module

                                            if ~isa(M(m).Prior(1,d).CovFun, 'function_handle') % function handle
                                                error('hyperparameters with no function');
                                            end

                                            % set hyperparameter values for this component
                                            HPs = M(m).HP(r,d);
                                            HP_fittable = HPs.fit;
                                            HPs.HP(HP_fittable) = HP(this_HPidx); % fittable values

                                            % posterior mean and covariance for associated weights
                                            reg_idx = (1:ss(r,d)) + sum(ss(:,1:d-1),'all') + sum(ss(1:r-1,d)) + midx; % index of regressors in design matrix

                                            this_cov =  obj.score.FreeCovariance(reg_idx,reg_idx) ; % free posterior covariance for corresponding regressor

                                            % if project on a
                                            % hyperparameter-dependent
                                            % basis, move back to original
                                            % space
                                            W = obj.regressor(m).Weights(d); % corresponding set of weight

                                            this_mean =  W.PosteriorMean(r,:);
                                            this_scale = W.scale;
                                            this_P = PP{m}{r,d};
                                            this_Prior = obj.regressor(m).Prior(r,d);
                                            this_PriorMean = this_Prior.PriorMean;
                                            B = W.basis; % here we're still in projected mode

                                            if ~isempty(B)
                                                if ~B.fixed
                                                    % working in original
                                                    % space (otherwise working
                                                    % in projected space)
                                                    this_mean = this_mean*B.B;
                                                    this_PriorMean = this_PriorMean*B.B;
                                                    this_cov = B.B' * this_cov *B.B;

                                                    assert(W.constraint=='f', 'EM not coded for non-free basis');
                                                else
                                                    2;
                                                end
                                                this_P = eye(length(this_cov));

                                            end
                                            this_cov = force_definite_positive(this_cov);
                                            %this_cov_chol = chol(this_cov);

                                            [~, gg] = this_Prior.CovFun(this_scale, HPs.HP, B);
                                            if isstruct(gg)  && isequal(this_P, eye(ss(r,d)))  && (isempty(B)||B.fixed) % function provided to optimize hyperparameters
                                                %  work on this to also take into account constraints (marginalize in the free base domain)

                                                HPnew = gg.EM(this_mean,this_cov); % find new set of hyperparameters
                                            else % find HP to maximize cross-entropy between prior and posterior

                                                optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                                                    'CheckGradients',check_grad,'display','off','MaxIterations',1000);

                                                % find HPs that minimize negative
                                                % cross-entropy
                                                mstep_fun = @(hp) mvn_negxent(this_Prior.CovFun, this_PriorMean, this_scale, this_mean, this_cov,this_P, hp, HPs, B);

                                                % if isinf(mstep_fun(HP(this_HPidx)))
                                                %     2;
                                                % end

                                                ini_val = mstep_fun(HP(this_HPidx));

                                                assert(~isinf(ini_val) && ~isnan(ini_val), ...
                                                    'M step cannot be completed, probably because covariance prior is not full rank');

                                                % compute new set of
                                                % hyperparameters that
                                                % minimize
                                                HPnew = fmincon(mstep_fun,...
                                                    HP(this_HPidx),[],[],[],[],M(m).HP(r,d).LB(HP_fittable),M(m).HP(r,d).UB(HP_fittable),[],optimopt);
                                            end
                                            HP(this_HPidx) = HPnew;  % select values corresponding to fittable HPs

                                        else
                                            error('not coded: cannot optimize over various components at same time');
                                        end
                                    end
                                    ee = ee+1;
                                end

                            end
                            midx = midx + sum(ss(1,:)) * rank(m); % jump index by number of components in module
                        end

                        % for decomposition in basis functions, convert weights back to
                        % original domain
                        obj.regressor = project_from_basis( obj.regressor);

                        % has converged if improvement in LLH is smaller than epsilon
                        iter = iter + 1; % update iteration counter;
                        LogEvidence = obj.score.LogEvidence;
                        fprintf('HP fitting: iter %d, log evidence %f\n',iter, LogEvidence);
                        % HP
                        converged = abs(old_logjoint-LogEvidence)<HP_TolFun;
                        old_logjoint = LogEvidence;
                        logjoint_iter(iter) = LogEvidence;

                        still = (iter<maxiter) && ~converged;

                    end
                    %  obj = obj;
                    % obj.regressor = M2;
                    obj.param = param_tmp;

                case 'cv' % gradient search to minimize cross-validated log-likelihood

                    % clear persisitent value for best-fitting parameters
                    cv_score();

                    objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient
                    check_grad = strcmpi(use_gradient,'check'); % whether to use gradient

                    % run gradient descent
                    errorscorefun = @(P) cv_score(obj, P, HPidx, 0);
                    optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                        'CheckGradients',check_grad,'display',display,'MaxIterations',maxiter);
                    HP = fmincon(errorscorefun, HPini,[],[],[],[],HP_LB,HP_UB,[],optimopt); % optimize

                    %% run estimation again with the optimized hyperparameters to retrieve weights
                    obj =  cv_score(obj, HP, HPidx, 1);
            end


            % allocate fitted hyperparameters to each module
            M = M.set_hyperparameters(HP);
            %             for m=1:obj.nMod
            %                 for d=1:M(m).nDim
            %                     for r=1:size(M(m).HP,1)
            %                         this_HP_fittable = M(m).HP(r,d).fit;
            %                         M(m).HP(r,d).HP(this_HP_fittable) = HP(HPidx{m}{r,d}); % associate each parameter to corresponding module
            %                     end
            %                 end
            %             end

        end



        %% %%%%% INFERENCE (ESTIMATE WEIGHTS) %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function   obj = infer(obj, param)
            % M = M.infer(); or M = M.infer(param);
            % INFERS (ESTIMATES) WEIGHTS OF GUM USING LAPLACE APPROXIMATION.
            % Optional parameters are passed as fiels in structure param:
            %- 'maxiter': maximum number of iterations of IRLS estimation algorithm
            %(default 100)
            %- 'miniter': minimum number of iterations of IRLS estimation algorithm
            %(default 4)
            %- 'TolFun': Tolerance value for LLH function convergence (default:1e-12)
            %
            %- 'ordercomponent': whether to order components by descending order of
            %average variance (default: true if all components have same constraints)
            %
            % - 'initialpoints': number of initial points for inference (default:10)
            % - 'verbose': 'on' (default),'off','little','full'
            %
            % M is the GUM model after inference.
            % M.regressor provides the estimated regressors, with fields
            % - 'se': a cell array composed of the vector of standard errors of the mean (s.e.m.)
            % for each set of weights.
            % - 'T': Wald T-test of significance for each weight
            % - 'p': associated probability%
            %
            % M.score is a structure with following fields:
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
                param.maxiter = 100; % maximum number of iterations
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

            if isfield(param, 'CovPriorJacobian') % whether we compute gradient for hyperparameters
                CovJacob = param.CovPriorJacobian; % cell with how prior for each weight dimension depends on each hyperparameter
                do_grad_hyperparam = 1;
            else
                CovJacob = [];
                do_grad_hyperparam = 0;
            end

            nM = obj.nMod;
            M = obj.regressor;

            if isgam(obj)
                obj.param.initialpoints = 1; % if no multilinear term, problem is convex so no local minima
            end


            %% evaluate prior covariances for given hyperparameters
            if verbose
                fprintf('Evaluating prior covariance matrix...');
            end
            M = compute_prior_covariance(M, false);
            if verbose
                fprintf('done\n');
            end

            %% compute prior mean and covariance and initialize weight
            M = M.check_prior_covariance();
            M = M.compute_prior_mean();
            M = M.initialize_weights(obj.obs);
            obj.regressor = M;

            singular_warn = warning('off','MATLAB:nearlySingularMatrix');

            % check hyper prior gradient matrix has good shape
            if do_grad_hyperparam
                % nParamTot = sum([obj.regressor.nTotalParameters]);  % total number of weights
                nParamTot = obj.score.nParameters;
                if size(CovJacob,1)~=nParamTot || size(CovJacob,2)~=nParamTot
                    error('The number of rows and columns in field ''CovPriorJacobian'' (%d) must match the total number of weights (%d)',size(CovJacob,1),nParamTot);
                end
            end

            %% fit weights on full dataset implement the optimization (modified IRLS) algorithm
            obj = IRLS(obj);
            Sfit = obj.score;
            M = obj.regressor;

            % score on test set
            if isfield(obj.param, 'testset') && ~isempty(obj.param.testset)
                testset = obj.param.testset;

                obj_test = extract_observations(obj,testset); % extract test set
                [obj_test,testscore] = LogLikelihood(obj_test); % evaluate LLH
                accuracy_test = Accuracy(obj_test); % and accuracy

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
                    %  obj.regressor(m).U_CV = cell(rank(m),M(m).nDim); % fitted parameters for each permutation
                    %                     for d=1:M(m).nDim
                    %                         obj.regressor(m).Weights(d).U_CV = cell(1,M(m).nDim); % fitted parameters for each permutation
                    %                         for r=1:rank(m)
                    %                          %   obj.regressor(m).U_CV{r,d} = zeros(nSet,obj.regressor(m).nWeight(d));
                    %                                                         obj.regressor(m).Weights(d).U_CV{r} = zeros(nSet,M(m).Weights(d).nWeight);
                    %
                    %                         end
                    %                     end
                    variance{m} = zeros(nSet,M(m).rank); % variance across observations for each component

                end
                % U_CV{1,D+1} = zeros(nperm,m(D+1));
                validationscore = zeros(1,nSet); % score for each (LLH per observation)
                accuracy = zeros(1,nSet); % proportion correct
                exitflag_CV = zeros(1,nSet); % exit flag (converged or not)
                U_CV = zeros(obj.score.nParameters,nperm);

                if do_grad_hyperparam
                    grad_hp = zeros(size(CovJacob,3),nSet); % gradient matrix (hyperparameter x permutation)

                    PP = projection_matrix(M,'all'); % global transformation matrix from full parameter set to free basis
                else
                    PP = []; % just for parfor
                end

                % spmd
                warning('off','MATLAB:nearlySingularMatrix');
                % end

                s = 1; % !!warning: dispersion parameter - change for normal observations

                % parfor p=1:nperm % for each permutation
                for p=1:nSet % for each permutation

                    if verbose
                        fprintf('Cross-validation set %d/%d', p, nSet);
                    end

                    trainset = CV.training(p);
                    validationset = CV.test(p);
                    if iscell(trainset) % CV structure
                        trainset = trainset{1};
                        validationset = validationset{1};

                    end

                    %fit on training set
                    obj_train = IRLS(extract_observations(obj,trainset));
                    exitflag_CV(p) = obj_train.score.exitflag;

                    allU = concatenate_weights(obj_train);
                    U_CV(:,p) = allU;

                    %compute score on testing set (mean log-likelihood per observation)
                    obj_v = extract_observations(obj,validationset); % model with validat
                    obj_v.regressor = set_weights(obj_v.regressor, obj_train.regressor); % set weights computed from training data
                    obj_v.Predictions.rho = [];

                    [obj_v,validationscore(p),grad_validationscore] = LogLikelihood(obj_v); % evaluate LLH, gradient of LLH
                    accuracy(p) = Accuracy(obj_v); % and accuracy

                    if isempty(obj.ObservationWeight)
                        n_Validation = length(validationset);
                    else
                        n_Validation = sum(obj.ObservationWeight(validationset));

                    end
                    validationscore(p) = validationscore(p)/ n_Validation; % normalize by number of observations
                    grad_validationscore = grad_validationscore / n_Validation; % gradient w.r.t each weight

                    % compute gradient of score over hyperparameters
                    if do_grad_hyperparam

                        PCov = PosteriorCov(obj_train); % posterior covariance computed from train data
                        this_gradhp = zeros(1,size(CovJacob,3));
                        for q=1:size(CovJacob,3) % for each hyperparameter
                            gradgrad = PP*CovJacob(:,:,q)'*allU'; % LLH derived w.r.t to U (free basis) and hyperparameter
                            gradU = - PP' * PCov * gradgrad;% derivative of inferred parameter U w.r.t hyperparameter (full parametrization)
                            this_gradhp(q) = grad_validationscore * gradU/s; % derivate of score w.r.t hyperparameter
                        end
                        grad_hp(:,p) = this_gradhp;
                    end

                    if verbose
                        fprintf('done\n');
                    end

                end

                M = set_weights(M,U_CV,[],'U_CV'); % add MAP weights for each CV set to weight structure
                warning(singular_warn.state,'MATLAB:nearlySingularMatrix');

                %  spmd
                %      warning(singular_warn.state,'MATLAB:nearlySingularMatrix');
                %  end

                n_nonconverged = sum(exitflag_CV<=0);
                if  n_nonconverged>0
                    warning('gum:notconverged', 'Failed to converge for %d/%d permutations', n_nonconverged, nSet);
                end

                Sfit.validationscore = mean(validationscore);
                Sfit.validationscore_all = validationscore;
                Sfit.accuracy_validation = mean(accuracy);
                Sfit.accuracy_all = accuracy;
                Sfit.exitflag_CV = exitflag_CV;
                Sfit.converged_CV = sum(exitflag_CV>0); % number of permutations with convergence achieved
                % S.variance_CV = variance_CV;
                if isfield(obj.param, 'testset')
                    Sfit.testscore = testscore;
                    Sfit.accuracy_test = accuracy_test;
                end
                if do_grad_hyperparam
                    Sfit.grad = mean(grad_hp,2); % gradient is mean of gradient over permutations
                end

            elseif do_grad_hyperparam % optimize parameters directly likelihood of whole dataset (not recommended)

                PP = projection_matrix(M,'all'); % projection matrix for each dimension

                %compute score on whole dataset (mean log-likelihood per observation)
                [obj,validationscore,grad_validationscore] = LogLikelihood(obj); % evaluate LLH
                Sfit.accuracy = Accuracy(obj); % and accuracy

                % explained variance

                if isempty(obj.ObservationWeight)
                    nWeightedObs = obj.nObs;
                else
                    nWeightedObs = sum(obj.ObservationWeight);
                end
                Sfit.validationscore = validationscore/ nWeightedObs; % normalize by number of observations
                grad_validationscore = grad_validationscore / nWeightedObs; % gradient w.r.t each weight

                allU = concatenate_weights(M);

                % [~,H] = PosteriorCov(obj); % Hessian w.r.t weights
                %Hinv = Hessian(obj); % Hessian w.r.t weights
                PCov = PosteriorCov(obj); % posterior covariance computed from train data
                grd = zeros(size(CovJacob,3),1);
                for q=1:size(CovJacob,3) % for each hyperparameter
                    % Ucat = cellfun(@(x) x.U, M, 'uniformoutput',0);
                    % Ucat = [Ucat{:}];
                    warning('looks like should be derivative for inverse of GHP here');
                    gradgrad = PP*CovJacob(:,:,q)'*allU'; % LLH derived w.r.t to U (free basis) and hyperparameter
                    gradU = - PP' * PCov * gradgrad;% derivative of inferred parameter U w.r.t hyperparameter (full parametrization)
                    grd(q) = grad_validationscore * gradU/s; % derivate of score w.r.t hyperparameter
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
            [Sfit.FreeCovariance, B, invHinvK]= PosteriorCov(obj);
            if verbose
                fprintf('done\n');
            end

            %Covariance
            P = projection_matrix(M,'all'); % projection matrix for each dimension
            Sfit.covb = P'*Sfit.FreeCovariance* P; % see covariance under constraint Seber & Wild Appendix E
            invHinvK = P'*invHinvK*P;

            % standard error of estimates
            all_se = sqrt(diag(Sfit.covb))';

            % T-statistic for the weights
            allU = concatenate_weights(M);
            all_T = allU ./ all_se;

            % p-value for significance of each coefficient
            %all_p = 1-chi2cdf(all_T.^2,1);
            all_p = 2*normcdf(-abs(all_T));

            % distribute values of se, T, p and V to different regressors
            % !! we should use set_weights method in regressor for that
            midx = 0;
            for m=1:obj.nMod
                rr = rank(m);
                ss = [M(m).Weights.nWeight];
                dd = M(m).nDim;

                for d=1:dd
                    W = M(m).Weights(d);
                    for r=1:M(m).rank
                        idx = (1:ss(d)) + (r-1)*ss(d) + rr*sum(ss(1:d-1)) + midx; % index of regressors in design matrix
                        W.PosteriorStd(r,:) = all_se(idx);
                        W.T(r,:) = all_T(idx);
                        W.p(r,:) = all_p(idx);
                        W.PosteriorCov(:,:,r) = Sfit.covb(idx,idx);
                        if any(strcmp(W.type, {'continuous','periodic'}))
                            W.invHinvK(:,:,r) = invHinvK(idx,idx);
                        end
                    end
                    M(m).Weights(d) = W;
                end


                %                 idx = midx + (1:rr*sum(ss)); % index of regressors for this module
                %                 % convert to cells
                %                 M(m).se = num2dimcell(all_se(idx),ss,rr);
                %                 M(m).T = num2dimcell(all_T(idx),ss,rr);
                %                 M(m).p = num2dimcell(all_p(idx),ss,rr);
                %
                midx = midx + sum(ss) * rr; % jump index by number of components in module

            end

            Sfit.exitflag = Sfit.exitflag;
            Sfit.exitflag_allstarting = Sfit.exitflag_allstarting;

            %% log-likelihood and approximation to log-evidence

            % LLH at inferred params
            [obj,Sfit.LogLikelihood] = LogLikelihood(obj);
            obj = predictor_variance(obj);

            % number of free parameters
            nFreePar = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));

            %  nFreePar = [M.nFreeParameters]; %cellfun(@(x) sum(x(:)), M);% number of free parameters for each component
            %  nFreePar = sum(nFreePar);
            obj.score.nFreeParameters = nFreePar;

            % model evidence using Laplace approximation (Bishop - eq 4.137)  -
            % requires that a prior has been defined - Eq 16
            LD = logdet(B);
            %S.LogEvidence = S.LogLikelihood + nfreepar/2*log(2*pi) - logdet/2;
            Sfit.LogEvidence = Sfit.LogLikelihood - LD/2;

            PP = projection_matrix(M);
            for m=1:nM % add part from prior
                for d=1:M(m).nDim
                    for r=1:M(m).rank % assign new set of weight to each component
                        dif =  (M(m).Weights(d).PosteriorMean(r,:) - M(m).Prior(r,d).PriorMean)*PP{m}{r,d}'; % distance from prior mean (projected)
                        this_cov = PP{m}{r,d} * M(m).Prior(r,d).PriorCovariance * PP{m}{r,d}'; % corresponding covariance prior
                        % logdet = 2*sum(log(diag(chol(PP{m}{d}*this_cov*PP{m}{d}')))); % log-determinant of prior covariance (fast reliable way)
                        % S.LogEvidence = S.LogEvidence -  (dif/this_cov)*dif'/2  - logdet/2 -M(m).nfreepar(1,d)/2*log(2*pi); % log-prior for this weight
                        inf_var = isinf(diag(this_cov)); % do not include weights with infinite prior variance
                        if any(~inf_var)
                            dif = dif(~inf_var);
                            Sfit.LogEvidence = Sfit.LogEvidence -  (dif/this_cov(~inf_var,~inf_var))*dif'/2; % log-prior for this weight
                        end
                    end
                end
            end

            Sfit.BIC = nFreePar*log(obj.nObs) -2*Sfit.LogLikelihood; % Bayes Information Criterion
            Sfit.AIC = 2*nFreePar - 2*Sfit.LogLikelihood; % Akaike Information Criterior
            Sfit.AICc = Sfit.AIC + 2*nFreePar*(nFreePar+1)/(obj.nObs-nFreePar-1); % AIC corrected for sample size
            Sfit.LogJoint_allstarting = Sfit.LogJoint_allstarting; % values for all starting points

            % compute r2
            if strcmp(obj.obs, 'normal')
                if isempty(obj.ObservationWeight)
                    SSE = sum((obj.T-obj.Predictions.rho).^2); % residual error
                    SST = (obj.nObs-1)*var(obj.T); % total variance
                else
                    SSE = sum(obj.ObservationWeight .* (obj.T-obj.Predictions.rho).^2);
                    SST = (sum(obj.ObservationWeight)-1) * var(obj.T, obj.ObservationWeight);
                end
                Sfit.r2 = 1- SSE/SST; % r-squared
            end

            % model evidence using Laplace approximation (Bishop - eq 4.137)
            % S.LLH_model = S.LLH - U'*sigma1*U/2 - V'*sigma2*V/2 + (m1+m2)/2*log(2*pi) - log(det(Inff))/2;


            %   for projection on space, convert weights back to
            %   original domain
            if ~isfield(param, 'originalspace') || param.originalspace
                M = project_from_basis(M);
            end

            obj.regressor = M;

            %  obj = obj.clear_data;

            score_fields = fieldnames(Sfit);
            for f=1:length(score_fields)
                obj.score.(score_fields{f}) = Sfit.(score_fields{f});
            end

            obj.score.FittingTime = toc;
        end

        %% BOOTSTRAPPING
        function [obj, bootsam] = boostrapping(obj, nBootstrap, varargin)
            % M = M.boostrapping(nBootstrap) performs bootstrapping to
            % approximate the posterior distribution of weights.
            % nBootstrap is the number of boostraps used (by default: 100).
            % Set of weights for each bootstrap and confidence intervals are added as fields 'U_boostrap' and 'ci_boostrap' in weights  structure
            %
            % M.boostrapping(nBootstrap, Name, Value) specifies options using one or more name-value arguments.
            % Names can be the following:
            % - 'alpha': Significance level for confidence interval, scalar between 0 and 1 (default: 0.05).
            % Computes the 100*(1-Alpha) bootstrap confidence interval of each weight.
            % - 'type': confidence interval type, can be set to 'per' (percentile method, default) or 'norm' (fits normal posterior)
            % -'verbose': if Value (boolean) is set to true (default),
            % an '*' on the command line indicates every new bootstrap completed.
            %
            % [M, bootsam] = M.boostrapping(nBootstrap) provide the bootstrap sample indices,
            % returned as an n-by-nBootstrap numeric matrix, where n is the number of observation values in M.
            %
            % Note: if multiple GUM models with same number of observations are provided as input, the same bootstraps are used for all models
            %
            % See also bootci, bootstrp
            assert(isscalar(nBootstrap) && nBootstrap>0, 'nBootstrap should be a positive scalar');

            verbose = true;
            alpha = 0.05;
            type = 'norm';

            assert(mod(length(varargin),2)==0, 'Name-Value arguments should be provided in pairs');
            for v=1:2:length(varargin)
                switch lower(varargin{v})
                    case 'alpha'
                        alpha = varargin{v+1};
                        assert(isscalar(alpha) && alpha>=0 && alpha<=1,...
                            'alpha must be a scalar between 0 and 1');
                    case 'type'
                        type = varargin{v+1};
                        assert(isstring(type) && (strcmp(type,'norm')||strcmp(type,'per')),...
                            'value for type is either ''per'' or ''norm''');
                    case 'verbose'
                        verbose = varargin{v+1};
                end

                n = obj(1).nObs;
                if length(obj)>1
                    assert([obj.nObs]==n, 'all models must have the same number of observations');
                end

                if verbose
                    fprintf('Computing %d boostraps for %d models: ',nBootstrap, length(obj));
                end

                if nargout>1
                    bootsam = zeros(n, nBootstrap);
                end

                % pre-allocate boostrap weights
                U_bt = cell(1,numel(obj));
                for i=1:numel(obj)
                    U_bt{i} = zeros(n,obj(i).score.nParameters);
                end

                for p=1:nBootstrap % for each permutation

                    bt_set = randi(n,1,n); % generating boostrap (sampling with replacement)
                    if nargout>1
                        bootsam(:,p) = bt_set;
                    end

                    %fit on bootstrap data set
                    for i=1:numel(obj)
                        obj_bt = IRLS(extract_observations(obj(i),bt_set));

                        Ucat = concatenate_weights(obj_bt);
                        U_bt{i}(:,p) = Ucat;
                    end


                    if verbose
                        fprintf('*');
                    end

                end

                % compute confidence intervals
                ci_bt = cell(size(obj));
                for i=1:numel(obj)
                    if strcmpi(type,'norm')
                        %!!! check this is correct (ref in doc bootci)
                        U_mean = mean(U_bt{i},2); % mean and std over bootstraps
                        U_std = std(U_bt{i},[],2);
                        ci_bt{i} = U_mean + norminv(alpha/2)*U_std*[1 -1];
                    else
                        ci_bt{i} = quantile(U_bt{i}', [alpha/2 1-alpha/2])'; % compute quantiles on boostrap samples
                    end

                end

                % add bootstrap weights and C.I. to weight structures
                for i=numel(obj)
                    obj(i).regressor = set_weights(obj(i).regressor,U_bt{i},[],'U_bootstrap');
                    obj(i).regressor = set_weights(obj(i).regressor,ci_bt{i},[],'ci_boostrap');

                end

                if verbose
                    fprintf('\n done!\n');
                end

            end
        end

        %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% IMPLEMENT THE LOG-JOINT MAXIMIZATION ALGORITHM (modified IRLS)
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function  obj = IRLS(obj)

            nM = obj.nMod; % number of modules
            n = obj.nObs; % number of observations
            M = obj.regressor;
            nD = [M.nDim]; % number of dimension for each module
            D = max([M.nFreeDimensions]);

            if isempty(obj.mixture)
                nC = 1; % number of components
                idxComponent = ones(1,nM); % indices of component
            else
                nC = obj.mixture.nComponent;
                idxComponent = obj.mixture.idxComponent;
            end

            %stepsize = 1; % should always stay 1 unless LLH decreases after some step(s)

            % logprior = cell(1,nM);
            rank = zeros(1,nM);
            for m=1:nM
                rank(m) = M(m).rank;
                % logprior{m} = zeros(rank(m),nD(m));
            end

            % order of updates for each dimensions
            UpdOrder = UpdateOrder(M);

            PP = projection_matrix(M); % free-to-full matrix conversion

            % projection matrix and prior covariance for set of weights
            P = cell(nC,D);
            Lambda = cell(nC,D);
            PrC = prior_covariance_cell(M);
            for cc = 1:nC
                P(cc,:) = blkdiag_subset(PP(idxComponent==cc), UpdOrder(idxComponent==cc,:)); % concatenate projection matrix corresponding to different modules for each dimension
                Lambda(cc,:) = blkdiag_subset(PrC(idxComponent==cc), UpdOrder(idxComponent==cc,:)); % group covariance corresponding to different modules for each dimension (full parameter space)
            end

            % prior mean for each set of weight
            priormean = prior_mean_cell(M);
            mu = cell(1,D);
            for d=1:D
                this_d = UpdOrder(:,d); % over which dimension we work for each model
                for m=1:nM
                    mu{d} = [mu{d} priormean{m}{:,this_d(m)}]; % add mean values of prior
                end
            end

            % we use sparse coding if any of the data array is sparse
            %SpCode = any(cellfun(@issparse, {M.Data}));
            SpCode = false(1,nC);
            for cc=1:nC
                SpCode(cc) = any(cellfun(@issparse, {M(idxComponent==cc).Data}));

            end

            initialpoints = obj.param.initialpoints;
            maxiter = obj.param.maxiter;
            miniter = obj.param.miniter;

            % if frequentist linear regression: converges to analytical solution in one update
            simple_linear_regression =  isempty(obj.mixture) && strcmp(obj.obs,'normal') && obj.isgam && all(isinf(diag(global_prior_covariance(M))));
            if simple_linear_regression
                maxiter = 1;
                miniter = 0;
            end
            TolFun = obj.param.TolFun;

            %  U_allstarting = zeros(obj.score.nParameters, initialpoints); % estimated weights for all starting points
            U_allstarting = zeros(length(concatenate_weights(M)), initialpoints); % estimated weights for all starting points


            logjoint_allstarting = zeros(1,initialpoints); % log-joint for all starting points
            exitflag_allstarting = zeros(1,initialpoints); % exitflag for all starting points
            logjoint_hist_allstarting = cell(1,initialpoints); % history log-joint for all starting points
            all_scaling = zeros(1,initialpoints);

            % number of regressors for each dimension
            nReg = zeros(1,D);
            nFree = zeros(1,D); % number of free parameters
            size_mod = zeros(nM,D); % size over each dimension for each module
            precision = cell(nC,D); % for infinite variance

            % project prior mean and covariance onto free space
            KP = cell(nC,D);
            nu = cell(nC,D); % projected prior mean
            for d=1:D
                nFree(d) = size(P{cc,d},1); % number of free parameters for this set of weight
                this_d = UpdOrder(:,d)'; %min(d,nDim); % over which dimension we work for each model
                for m=1:nM
                    size_mod(m,d) = M(m).Weights(this_d(m)).nWeight;
                    nReg(d) = nReg(d) + rank(m)*size_mod(m,d); % add number of regressors for this module
                end

                for cc=1:nC
                    %idxC = idxComponent==cc;
                    Lambda{d} = P{cc,d}*Lambda{cc,d}*P{cc,d}'; % project onto free basis

                    Lambda{cc,d} = force_definite_positive(Lambda{cc,d}); % make sure it is definite positive

                    if any(isinf(Lambda{cc,d}(:))) % use precision only if there is infinite covariance (e.g. no regularization)
                        %  wrn = warning('off', 'MATLAB:singularMatrix');
                        nonzeroprec = ~isinf(diag(Lambda{cc,d})); % weights with nonzero precision (i.e. finite variance)
                        precision{cc,d} = zeros(size(Lambda{cc,d})); %
                        precision{cc,d}(nonzeroprec,nonzeroprec) = inv(Lambda{cc,d}(nonzeroprec,nonzeroprec));
                        %  warning(wrn.state, 'MATLAB:singularMatrix');
                    elseif SpCode(cc)
                        KP{cc,d} = Lambda{cc,d}* P{cc,d}; % we're going to need it several times
                    end

                    nu{cc,d} = P{cc,d}*mu{cc,d}';
                end
            end

            % in case we may use Newton update on full dataset
            if D>1
                if nC>1
                    error('not coded yet');
                end

                % we create dummy 'regressor' variables to store the gradients and constant weights
                Udc_dummy = clear_data(M);
                B_dummy = repmat({{}},1, nM);

                % full prior covariance matrix
                Kall = global_prior_covariance(M);
                Pall = projection_matrix(M,'all'); % full-to-free basis

                % project onto free basis
                if issparse(Kall) && issparse(Pall)
                    Kall =  Pall*Kall*Pall'; % not even sure it wouldn't faster if we convert to full
                else
                    Pall = full(Pall); % in case Pall is sparse, faster this way
                    Kall =  Pall*Kall*Pall';
                end

                % use precision only if there is infinite covariance (e.g. no regularization)
                if any(isinf(Kall(:)))
                    nonzeroprec = ~isinf(diag(Kall)); % weights with nonzero precision (i.e. finite variance)
                    precision_all = zeros(size(Kall)); %
                    precision_all(nonzeroprec,nonzeroprec) = inv(Kall(nonzeroprec,nonzeroprec));
                end
                nFree_all = size(Kall,1);
            end


            %% repeat algorithn with each set of initial points
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
                [~,logprior] = LogPrior(obj);


                % initial value of log-joint
                obj = Predictor(obj);
                obj = LogJoint(obj,[],true);
                logjoint = obj.score.LogJoint;

                iter = 1;
                not_converged = true;
                logjoint_hist = zeros(1,maxiter);
                Uall = nan(size(concatenate_weights(M)));
                dLJ = nan;
                prev_dLJ = nan;
                dU = nan(size(concatenate_weights(M)));
                prev_dU = dU;

                % thresholds for switching to full hessian
                % update (resp. on cosine of successive
                % updates, consistency LB and UB
                thr_FHU = [.9 .8 1.2];
                c_fpd = 0; % minimal eigenvalue for forcing def positive full hessian matrix
                % (if too small, may not work; if too large, converging may take longer)

                % weighting by observations weights
                if isempty(obj.ObservationWeight) && nC==1
                    weighted_fun = @(x,cc) x;
                elseif nC ==1
                    weighted_fun = @(x,cc) x.*obj.ObservationWeight;
                elseif isempty(obj.ObservationWeight)
                    weighted_fun = @(x,cc) x.*obj.mixture.Posterior(:,cc);
                    warning('move down??');
                else
                    weighted_fun = @(x,cc) x.*obj.ObservationWeight .*obj.mixture.Posterior(:,cc);
                end
                weighted_sum = @(x,cc) sum(weighted_fun(x,cc));
                unbiased_mse = @(x,cc) weighted_sum(x.^2,cc)/obj.score.df; % unbiased mean squared error

                % dispersion parameter from exponential family
                FixedDispersion = ~strcmp(obj.obs, 'normal');
                if ~FixedDispersion
                    s = unbiased_mse(obj.T - weighted_sum(obj.T)/weighted_sum(ones(obj.nObs,1))); % variance of centered data
                else
                    s = 1;
                end
                obj.score.scaling = s;

                FullHessianUpdate = false(1,nC);

                %% loop weight update until convergence
                while not_converged

                    old_logjoint = logjoint;

                    for c=1:nC
                        idxC = idxComponent==cc;

                        for d=1:D % for each dimension

                            this_d = UpdOrder(:,d)'; % over which dimension we work for each model

                            % concatenate set of weights for this dimension for each component and each module
                            UU = concatenate_weights(M(idxC), this_d(idxC));

                            % design matrix (project tensors in all expcept
                            % this dimension)
                            Phi = design_matrix(M(idxC),[], this_d(idxC), 0);

                            if iter==1 && d==1 && cc==1 && any(isnan(UU)) && ~strcmp(obj.obs, 'normal')
                                % make sure we stay in "safe" region of weight space
                                switch obj.link
                                    case 'log'
                                        this_rho = log(obj.T+.25);
                                    case {'logit','probit'}
                                        this_rho = log((obj.T+.5)./(1.5-obj.T));
                                        %  case 'normal'
                                        %      obj.Predictions.rho = obj.T/2;
                                end
                            elseif d==1 || ~FullHessianUpdate(c)
                                this_rho = Phi*UU'; % predictor
                            end
                            obj.Predictions.rho(:,cc) = this_rho;

                            switch obj.link
                                case 'logit'
                                    Y = 1 ./ (1 + exp(-this_rho)); % expected mean
                                    R = Y .* (1-Y) ; % derivative wr.t. predictor
                                case 'log'
                                    Y = exp(this_rho); % rate
                                    R = Y;
                                    R = min(R,1e10); % to avoid badly scaled hessian
                                case 'identity'
                                    Y = this_rho;
                                    R = ones(obj.nObs,1);
                                case 'probit'
                                    Y = normcdf(this_rho);
                                    sgn = sign(obj.T-0.5); % convert to -1/+1
                                    error('');
                                    R =  sgn .*normpdf(this_rho) ./ Y;
                            end

                            % remove constant parts from projected activation
                            [rho_tilde, UconstU] = remove_constrained_from_predictor(M(idxC), this_d(idxC), this_rho, Phi, UU);

                            % compute gradient
                            if ~FullHessianUpdate(c)
                                G = weighted_fun(R .* rho_tilde + (obj.T-Y),cc); % inside equation 12
                            else
                                G =  weighted_fun(obj.T-Y,cc);
                            end
                            Rmat = spdiags(weighted_fun(R,cc), 0, n, n);

                            if ~SpCode
                                Psi = Phi*P{cc,d}';
                            end

                            if FullHessianUpdate(c)
                                inf_cov = any(isinf(Kall(:)));
                            else
                                inf_cov = any(isinf(Lambda{cc,d}(:)));
                            end
                            if ~inf_cov %finite covariance matrix
                                if SpCode % for sparse matrix, more efficient this way
                                    B = KP{cc,d}*(Phi'*G) + s*nu{cc,d};
                                else
                                    B = Lambda{cc,d}*(Psi'*G) + s*nu{cc,d};
                                end
                            else % any infinite covariance matrix (e.g. no prior on a weight)
                                if SpCode
                                    B = P{cc,d}*(Phi'*G) + s*precision{cc,d}*nu{cc,d};
                                else
                                    B = Psi'*G + s*precision{cc,d}*nu{cc,d};
                                end
                            end


                            if ~FullHessianUpdate(c) %% update weights just along that dimension

                                % Hessian matrix on the free basis (eq. 12)

                                if ~inf_cov %finite covariance matrix
                                    if SpCode
                                        H = KP{cc,d}*(Phi'*Rmat*Phi)*P{cc,d}' + s*eye(nFree(cc,d)); % should be faster this way
                                    else
                                        H = Lambda{cc,d}* Psi'*Rmat*Psi + s*eye(nFree(cc,d)); % Hessian matrix on the free basis (equation 12)
                                    end

                                else % any infinite covariance matrix (e.g. no prior on a weight)
                                    if SpCode
                                        H = P{cc,d}*(Phi'*Rmat*Phi)*P{cc,d}' + s*precision{cc,d};
                                    else
                                        H = Psi'*Rmat*Psi + s*precision{cc,d};
                                    end
                                end

                                % new set of weights eq.12 (projected back to full basis)
                                xi = (H\B)' * P{cc,d};

                                %  while strcmp(obs,'poisson') && any(Phi*(Unu+[repelem(U_const(1:rank),m(d)) zeros(1,m(D+1))])'>500) % avoid jump in parameters that lead to Inf predicted rate

                                % add new set of weights to regressor object
                                obj.regressor(idxC) = set_weights(M(idxC), UU, this_d(idxC));
                                for m= find(idxC)
                                    d2 = this_d(m);
                                    logprior{m}(:,d2) = LogPrior(obj.regressor(m),d2); % log-prior for this weight
                                end

                                if ~FixedDispersion
                                    s(cc) = unbiased_mse(obj.T - obj.Predictions.rho(:,cc)); %  scaling parameter: unbiased mse
                                    obj.score.scaling = s;
                                end

                                % compute log-joint
                                interim_logjoint = logjoint;
                                [obj,logjoint] = LogJoint(obj,logprior, true);
                                %if logjoint<interim_logjoint-1e-6
                                %      2; % debug
                                %  end


                                %% end check


                                % step halving if required (see glm2: Fitting Generalized Linear Models
                                %    with Convergence Problems - Ian Marschner)
                                obj.Predictions.rho(:,cc) = Phi*(xi'+UconstU);
                                %  while LogJoint(rho, T, w, obs,prior_score)<old_logjoint %% avoid jumps that decrease the log-joint
                                %      Unu = (UU+Unu)/2;  %reduce step by half
                                %      rho = Phi*(Unu'+UconstU);
                                %  end
                                while iter<4 && strcmp(obj.link,'log') && any(abs(obj.Predictions.rho(:,cc))>500) % avoid jump in parameters that lead to Inf predicted rate
                                    xi = (UU+xi)/2;  %reduce step by half
                                    obj.Predictions.rho(:,cc) = Phi*(xi'+UconstU);
                                end


                                compute_logjoint = true;


                                cnt = 0;
                                diverged = false;

                                while compute_logjoint
                                    obj.regressor(idxC) = set_weights(M(idxC),xi+UconstU', this_d(idxC));

                                    for m=find(idxC)
                                        d2 = this_d(m);
                                        logprior{m}(:,d2) = LogPrior(obj.regressor(m),d2); % log-prior for this weight
                                    end

                                    if ~FixedDispersion
                                        s(cc) = unbiased_mse(obj.T - obj.Predictions.rho(:,cc)); %  scaling parameter: unbiased mse
                                        obj.score.scaling = s;
                                    end

                                    % compute log-joint
                                    [obj,logjoint] = LogJoint(obj,logprior, true);

                                    compute_logjoint = (logjoint<interim_logjoint-1e-3);
                                    if compute_logjoint % if log-joint decreases,
                                        xi = (UU-UconstU'+xi)/2;  %reduce step by half
                                        obj.Predictions.rho(:,cc) = Phi*(xi'+UconstU);

                                        cnt = cnt+1;
                                        ljc(cnt) = logjoint;
                                        if cnt>100 %% if we have halved 100 times step to new weights, we probably have diverged
                                            diverged = true;
                                            break;
                                        end
                                    end
                                end
                                halving_counts(d, iter) = cnt; %% not used - just for diagnostics
                                %if cnt>10
                                %    3;
                                %end

                                M(idxC) = set_weights(M(idxC), xi+UconstU', this_d((idxC)));
                                %  M = obj.regressor;
                            else
                                %% prepare for Newton step in full weight space
                                B_dummy(idxC) = set_free_weights(Udc_dummy(idxC),B', B_dummy(idxC), this_d(idxC));
                                Udc_dummy(idxC) = set_weights(Udc_dummy(idxC), UconstU', this_d(idxC));

                            end
                        end

                        %% DIRECT NEWTON STEP ON WHOLE PARAMETER SPACE
                        if FullHessianUpdate(c)

                            UU = concatenate_weights(M(idxC)); % old set of weights
                            UconstU = concatenate_weights(Udc_dummy(idxC)); % fixed weights
                            % B = concatenate_weights(B_dummy); % gradient over all varialves
                            B = cellfun(@(x) [x{:}], B_dummy(idxC),'unif',0);
                            B = [B{:}]';

                            % compute full Hessian of Log-Likelihood (in unconstrained space)
                            Hess = Hessian(obj, c_fpd);

                            % compute hessian matrix of log-joint on the unconstrained basis
                            if ~inf_cov %finite covariance matrix
                                B = B + Kall*(Hess*(Pall*UU'));  %% !!! check if projections are well done!!!

                                H = Kall* Hess + s*eye(nFree_all(cc)); % K * Hessian matrix on the free basis

                            else % any infinite covariance matrix (e.g. no prior on a weight)
                                B = B + Hess*(Pall*UU');  %% !!! check if projections are well done!!!
                                %!!!!%%
                                H = Hess + s*precision_all; % Hessian matrix on the free basis
                            end

                            xi = (H\B)' * Pall; % new set of weights (projected back to full basis)

                            obj.regressor(idxC) = set_weights(M(idxC),xi+UconstU);
                            obj = Predictor(obj); % compute rho
                            if ~FixedDispersion
                                s(cc) = unbiased_mse(obj.T - obj.Predictions.rho(:,cc)); %  update scaling parameter: unbiased mse
                                obj.score.scaling = s;
                            end
                            [obj,logprior] = LogPrior(obj); % compute log-prior
                            [obj,logjoint] = LogJoint(obj,[],true);


                            if (logjoint<old_logjoint-1e-3) % full step didn't work: go back to previous weights and run
                                FullHessianUpdate = 0;
                                obj.regressor(idxC) = M(idxC);
                                obj = Predictor(obj); % compute rho
                                %!! change to function for updating dispersion parameter
                                if ~FixedDispersion
                                    s(cc) = unbiased_mse(obj.T - obj.Predictions.rho(cc)); %  update scaling parameter: unbiased mse
                                    obj.score.scaling = s;
                                end
                                [obj,logprior] = LogPrior(obj); % compute log-prior
                                [obj,logjoint] = LogJoint(obj,[],true);

                                old_logjoint = logjoint - 2*TolFun; % just to make sure it's not flagged as 'converged'

                                % change thresholds for switching to full
                                % hessian update to make it less likely
                                %thr_FHU = .5 + thr_FHU/2; % twice as close to 1
                                if c_fpd>0
                                    c_fpd = 10*c_fpd; % increase chances that Hessian will really be def positive
                                else
                                    c_fpd = 1e3*eps;
                                end

                            else
                                M(idxC) = obj.regressor(idxC);
                            end
                        end

                        %  Eq 13: update scales (to avoid slow convergence) (only where there is
                        % more than one component without constraint
                        % inforce constraint during optimization )
                        % if iter<5
                        recompute = false;
                        for m=find(idxComponent==cc & nD>1)
                            ct = [M(m).Weights.constraint];
                            for r=1:rank(m)
                                alpha = zeros(1,nD(m));
                                free_weights = ct(r,:)=='f';
                                n_freeweights = sum(free_weights);
                                if n_freeweights>1
                                    this_nLP = -logprior{m}(r,free_weights);
                                    mult_prior_score = prod(this_nLP )^(1/n_freeweights);
                                    if mult_prior_score>0
                                        alpha(free_weights) = sqrt( mult_prior_score./ this_nLP); % optimal scaling factor

                                        for d=find(free_weights)
                                            obj.regressor(m).Weights(d).PosteriorMean(r,:) = obj.regressor(m).Weights(d).PosteriorMean(r,:) * alpha(d);
                                            logprior{m}(r,d) = logprior{m}(r,d)*alpha(d)^2;
                                        end
                                        recompute = true;
                                    end
                                end
                            end
                        end

                        %% compute predictor
                        if recompute
                            obj(idxC) = Predictor(obj(idxC));
                            if ~FixedDispersion
                                s = unbiased_mse(obj.T - obj.Predictions.rho(:,cc)); %  update scaling parameter: unbiased mse
                                obj.score.scaling = s;
                            end

                            % compute log-joint
                            [obj,logjoint] = LogJoint(obj,logprior);

                            M = obj.regressor;

                        end
                    end

                    if any(~isreal(Uall))
                        error('imaginary weights...something went wrong')
                    end

                    prev_dLJ = dLJ;
                    dLJ = logjoint - old_logjoint;
                    if D>1

                        prev_Uall = Uall;
                        prev_dU = dU;
                        Uall = concatenate_weights(M);
                        dU = Uall - prev_Uall; % weight updates
                        cos_successive_updates(iter) = dot(prev_dU,dU)/(norm(prev_dU)*norm(dU));
                        rat_successive_updates(iter) = norm(dU)^2/norm(prev_dU)^2;
                        dU_hist(:,iter) = dU;
                        rat_successive_logjoint(iter) = dLJ / prev_dLJ;
                        consistency = rat_successive_logjoint(iter) /  rat_successive_updates(iter);

                        % move to full hessian update if there are signs that
                        % has fallen into convex region of parameter space
                        if ~FullHessianUpdate(c) && ...
                                cos_successive_updates(iter)>thr_FHU(1) && consistency>thr_FHU(2) && consistency<thr_FHU(3)


                            FullHessianUpdate(c) = true;
                        end
                        FHU(:,iter) = FullHessianUpdate(c);
                    end

                    if strcmp(obj.param.verbose, 'on')
                        fprintf(DisplayChar(iter, any(FullHessianUpdate)));
                    elseif strcmp(obj.param.verbose, 'full')
                        fprintf('iter %d, log-joint: %f\n',iter,logjoint);
                    end

                    logjoint_hist(iter) = logjoint;
                    iter = iter+1;
                    tolfun_negiter = 1e-4; % how much we can tolerate log-joint to decrease due to numerical inaccuracies
                    not_converged = ~diverged && (iter<=maxiter) && (iter<miniter || (dLJ>TolFun) || (dLJ<-tolfun_negiter) ); % || (LLH<oldLLH))
                end %% end of algorithm iteration loop


                logjoint_hist(iter:end) = [];

                %% compute variance of each order components and sort
                obj = predictor_variance(obj);
                % var_c = cell(1,nM);
                for m=1:nM
                    %                     var_c{m} = zeros(1,rank(m));
                    %                     if all(M(m).nWeight>0)
                    %                         for r=1:rank(m)
                    %                             %    var_c{mm}(r) = var(projdim(X,U,r,spd,zeros(1,0))); % variance of overall activation for each component
                    %                             var_c{m}(r) = var(ProjectDimension(M(m),r,zeros(1,0))); % variance of overall activation for each component
                    %                         end
                    %                         if M(m).ordercomponent && rank(m)>1
                    %                             [var_c{m},order] = sort(var_c{m},'descend'); % sort variances by descending order
                    %                             M(m).U = M(m).U(order,:); % reorder
                    %                         end
                    %                     end


                    if all([M(m).Weights.nWeight]>0) && M(m).ordercomponent && rank(m)>1
                        [~,order] = sort(obj.score.PredictorVariance(:,m),'descend'); % sort variances by descending order
                        for d=1:M(m).nDim
                            M(m).Weights(d).PosteriorMean = M(m).Weights(d).PosteriorMean(order,:); % reorder
                        end
                        obj.score.PredictorVariance(:,m) = obj.score.PredictorVariance(order,m);
                    end
                end

                obj.score.scaling = s; %  scaling parameter: unbiased mse

                extflg = (logjoint>old_logjoint-tolfun_negiter) && (logjoint - old_logjoint < TolFun); %)
                extflg = simple_linear_regression || extflg;
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
                        fprintf(msg, iter-1, logjoint);
                    case 'little'
                        fprintf(DisplayChar(ip));
                end

                %% store weights, log-joint and exitflag for this starting point
                U_allstarting(:,ip) = concatenate_weights(M);
                logjoint_allstarting(ip) = logjoint;
                logjoint_hist_allstarting{ip} = logjoint_hist;
                exitflag_allstarting(ip) = extflg;
                all_scaling(ip) = s;
                obj.score.exitflag = extflg;
            end


            if strcmp(obj.param.verbose, 'little')
                fprintf('\n');
            end

            %% find starting point with highest log-joint
            if initialpoints>1
                [logjoint, ip] = max( logjoint_allstarting);
                obj.score.LogJoint = logjoint;
                obj.score.scaling = all_scaling(ip);
                obj.score.exitflag = exitflag_allstarting(ip);
                M = set_weights(M, U_allstarting(:,ip));
                M = set_weights(M, U_allstarting, [], 'U_allstarting');

                obj.regressor = M;

            end

            % S = obj.score;
            %  S.Logjoint = logjoint;
            obj.score.LogJoint_allstarting = logjoint_allstarting;
            obj.score.LogJoint_hist_allstarting = logjoint_hist_allstarting;
            obj.score.exitflag_allstarting = exitflag_allstarting;

            obj.score.scaling = obj.score.scaling;
            % obj.score = S;

        end

        %% sample weights from gaussian prior
        function M = sample_weights_from_prior(M)
            for m=1:length(M)
                M(m).sample_weights_from_prior;
            end
        end



        %% COMPUTE PREDICTOR RHO
        function  [obj,rho] = Predictor(obj,idxC)
            %obj = Predictor(obj)
            % computes predictor rho.
            %
            % [obj,rho] = Predictor(obj)
            %
            %Predictor(obj, cc) only for component cc (for mixture models)
            if isempty(obj.mixture)
                nC = 1;
                idxComponent = ones(1,obj.nMod);
            else
                nC = obj.mixture.nComponent;
                idxComponent = obj.mixture.idxComponent;
            end
            if nargin<2
                idxC = 1:nC;
            end

            if ~isempty(obj.Predictions.rho)
                rho =obj.Predictions.rho;
                rho(:,idxC) = 0;
            else
                rho = zeros(obj.nObs,nC);
            end

            for cc=idxC % for each mixture component
                for m= find(idxComponent==cc)
                    rho(:,cc) = rho(:,cc) + Predictor(obj.regressor(m));
                end
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
                nReg = nReg + M(m).rank*M(m).Weights(D(m)).nWeight; % add number of regressors for this module
            end
        end

        %% TEST IF MODEL IS A GAM (dimension=1)
        function bool = isgam(obj)
            % bool = isgam(M)
            % tests if model M is a GAM (generalized additive model), i.e.
            % if all regressors are one-dimensional (regressors can be of
            % any type, continuous, linear, categorical, etc.)

            % recursive call if more than one model
            if size(obj)>1
                bool = false(size(obj));
                for i=1:numel(obj)
                    bool(i) = isgam(obj(i));
                end
                return;
            end

            bool = all([obj.regressor.nFreeDimensions]<=1);
        end

        %% TEST IF MODEL IS A GLM (dimension=1 & linear or categorical regressors)
        function bool = isglm(obj)
            % bool = isglm(M)
            % tests if model M is a GLM (generalized linear model), i.e.
            % if all regressors are one-dimensional and linear or
            % categorical

            % recursive call if more than one model
            if size(obj)>1
                bool = false(size(obj));
                for i=1:numel(obj)
                    bool(i) = isgam(obj(i));
                end
                return;
            end


            bool = all([obj.regressor.nFreeDimensions]<=1);
            if bool
                W = [obj.regressor.Weights];
                Wtype = {W.type};
                bool = bool & all(strcmp(Wtype,'linear') | strcmp(Wtype,'categorical') | strcmp(Wtype,'constant'));
            end
        end

        %% TEST IF MODEL WEIGHTS HAVE BEEN ESTIMATED (OR SET)
        function bool = isestimated(obj)
            % bool = isestimated(M)
            % tests if weights in model M have been estimated (or assigned)

            bool = false(size(obj));
            for m=1:numel(obj)
                bool(m) = ~isempty(obj(m).concatenate_weights());
            end
        end

        %% TEST IF ANY REGRESSOR HAS INFINITE COVARIANCE
        function bool = is_infinite_covariance(obj)
            % bool = is_infinite_covariance(M) tests if any regressor in
            % model M has infinite prior covariance

            %  K = {obj.regressor.sigma};
            K = prior_covariance_cell(obj.regressor, true); % group prior covariacne from all modules
            bool = any(cellfun(@(x) any(isinf(x(:))), K));

        end

        %% REMOVE DATA (to make it lighter after fitting)
        function   obj = clear_data(obj)
            % M = clear_data(M)
            % removes data (observations and regressor values) to make it lighter
            %warning: data is definitely lost!

            obj.T = [];
            obj.ObservationWeight = [];

            for m=1:obj.nMod

                % remove data field from output
                obj.regressor(m) = clear_data(obj.regressor(m));
            end
        end

        %% SAVE MODEL TO FILE
        function save(obj, filename)
            %  save(M, filename)
            % saves model M to -mat file 'filename'
            if ~ischar(filename)
                error('filename should be a string array');
            end
            try
                save(filename, 'obj');
            catch % sometimes it fails using standard format, so need to do it using Version 7.3 format
                save(filename, 'obj', '-v7.3');
            end
        end

        %% LINK FUNCTION
        function f = inverse_link_function(obj,rho)
            % f = inverse_link_function(M) returns handle to inverse link
            % function.
            % a = inverse_link_function(M, rho) returns values of inverse
            % link function evaluated at points rho
            switch obj.link
                case 'identity'
                    f = @(x) x;
                case 'logit'
                    f = @(x) 1./(1+exp(-x));
                case 'log'
                    f = @exp;
                case 'probit'
                    f = @normcdf;
            end

            if nargin>1 % evaluate at data points
                f = f(rho);
            end
        end

        %% COMPUTE EXPECTED VALUE
        function [obj,Y,R] = ExpectedValue(obj)
            % obj = ExpectedValue(obj)
            % computes expected value Y (i.e. predictor passed through inverse
            % link function)
            %
            % [obj,Y] = ExpectedValue(obj)
            %
            % [obj,Y,R] = ExpectedValue(obj)
            % R is the derivative w.r.t predictor (to compute Hessian and IRLS)
            if isempty(obj.Predictions.rho)
                [obj,rho] = Predictor(obj);
            else
                rho = obj.Predictions.rho;
            end

            % pass predictor through inverse link function
            Y = obj.inverse_link_function(rho);

            if nargout>2

                switch obj.link
                    case 'logit'
                        %  Y = 1 ./ (1 + exp(-rho)); % predicted probability
                        R = Y .* (1-Y) ; % derivative wr.t. predictor

                    case 'probit'
                        %  Y = normcdf(rho);
                        %  if nargout>2
                        % see e.g. Rasmussen 3.16
                        n = normpdf(rho);
                        Ysgn = Y;
                        Ysgn(obj.T==0) = 1-Y(obj.T==0);
                        sgn = sign(obj.T-.5); % convert to -1/+1
                        R = (n./Ysgn)^2 + sgn.*rho.*n./Ysgn;
                        % end

                    case 'log'
                        % Y = exp(rho);
                        % if nargout>2
                        R = Y;
                        %end
                    case 'identity'
                        %Y = rho;
                        %if nargout>2
                        R = ones(obj.nObs,1);
                        %end
                end
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
                case 'normal'
                    smp = Y + sqrt(obj.score.scaling)*randn(obj.nObs,1); % generate from normal distribution
            end
            obj.Predictions.sample = smp;

        end

        %% COMPUTE LOG-LIKELIHOOD
        function  [obj,LLH, gd] = LogLikelihood(obj,fit_mixture_weights)
            %  M = LogLikelihood(M) computes the LogLikelihood of the model
            %
            %  [M,LLH] = LogLikelihood(M)
            %
            % [M,LLH, grad] = LogLikelihood(M) computes the gradient of the
            % LogLikelihood of the parameters w.r.t weights, evaluated at
            % maximum a-posteriori weights
            %
            % LogLikelihood(M, true) for mixture models to update mixture parameters

            % compute predictor if not provided
            if isempty(obj.Predictions.rho)
                obj = Predictor(obj);
            end

            % compute log-likelihood
            switch obj.link
                case 'logit'
                    Y = 1 ./ (1 + exp(-obj.Predictions.rho)); % predicted probability
                case 'log'
                    Y = exp(obj.Predictions.rho);
                case 'identity'
                    Y = obj.Predictions.rho;
                case 'probit'
                    Y = normcdf(obj.Predictions.rho);

            end

            switch obj.obs
                case 'binomial'
                    lh = log(Y.*obj.T + (1-Y).*(1-obj.T));
                case 'poisson'
                    lh = obj.T.*log(Y) - Y -logfact(obj.T);
                case 'normal'
                    if isfield(obj.score, 'scaling')
                        s = obj.score.scaling;
                    else
                        s = var(Y);
                    end
                    lh = - (Y-obj.T).^2./(2*s) - log(2*pi*s)/2;

            end
            if ~isempty(obj.mixture)
                if nargin>1 && fit_mixture_weights
                    % compute M-step (mixture weight update) and E-step for mixture model
                    [obj.mixture, LLH] = obj.mixture.ParameterFitting(lh);
                else
                    % compute E-step for mixture model
                    [obj.mixture, LLH] = obj.mixture.PosteriorFun(lh);
                end
            elseif isempty(obj.ObservationWeight)
                LLH = sum(lh);
            else
                LLH = sum(obj.ObservationWeight .* lh);
            end

            if ~isempty(obj.mixture)
                obj.Predictions.Expected = sum(Y .* obj.mixture.Posterior,2); % marginalize over component
                obj.Predictions.ExpectedComponent = Y;
            else
                obj.Predictions.Expected = Y;
            end
            obj.score.LogLikelihood = LLH;


            % compute gradient of LLH w.r.t weights
            if nargout>2 %
                gd = [];
                % prediction error: difference between target and predictor
                err = (obj.T-Y)';
                if ~isempty(obj.ObservationWeight)
                    err = obj.ObservationWeight' .* err;
                end

                M = obj.regressor;

                for m=1:obj.nMod % for each module

                    for d=1:M(m).nDim % for each dimension
                        for r=1:M(m).rank
                            Phi =  ProjectDimension(M(m),r,d);
                            if M(m).Weights(d).nWeight==1 % correct a bug in ProjectDimension that outputs row vector instead of column
                                Phi = Phi';
                            end
                            this_gd = err*Phi; % gradient of LLH w.r.t each weight in U{r,d}
                            gd = [gd this_gd];
                        end
                    end
                end
            end
        end

        %% computes accuracy for set of parameters
        function accuracy = Accuracy(obj)
            %[obj,Y] = Accuracy(obj); % compute accuracy of model
            %prediction

            Y = obj.Predictions.Expected;
            switch obj.obs
                case 'binomial'

                    correct = obj.T==(Y>.5); % whether each observation is correct (using greedy policy)
                    accuracy = weighted_mean(correct, obj.ObservationWeight); % proportion of correctly classified

                case 'poisson' % poisson
                    %  Y = exp(obj.Predictions.rho);
                    %  LLH = sum(obj.ObservationWeight .* (obj.T.*log(Y)-Y-logfact(obj.T)));
                    correct = (obj.T==Y);
                    accuracy = weighted_mean(correct, obj.ObservationWeight); % proportion of correctly classified
                case 'normal'
                    accuracy = nan;

            end
        end

        %% computes explained variance of model
        function [obj, EV] = ExplainedVariance(obj)
            % M = ExplainedVariance(M)
            %  [M,EV] = ExplainedVariance(M)

            Rse = ( obj.Predictions.Expected -obj.T).^2; % residual squared error
            EV = 1- weighted_mean(Rse, obj.ObservationWeight) / var(obj.T,obj.ObservationWeight);

            obj.score.ExplainedVariance = EV;
        end

        %% COMPUTE LOG-PRIOR
        function  [obj,LP] = LogPrior(obj)
            % M = LogPrior(M) computes the log-prior of the model
            %[M,logprior] = LogPrior(M)
            %
            LP = cell(1,obj.nMod);
            % compute log-prior for each weight in each regressor
            for m=1:obj.nMod
                LP{m} = LogPrior(obj.regressor(m));
            end

            % sum over all
            obj.score.LogPrior = sum(cellfun(@(x) sum(x(:)), LP));
        end



        %% COMPUTE LOG-POSTERIOR
        function  [obj,logjoint] = LogJoint(obj, prior_score, fit_mixture_weights)
            % M = LogJoint(M) computes the LogJoint of the mode
            % [M,logjoint] = LogJoint(M)
            %
            % [M,logjoint] = LogJoint(obj, prior_score)
            %
            % LogJoint(obj, prior_score, fit_mixture_weights) or
            % LogJoint(obj, [], fit_mixture_weights) to update mixture parameters (for mixture models)
            %

            % compute log-likelihood first
            if nargin<3
                fit_mixture_weights = false;
            end
            obj = LogLikelihood(obj, fit_mixture_weights);

            if nargin>1  && ~isempty(prior_score)           % log-prior provided by input
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


        %% COMPUTE HESSIAN OF NEG LOG-LIKELIHOOD (IN UNCONSTRAINED SPACE)
        function [H,P] = Hessian(obj, c_fpd)
            %  H = Hessian(obj) computes the Hessian of the negative
            %  Log-Likelihood in unconstrained space of weights
            %
            %[H,P] = Hessian(obj)
            % P is the project matrix from full parameter space to free
            % parameter space (over all set of regressors)
            M = obj.regressor;
            n = obj.nObs;
            full_idx = [0 cumsum([M.nParameters])]; % position for full parameter in each dimension

            if nargin<2
                c_fpd = [];
            end
            %% compute predictor, prediction values and R
            [obj,Y,R] = ExpectedValue(obj);

            if ~isempty(obj.ObservationWeight)
                R = R .* obj.ObservationWeight;
            end

            W = [M.Weights];

            %if all( cellfun(@(x) all(x>0),{M.nWeight})>0) % unless there is an empty model
            if all([W.nWeight]>0) % unless there is an empty model

                Rmat = spdiags(R, 0, n, n);
            elseif ~isempty(obj.ObservationWeight)
                Rmat = spdiags(obj.ObservationWeight,0,n,n);
            else
                Rmat = speye(n);
            end

            % full design matrix
            Phi = design_matrix(M,[], 0, 0);

            %% Hessian for full parameter set
            H_full = Phi'*Rmat*Phi;

            %% add specific elements for non-diagonal blocks (i.e. different set of weights) for same component

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
                            Ha2 = ProjectDimension(M(m),r,[d f],1,nPE); % add collapsing over observable dimension (Y-T),

                            %  H_full(fdx,fdx2 ) = H_across;
                            H_full(fdx,fdx2 ) = H_full(fdx,fdx2 ) + Ha2;
                            H_full(fdx2,fdx ) =  H_full(fdx,fdx2)';
                        end
                    end
                end
                midx = midx + M(m).nDim * rr; % jump index by number of components in module
            end

            %% project from full parameter space to free basis

            P = projection_matrix(M,'all');  %  projection matrix from full to unconstrained space
            P = full(P);
            H = P*H_full*P';
            H = (H+H')/2; % ensure that it's symmetric (may lose symmetry due to numerical problems)

            if c_fpd>0
                H = force_definite_positive(H, c_fpd);
            end
        end

        %% COMPUTE POSTERIOR COVARIANCE
        function [V, B, invHinvK] = PosteriorCov(obj)
            % V = PosteriorCov(M) computes the posterior covariance in free basis for
            % model M

            % compute Hessian of likelihood
            [H,P] = Hessian(obj);

            %% compute covariance prior
            M = obj.regressor;
            K = prior_covariance_cell(M, true);  % group prior covariance from all modules
            for i=1:length(K)
                K{i} = force_definite_positive(K{i});
            end
            inf_terms = any(cellfun(@(x) full(any(isinf(x(:)))), K));
            K_noinf = K; % prior covariance without infinite terms
            if inf_terms
                % if infinite covariance and no hyperparameter, remove
                % from computation of log marginal evidence
                hp = [M.HP];
                inf_no_hp = cellfun(@(x) any(isinf(x(:))), K) & cellfun(@isempty, {hp.HP});
                for i=inf_no_hp
                    %% !! finish this
                end
            end

            K = blkdiag(K{:}); % prior is block-diagonal
            Kfree = P*K*P'; % project on free basis
            K_noinf = blkdiag(K_noinf{:});
            K_noinf_free = P*K_noinf*P'; % prior covariance in free basis

            if ~issymmetric(Kfree) % often not completely symmetric due to numerical errors
                Kfree = (Kfree+Kfree')/2;
            end

            % not sure I can use this if dim>1 because the
            %likelihood may no longer be convex, so this can have neg
            %eigenvalues
            %sqW = sqrtm(W);
            %B = eye(free_idx(end)) + sqW*Kfree*sqW; % formula to computed marginalized evidence (eq 3.32 from Rasmussen & Williams 2006)
            nFreeParameters = sum(cellfun(@(x) sum(x(:)), {M.nFreeParameters}));%nFreeParameters = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));
            %   nFreeParameters = sum([M.nFreeParameters]);
            B = Kfree*H / obj.score.scaling + eye(nFreeParameters); % formula to compute marginalized evidence

            %       Wsqrt = sqrtm(full(H)); % matrix square root
            %   B = Wsqrt*Kfree*Wsqrt / obj.score.scaling + eye(nFreeParameters); % Rasmussen 3.26

            % I am not applying it because B is not symmetric - perhaps we
            % should if we would use the definition with Wsq (from Rasmussen)
            %   B = force_definite_positive(B);


            % compute posterior covariance
            if inf_terms || all([M.nFreeDimensions]<2)

                % use this if no prior on some variable, or if non convex
                % likelihood
                wrn = warning('off', 'MATLAB:singularMatrix');
                V = inv(H/obj.score.scaling + inv(Kfree));
                warning(wrn.state, 'MATLAB:singularMatrix');
            else
                V = B\Kfree; %inv(W + inv(Kfree));
                %  V = Kfree - Kfree*Wsqrt /inv(B)*Wsqrt*Kfree; % Rasmussen 3.27
            end
            V = full(V);

            % check that covariance is symmetric
            if norm(V-V')>1e-3*norm(V)
                if any(cellfun(@(x) any(sum(x=='f',2)>1), constraint_cell(M)))
                    warning('posterior covariance is not symmetric - this is likely due to having two free dimensions in our regressor - try adding constraints');
                else
                    warning('posterior covariance is not symmetric - dont know why');
                end
            end

            V = (V+V')/2; % may be not symmetric due to numerical reasons

            if nargout>2
                %% compute inv(Hinv+K) - used for predicted covariance for test datapoints
                invHinvK = inv(inv(H) + Kfree);
            end
        end


        %% COMPUTE NUMBER OF PARAMETER
        function obj = computer_n_parameters_df(obj)
            % M = computer_n_parameters(M)
            %

            % we count parameter in projected space
            M = obj.regressor.project_to_basis;

            obj.score.nParameters = sum([M.nTotalParameters]); % total number of parameters (after projecting to

            if isempty(obj.ObservationWeight)
                obj.score.df = obj.nObs - obj.score.nParameters; % degree of freedom
            else
                obj.score.df = sum(obj.ObservationWeight) - obj.score.nParameters; % degree of freedom
            end
        end

        %% SET WEIGHTS AND HYPERPARAMETERS FROM ANOTHER MODEL
        function [obj,I] = set_weights_and_hyperparameters_from_model(obj, varargin)
            [obj.regressor,I] = set_weights_and_hyperparameters_from_model(obj.regressor, varargin{:});
        end

        %% SET HYPERPARAMETERS FROM ANOTHER MODEL
        function [obj,I] = set_hyperparameters_from_model(obj, varargin)
            [obj.regressor,I] = set_hyperparameters_from_model(obj.regressor, varargin{:});
        end

        %% SET WEIGHTS FROM ANOTHER MODEL
        function [obj,I] = set_weights_from_model(obj, varargin)
            [obj.regressor,I] = set_weights_from_model(obj.regressor, varargin{:});
        end

        %% CONCATENATE ALL WEIGHTS
        function U = concatenate_weights(obj)
            % U = concatenate_weights(M)
            % concatenates all weights from model M into a single vector
            U = concatenate_weights(obj.regressor);
            % U = [obj.regressor.U];
            % U = [U{:}];
        end

        %% CONCATENATE WEIGHTS OVER POPULATION
        function obj = concatenate_over_models(obj, place_first)
            % M = concatenate_over_models(M)
            % concatenates weights from array of models (e.g. created by
            % splitting a model) into a single model (usually for plotting
            % and storing)
            %
            % M = concatenate_over_models(M, true) to place model index as
            % first dimension in weights
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
                %                 nReg = cellfun(@(x) x(m).nWeight, {obj.regressor},'unif',0);
                %                 nReg = cat(1,nReg{:});
                %                 if ~all(nReg==nReg(1,:),'all')
                %                     error('number of regressors in module %d differs between models', m);
                %                 end


                W = cellfun(@(x) x(m).Weights, {obj.regressor},'unif',0); % set of weights for this module, for all models
                W = cat(1,W{:}); % weights structure array, nObj x nDim
                fnames = fieldnames(W);
                W = struct2cell(W); % fields x nObj x nDim

                field_fun = @(F) permute(W(strcmp(F, fnames),:,:),[2 3 1]); % function to select values for a given field
                U = field_fun('PosteriorMean'); % nObj x nDim cell array
                se = field_fun('PosteriorStd');
                TT = field_fun('T');
                p = field_fun('p');
                scale = field_fun('scale');

                % in case scale is not specified, use default values
                NoScale = cellfun(@isempty, scale);
                DefaultScale = cellfun(@(x) 1:length(x), U, 'unif',0);
                scale(NoScale) = DefaultScale(NoScale);

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
                        obj(1).regressor(m).Weights(d).scale = sc; %update scale
                    end
                end

                if place_first && rank(1)>1
                    %move rank from dim 1 to dim 3, i.e. to size nObj x nLevel x rank
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

                % concatenate over models
                for d=1:nD(1) % loop through all dimensions of regressor
                    WW = obj(1).regressor(m).Weights(d);

                    WW.PosteriorMean = cat(dd, U{:,d});
                    WW.PosteriorStd = cat(dd, se{:,d});
                    WW.T = cat(dd, TT{:,d});
                    WW.p = cat(dd, p{:,d});

                    obj(1).regressor(m).Weights(d) = WW;
                end


                %% concatenate hyperparameters
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
            % Sc = concatenate_score(M) concatenates scores over model
            % array M (for model selection / model comparison)

            n =  numel(obj);

            all_score = {obj.score};

            with_score = ~cellfun(@isempty, all_score); % models with scores

            all_fieldnames = cellfun(@fieldnames, all_score(with_score),'unif',0);
            all_fieldnames = unique(cat(1,all_fieldnames{:})); % all field names

            % all possible metrics
            metrics = metrics_list();

            % select metrics that are present in at least one model
            metrics = metrics(ismember(metrics, all_fieldnames));

            Sc = struct;
            for i=1:length(metrics)
                mt = metrics{i};

                % pre-allocate values for each model
                X = nan(length(all_score{1}.(mt)),length(obj));

                % add values from each model where value is present
                for m=1:n
                    if ~isempty(all_score{m}) && isfield(all_score{m}, mt) && ~isempty(all_score{m}.(mt))
                        X(:,m) =  all_score{m}.(mt);
                    end
                end
                Sc.(mt) = X;
            end

        end

        %% EXPORT SCORES TO TABLE
        function T = export_scores_to_table(obj)
            % T = export_scores_to_table(M);
            % exports score to table.
            if numel(obj)>1
                S = obj.concatenate_score;
            else
                S = obj.score;
            end
            fn = fieldnames(S);
            for f = 1:length(fn) % make sure they're all row vectors
                X = S.(fn{f});
                if isvector(X)
                    S.(fn{f}) = X(:);
                end
            end
            T = struct2table(S);
        end


        %% EXPORT SCORE TO CSV FILE
        function export_score_to_csv(obj, filename)
            % M.export_score_to_csv(filename) exports model scores as csv file.
            %
            writetable(obj.export_scores_to_table, filename);
        end

        %% AVERAGE WEIGHTS OVER POPULATION
        function obj = population_average(obj)
            % M = population_average(M) averages weights across models in
            % array of models M. Produces a single model M.

            n = length(obj); % size of population (one model per member)

            % first concatenate
            obj = concatenate_over_models(obj, true);

            % now compute average and standard deviation of weights over
            % population
            for m=1:obj.nMod

                if obj.regressor(m).rank(1)>1 % !! check this is how this is done in concatenate_over_models
                    dd = 3; % weight x rank x model
                else
                    dd = 1; % model x weight
                end

                for d=1:obj.regressor(m).nDim
                    W = obj.regressor(m).Weights(d);
                    X = W.PosteriorMean;

                    W.PosteriorMean = mean(X,dd); % population average
                    W.PosteriorStd = std(X,[],dd)/sqrt(n); % standard error of the mean
                    W.T = W.PosteriorMean ./ W.PosteriorStd; % wald T value
                    W.p = 2*normcdf(-abs(W.T)); % two-tailed T-test w.r.t 0

                    obj.regressor(m).Weights(d) = W;

                    % mean and std of hyperparameters
                    H = obj.regressor(m).HP(d);
                    HPHP = H.HP;
                    obj.regressor(m).HP(d).HP = mean(HPHP,1);
                    obj.regressor(m).HP(d).std = std(HPHP,[],1);
                end
            end


        end

        %% EXPORT WEIGHTS TO TABLE
        function T = export_weights_to_table(obj)
            T = export_weights_to_table(obj.regressor);
        end

        %% EXPORT WEIGHTS TO CSV FILE
        function export_weights_to_csv(obj, filename)
            % M.export_weights_to_csv(filename) exports weights data as csv file.
            %
            export_weights_to_csv(obj.regressor, filename);
        end

        %% EXPORT HYPERPARAMETERS TO TABLE
        function T = export_hyperparameters_to_table(obj)
            T = export_hyperparameters_to_table(obj.regressor);
        end

        %% EXPORT HYPERPARAMETERS TO CSV FILE
        function export_hyperparameters_to_csv(obj, filename)
            % M.export_hyperparameters_to_csv(filename) exports hyperparameter data as csv file.
            %
            export_hyperparameters_to_csv(obj.regressor, filename);
        end

        %% COMPUTE PREDICTOR VARIANCE FROM EACH MODULE
        function [obj, PV] = predictor_variance(obj)
            % M = predictor_variance(M)
            % computes raw variance PV over dataset for each module in model.
            % PV is an array of length nMod. The variance is not
            % normalized. For normalized variance see gum.predictor_explained_variance
            %
            % [M, PV] = predictor_variance(M)
            %
            % Not to be confounded with compute_rho_variance

            if ~isscalar(obj)
                PV = cell(size(obj));
                for m=1:numel(obj)
                    [obj(m),PV{m}] = predictor_variance(obj(m));
                end
                return;
            end

            rank = [obj.regressor.rank];
            PV = nan(max(rank),obj.nMod);

            for m=1:obj.nMod
                for r=1:obj.regressor(m).rank
                    PV(r,m) = var(Predictor(obj.regressor(m),r));
                end
            end

            obj.score.PredictorVariance = PV;
        end

        %% COMPUTE PREDICTOR EXPLAINED VARIANCE FROM EACH MODULE
        function [obj, PEV] = predictor_explained_variance(obj)
            % M = predictor_explained_variance(M)
            % computes explained variance PV over dataset for each module in model.
            % PV is an array of length nMod.
            %
            % [M, PEV] = predictor_explained_variance(M)
            %
            % Not to be confounded with predictor_variance or compute_rho_variance
            if ~isscalar(obj) % recursive call
                PEV = cell(size(obj));
                for m=1:numel(obj)
                    [obj(m),PEV{m}] = predictor_explained_variance(obj(m));
                end
                return;
            end

            rank = [obj.regressor.rank];
            PEV = nan(max(rank),obj.nMod);

            for m=1:obj.nMod
                for r=1:obj.regressor(m).rank

                    % compute predictor for this regressor alone
                    rho = Predictor(obj.regressor(m),r); % rho
                    rho = regressor(rho, 'constant');

                    % we need to include offset so we fit a model with one
                    % parameter: offset
                    Msingle = gum(rho, obj.T, struct('observations',obj.obs,'ObservationWeight', obj.ObservationWeight));
                    Msingle = Msingle.infer(struct( 'verbose','off'));
                    [~,PEV(r,m)] = Msingle.ExplainedVariance; % compute explained variance

                    % alternative: not good
                    %   Y = obj.inverse_link_function(rho); % pass through inverse link function
                    %Mse = (Y -obj.T).^2; % model squared error
                    %PEV(r,m) = weighted_mean(Mse, obj.ObservationWeight) / var(obj.T,obj.ObservationWeight);
                end
            end

            obj.score.PredictorExplainedVariance = PEV;
        end

        %% COMPUTE ESTIMATED VARIANCE OF PREDICTOR
        function obj = compute_rho_variance(obj)
            % M = compute_rho_variance(M)
            % computes estimated variance of predictor for each datapoint.

            n = obj.nObs;
            M = obj.regressor;
            sigma = obj.score.covb;

            if any(isinf(sigma(:))) % if no prior defined on any dimension, cannot compute it
                obj.Predictions.rhoVar = nan(n,1);
                return;
            end

            nSample = 1000; % number of sample to compute variance

            full_idx = [0 cumsum([M.nParameters])]; % position for full parameter in each dimension

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
                            obj.regressor(m).Weights(d).PosteriorMean(r,:) = Us(fdx);
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

        %% CHECK FORMAT OF CROSS-VALIDATION
        function obj = check_crossvalidation(obj)

            if ~isfield(obj.param, 'crossvalidation') || isempty(obj.param.crossvalidation)
                % if no cross-validation is planned, we're fine!
                return;
            end

            CV = obj.param.crossvalidation;
            if iscell(CV) % permutationas are already provided as nperm x 2 cell (first col: train set, 2nd: train set)
                %  generateperm = 0;

                % if any([CV{:}]>obj.nObs)
                %     error('maximum crossvalidation index (%d) is larger than number of observations (%d)',max([CV{:}]),n);
                % end

                % replace by structure
                CV_tmp = struct;
                CV_tmp.NumObservations = obj.nObs;
                CV_tmp.NumTestSets = size(CV,1); % number of permutation sets
                CV_tmp.nTrain = length(CV{1,1});
                CV_tmp.nTest = length(CV{1,2});
                CV_tmp.training = CV(:,1)';
                CV_tmp.test = CV(:,2)';
                CV = CV_tmp;
            elseif isa(CV,'cvpartition') || isstruct(CV)
                % we're already set!

                %  nPerm = CV.NumTestSets;
                %  CV = cell(nPerm,2);
                %  for p=1:nPerm
                %      CV{p,1} = find(CV.training(p))'; % training set in first column
                %      CV{p,2} = find(CV.test(p))'; % test set in second colun
                %  end
                % ntrain = length(allperm{1,1});
                % ntest = length(allperm{1,2});



            elseif isscalar(CV) % only the number of observations in training set is provided draw the permutations now
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
                CV.NumObservations = obj.nObs;
                CV.NumTestSets = NumTestSets;
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

            % if the cross-validation set has been defined on
            % the whole dataset and model is fit on subset of
            % data, select corresponding subset of CVset
            if isfield(obj.param, 'split') && CV.NumObservations==length(obj.param.split)

                splt = obj.param.split;

                % create CV structure
                CV_tmp = struct;
                CV_tmp.NumObservations= sum(splt);
                CV_tmp.NumTestSets = CV.NumTestSets;

                for p=1:CV.NumTestSets
                    trainset = CV.training(p);
                    testset = CV.test(p);
                    if iscell(trainset)
                        trainset = trainset{1};
                        testset = testset{1};
                    end
                    CV_tmp.training{p} = trainset(splt); % use subset of training set
                    CV_tmp.test{p} = testset(splt); % use subset of test set
                end
                CV = CV_tmp;
            end

            %  generateperm = 0;
            assert(CV.NumObservations==obj.nObs, 'number of data points in cross-validation set does not match number of observations in dataset');

            obj.param.crossvalidation = CV;

            if isfield(obj.param,'testset') && any(obj.param.testset>obj.nObs)
                error('maximum test set index (%d) is larger than number of observations (%d)',max(obj.param.testset),obj.nObs);
            end
        end

        %% COMPUTE VIF (VARIANCE INFLATION FACTOR)
        function [V, V_free] = vif(obj,D)
            % V = vif(obj,D) computed the Variance Inflation Factor (VIF) for
            % a model. D is a vector specifying the dimensions on which to
            % project for each regressor (by default, all ones).
            %
            %[V, V_free] = vif(obj,D)

            if nargin<2 % by default, project on dimension 1
                D = ones(1,obj.nMod);
            end

            Phi = design_matrix(obj.regressor,[],D);

            PP = projection_matrix(obj.regressor); % free-to-full matrix conversion
            P = blkdiag_subset(PP, D(:)); % projection matrix
            P = P{1};

            Phi = Phi*P';

            R = corrcoef(Phi); % correlation matrix
            V_free = diag(inv(R))'; % VIF in free basis

            % project back to full basis and normalize
            V = (V_free*P) ./ (V_free*ones(size(P)));
        end


        %% PLOT DESIGN MATRIX
        function h = plot_design_matrix(obj, varargin)
            % plot_design_matrix(M) plots the design matrix
            %
            % plot_design_matrix(M,subset) to plot design matrix for a subset of observations
            %
            % plot_design_matrix(M,subset,dims)
            % plot_design_matrix(M,subset,dims, init_weight)
            %
            % h = plot_design_matrix(...) provide graphic handles


            h.Axes = [];
            h.Objects = {};

            [Phi, nReg, dims] = design_matrix(obj.regressor, varargin{:});

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
            %                 nReg(m) = M(m).rank*M(m).nWeight(D(m)); % add number of regressors for this module
            %
            %                 if isempty(M(m).U)
            %                     for d=1:M(m).nDim
            %                         M(m).U{d} = zeros(M(m).rank,M(m).nWeight(d));
            %                     end
            %                 end
            %
            %                 %M(m).Data = tocell(M(m).Data);
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
            %                     idx = ii + (1:M(m).nWeight(D(m))); % index of regressors
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
            f = 1;
            for m=1:obj.nMod
                h_txt = [];
                lbl = {M(m).Weights(dims{m}).label};
                for d=1:length(lbl)
                    if ~isempty(lbl{d})
                        h_txt(d) = text( mean(nRegCum(m:m+1))+1, 0.5, lbl{d},...
                            'verticalalignment','bottom','horizontalalignment','center');
                        if nReg(f)<.2*sum(nReg)
                            set(h_txt(d),'Rotation',90,'horizontalalignment','left');
                        end
                    end
                    f = f+1;
                end
                h.Objects = [h.Objects h_txt];
            end

            axis off;

        end

        %% PLOT POSTERIOR COVARIANCE
        function h = plot_posterior_covariance(obj)
            % plot_posterior_covariance(M) plot the posterior covariance of
            % the model
            %
            % h = plot_posterior_covariance(M) provide graphical handles

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
        function h = plot_weights(obj, U2, varargin)
            % plot_weights(M) plot weights from model M
            % M.plot_weights()
            % M.plot_weights(1:3)
            % M.plot_weights('regressor1')
            % M.plot_weights({'regressor1','regressor2'})
            %
            % M.plot_weights(1:3, h); to use subplots defined by handles
            %
            % nSub = M.plot_weights(..., 'nsubplot') to have number of
            % subplots
            %
            % If M is an array of models, will display weights in a grid of
            % subplots
            if ~all(isestimated(obj))
                error('weights have not been estimated or set');
            end


            if nargin<2 || isequal(U2, 'nsubplot') || isempty(U2)
                if nargin>=2 && isequal(U2, 'nsubplot')
                    varargin = {U2};
                end
                if length(obj)>1
                    U2 = [];
                else
                    U2 = 1:obj.nMod;
                end
            end

            if length(obj)>1

                % first check that all models all have same number of
                % subplots
                nObj = length(obj);
                nsub = zeros(1,nObj);
                for mm = 1:nObj
                    nsub(mm) = plot_weights(obj(mm), U2, 'nsubplot');
                end

                if ~all(nsub==nsub(1))
                    error('cannot plot weights for arrays of models if all models do not have the same number of subplots');
                end
                nsub = nsub(1);

                % now plot each model using recursive calls
                for mm=1:nObj
                    if nsub>1 % more than one subplot per model: subplots with one row per model, one column per regressor
                        for i=1:nsub
                            idx = (mm-1)*nsub + i;
                            this_h(i) = subplot(nObj, nsub, idx);
                        end
                    else % one subplot per model: all models in a grid
                        this_h = subplot2(nObj, mm);
                    end
                    h(mm) = plot_weights(obj(mm), U2, this_h);

                    % remove redundant labels and titles
                    if mm>1 && nsub>1
                        title(this_h,'');
                    end
                    if mm<nObj && nsub>1
                        xlabel(this_h, '');
                    end
                end
                return;
            end


            % select corresponding regressors to be plotted
            M = SelectRegressors(obj.regressor, U2);

            only_nsubplot = ~isempty(varargin) && isequal(varargin{1}, 'nsubplot');
            cols = defcolor;
            colmaps = {'jet','hsv','winter','automn'};

            NoFixedWeights = cell(1,obj.nMod);
            constraint = constraint_cell(M);
            for m=1:numel(M)%obj.nMod
                NoFixedWeights{m} = any(constraint{m}~='n',1); % regressors with constant
            end
            nNoFixedWeights  = cellfun(@sum,NoFixedWeights); % number of non-fixedd dimensions in each module
            nSubplots = sum(nNoFixedWeights); % number of subplots

            if only_nsubplot % only ask for number of subplots
                h = nSubplots;
                return;
            end

            % check if subplots are assigned
            if ~isempty(varargin) && ~isequal(varargin{1}, 'nsubplot')
                h.Axes = varargin{1};
                varargin(1) = [];
            else % otherwise define a grid of subplots in current figure
                for i=1:nSubplots
                    h.Axes(i) = subplot2(nSubplots,i);
                end
            end

            i = 1; % subplot counter
            c = -1; % color counter
            cm = 1; % colormap counter
            % h.Axes = [];
            h.Objects = {};

            for m=1:numel(M)%obj.nMod

                for d = find(NoFixedWeights{m}) % loop between subplots

                    axes(h.Axes(i)); % set as active subplot

                    W = M(m).Weights(d); % set of weights

                    % add label
                    if isempty(W.label)
                        W.label = sprintf('U%d_%d',m,d); % default label
                    end

                    % plot weights
                    [h_nu, c, cm] = plot_single_weights(W, M(m).rank, c, cm, cols, colmaps);

                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end
        end

        %% PLOT HYPERPARAMETERS
        function h = plot_hyperparameters(obj)
            % plot_hyperparameters(M) plots values of hyperparameters in model
            % M.


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


                for d=1:ndim(m)
                    h.Axes(end+1) = subplot2(nDim_tot,i);
                    if isempty(M(m).Weights(d).label)
                        M(m).Weights(d).label = sprintf('U%d_%d',m,d); % default label
                    end
                    title(M(m).Weights(d).label);

                    [~,~,h_nu] = wu(M(m).HP(d).HP',[],{M(m).HP(d).label},'bar');

                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end

        end

        %% PLOT DATA GROUPED BY VALUES OF PREDICTOR
        function h = plot_data_vs_predictor(obj, Q)
            % h = plot_data_vs_predictor(M)
            % h = plot_data_vs_predictor(M, Q)
            %
            % TODO: add raw plot

            if length(obj)>1 % if more than one model, plot each model in different subplot
                if nargin<2
                    Q = [];
                end
                h = cell(size(obj));
                for i=1:numel(obj)
                    [~, nRow, nCol] = subplot2(length(obj), i);
                    h{i} = obj(i).plot_data_vs_predictor(Q); % recursive call
                    if mod(i,nCol)~=1
                        ylabel('');
                    end
                    if ceil(i/nCol)<nRow
                        xlabel('');
                    end
                end
                return;
            end

            [~,rho] = Predictor(obj);

            if nargin<2 || isempty(Q) % number of quantiles: by default scales with square root of number of observations
                Q = ceil(sqrt(obj.nObs)/2);
            end

            H = quantile(rho,Q);

            H = [-inf H inf];

            if isempty(obj.ObservationWeight)
                w = ones(obj.nObs,1);
            else
                w = obj.ObservationWeight;
            end

            M = zeros(1,Q+1);
            sem = zeros(1,Q+1);
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

            % plot inverse link function as reference
            xx = linspace(min(xlim), max(xlim), 200);
            yy = obj.inverse_link_function(xx);
            switch obj.link
                case 'logit'
                    plot(xlim,.5*[1 1], 'color',.7*[1 1 1]);
                    plot([0 0],ylim, 'color',.7*[1 1 1]);
                case 'probit'
                    plot(xlim,.5*[1 1], 'color',.7*[1 1 1]);
                    plot([0 0],ylim, 'color',.7*[1 1 1]);
            end
            plot(xx,yy,'b');
            xlabel('\rho');
            ylabel('dependent variable');


        end


        %% PLOT SCORES (MODE COMPARISON)
        function [h,X] = plot_score(obj, score, ref, labels, plottype)
            %  plot_score(M, score)
            % plots score for different models (for model comparison)
            % score can be either
            % 'AIC','AICc','BIC','LogEvidence','LogJoint','LogPrior','FittingTime'
            % It can also be a cell array, in which case each score is
            % plotted in a different subplot
            %
            % plot_score(M, score, ref, labels)
            % ref provides an index to the reference model (its value is set to 0)
            %
            % plot_score(M, score, ref, labels)
            % provides labels to each model
            %
            % plot_score(M, score, ref, 'scatter') to use scatter plot
            % instead of bar plots
            %
            %  h = plot_score(...) provides graphical handles

            if nargin<3
                ref = [];
            end

            if nargin<4
                labels = {};
            end

            if nargin<5
                plottype = 'bar';
            end
            % if cell array of score, plot each one in a different subplot
            if iscell(score)
                nScore = length(score);
                X = cell(1,nScore);
                h = cell(1,nScore);
                for s=1:nScore
                    subplot(1,nScore,s);

                    [h{s},X{s}] = plot_score(obj, score{s}, ref, labels); % recursive call
                end
                return;
            end

            % list of all possible metrics
            metrics = metrics_list();

            i_score = find(strcmpi(score, metrics)); % find case-insensitive match
            if isempty(i_score)
                error(['incorrect score type:%s - Possible metrics include:\n ' repmat('%s, ',1,length(metrics)-1) '%s'], ...
                    score, metrics{:});
            end
            score = metrics{i_score};


            % concatenate scores of different models into single structure
            Sc =  concatenate_score(obj);
            X = Sc.(score); % extract relevant data
            if isvector(X)
                X = X(:);
            end

            if ~isempty(ref) % use one as reference
                X = X - X(ref,:);
                score = ['\Delta ' score];
            end

            % draw bars
            if strcmp(plottype, 'bar')
                h = barh(X);
            else
                hold on;
                smb = '.ox+*sdv^<>ph'; % all possible symbols
                smb = repmat(smb, 1, ceil(size(X,2)/length(smb)));
                for xx = 1:size(X,2) % for each set of data
                    h(xx) = plot(X(:,xx), 1:size(X,1), smb(xx));
                end
            end
            xlabel(score);

            if ~isempty(labels)
                yticks(1:length(labels))
                yticklabels(labels);
            end

        end

        %% PLOT BASIS FUNCTIONS OF REGRESSOR
        function h = plot_basis_functions(obj, varargin)
            % plot_basis_functions(M) plots basis functions in regressor
            %
            %plot_basis_functions(M, dims) to specify which regressor to
            %select
            %
            %h = plot_basis_functions(...) provides graphical handles
            h = plot_basis_functions(obj.regressor, varargin{:});
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

constraint = constraint_cell(M);
for m=1:nM % for each regressor object
    if all(constraint{m}(:)=='n') % if all fixed weights
        UpOrder(m,:) = ones(1,D); % then we really don't care
    else
        fir = first_update_dimension(M(m)); % find first update dimension
        no_fixed_dims = find(any(constraint{m} ~= 'n',1)); % find all dimensions whose weights aren't completely fixed
        fir = find(no_fixed_dims== fir);
        UpOrder(m,:) = 1+mod(fir-1+(0:D-1),length(no_fixed_dims)); % update dimension 'fir', then 'fir'+1, .. and loop until D
        UpOrder(m,:) = no_fixed_dims(UpOrder(m,:));
    end
end
end

%% for projection or covariance matrix, group matrices from all modules that are used in each update
function NN = blkdiag_subset(PP, update_o)
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


%% select regressors by index or label  (e.g. for plotting) !! should move to regressor
function M = SelectRegressors(M, idx)
if isstring(idx)
    idx = {idx};
end

AllWeights = [M.Weights];
AllLabel = {AllWeights.label};

% if regressor labels are provided, convert to labels
jdx = cell(1,length(idx));
if iscell(idx)
    for m=1:length(idx)
        ii = find(strcmp(idx{m},AllLabel));
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
end

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
if ~isempty(UU)
    M = M.set_weights(UU);
end

param = obj.param;
param.originalspace = false;
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
M = set_hyperparameters(M, HP, idx);

% evaluate covariances at for given hyperparameters
M = M.compute_prior_covariance;

% check that there is no nan value in covariance matrix
all_sigma = M.global_prior_covariance;
if any(isnan(all_sigma(:)))
    negME = nan;
    return;
end

obj.regressor = M;

% estimate weights from GUM
obj = obj.infer(param);

% for decomposition on basis functions, convert weights back to
% original domain
obj.regressor = project_from_basis(obj.regressor);

% negative marginal evidence
negME = -obj.score.LogEvidence;

% if best parameter so far, update the value for initial parameters
if negME < fval
    fval = negME;
    UU = concatenate_weights(M);
end

nfval = nfval+1;

end


%% CROSS-VALIDATED LLH SCORE
function [errorscore, grad] = cv_score(obj, HP, idx, return_obj)

persistent UU fval nfval;
%% first call with no input: clear persistent value for best-fitting parameters
if nargin==0
    fval = [];
    UU = [];
    nfval = 0;
    return;
end
first_eval = isempty(fval);
if first_eval
    fval = Inf;
end

%nMod = obj.nMod;
M = obj.regressor;

if ~isempty(UU)
    M = M.set_weights(UU);
end

% assign hyperparameter values to regressors
M = set_hyperparameters(M, HP, idx);

param = obj.param;
param.originalspace = false;
if strcmp(param.verbose, 'full')
    param.verbose = 'on';
elseif strcmp(param.verbose, 'on') && first_eval
    param.verbose = 'little'; %'on'; %;
else
    param.verbose = 'off';
end

if nfval>0
    param.initialpoints = 1; % just one initial point after first eval
end

% evaluate covariances at for given hyperparameters
if nargout>1 % with Jacobian of Prior
    [obj.regressor,param.CovPriorJacobian] = compute_prior_covariance(M);
else % same without gradient
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

% for decomposition on basis functions, convert weights back to
% original domain
obj.regressor = project_from_spectral(obj.regressor);

% if best parameter so far, update the value for initial parameters
if errorscore < fval
    fval = errorscore;
    UU = concatenate_weights(M);
end

nfval = nfval+1;

% if return full GUM object instead of score
if return_obj
    errorscore = obj;
end

end

%% converts to cell if not already a cell
%function x= tocell(x)
%if ~iscell(x)
%    x = {x};
%end
%end

%% negative cross-entropy of prior and posterior
% multivariate gaussians (for M-step of EM in hyperparameter optimization)
function [Q, grad] = mvn_negxent(covfun, mu, scale, m, Sigma, P, HP, HPs, B)
k = size(P,1); % dimensionality of free space

HPs.HP(HPs.fit) = HP;

msgid1 = warning('off','MATLAB:nearlySingularMatrix');
msgid2 = warning('off','MATLAB:SingularMatrix');

mdif = (m-mu)*P'; % difference between prior and posterior means (in free basis)

%if ~isempty(B) && B.fixed
%    scl = B.scale;
%else
scl = scale;
%end

[K, gradK] = covfun(scl, HPs.HP, B); % prior covariance matrix and gradient w.r.t hyperparameters
if isstruct(gradK), gradK = gradK.grad; end

if ~isempty(B) && ~B.fixed % work on original space

    nHP = length(HPs.HP);


    if isrow(B.scale)
        [B.B,~,~,gradB] = B.fun(B.scale, HPs.HP, B.params);
    else
        % more rows in scale means we fit different
        % functions for each level of splitting
        % variable
        [id_list,~,split_id] = unique(B.scale(2:end,:)','rows'); % get id for each observation
        B.B = zeros(0,size(B.scale,2)); % the matrix will be block-diagonal
        gradB = zeros(0,size(B.scale,2),nHP);
        for g=1:length(id_list)
            subset = split_id==g; % subset of weights for this level of splitting variable
            [this_B,~,~, this_gradB] = B.fun(B.scale(1,subset), HPs.HP, B.params);
            n_new = size(this_B,1);
            B.B(end+1 : end+n_new, subset) = this_B; %
            gradB(end+1 : end+n_new, subset,:) = this_gradB;
        end

    end

    % gradient for K for basis hyperparameter
    gradK_tmp = zeros(B.nWeight, B.nWeight,nHP);

    for p=1:nHP % for all fitted hyperparameter
        gradK_tmp(:,:,p) = B.B'*gradK(:,:,p)*B.B + 2*B.B'*K*gradB(:,:,p);
    end
    gradK = gradK_tmp;

    % project prior covariance on full space
    K = B.B'*K*B.B;

    % we make prior and posterior matrices full rank artificially to allow for the EM to rotate the
    % basis functions - now there's no guarantee that the LLH will increase
    % in each EM iteration, and no guarantee that it converges to a
    % meaningful result
    K = force_definite_positive(K, max(eig(K))*1e-3);
    Sigma = force_definite_positive(Sigma, max(eig(Sigma))*1e-3);

end
K = P*K*P'; % project on free base

KinvSigma = K \Sigma;
%KinvSigma = SigmaChol'* K \SigmaChol; % if L^T*L = Sigma, then trace(K^-1 Sigma) = trace(L*K^-1*L^T)
%KinvSigma = (KinvSigma+KinvSigma')/2; % make sure it's symmetric

% compute log-determinant (faster and more stable than log(det(X)),
% from Pillow lab code), since matrix is positive definite
%[C,noisPD] = chol(K);

%[C,noisPD] = chol(KinvSigma); % instead of log det(K), we compute -log det(L^T * K^-1 L) which is symmetric too and may deal better with non-full rank covariance prior
%[C,noisPD] = chol(KinvSigma);
% instead of log det(K), we compute -log det(K^-1 Sigma) which may deal better with non-full rank covariance prior

%if noisPD % if not semi-definite (must be a zero eigenvalue, so logdet is -inf)
%    warning('covariance is not semi-definite positive, hyperparameter fitting through EM may not work properly, you may change the covariance kernels or try fitting by cross-validation instead');
%LD = logdet(S);
LD = logdet(KinvSigma);
% LD = -logdet(K);
%Q = inf;
%grad = nan(1,nP);
%return;
%else
%    LD = 2*sum(log(diag(C)));
%end

% negative cross-entropy (equation 19)
Q = (trace(KinvSigma) - LD + (mdif/K)*mdif' + k*log(2*pi)   )/2;
%Q = (- LD )/2;


% gradient (note: gradient check may fail if K is close to singular but
% that's because of the numerical instability in computing Q, the formula
% for the gradient is correct)
grad= zeros(1,sum(HPs.fit));
cnt = 1; % HP counter
for p=find(HPs.fit) % for all fitted hyperparameter
    this_gK = P*gradK(:,:,p)*P';
    KgK = K\this_gK;
    grad(cnt) = -(  trace( KgK*(KinvSigma-eye(k)))   + mdif*KgK*(K\mdif')  )/2;
    %  grad(cnt) = -(  trace( KgK)    )/2;

    cnt = cnt + 1;
end

warning(msgid1.state,'MATLAB:nearlySingularMatrix');
warning(msgid2.state,'MATLAB:SingularMatrix');
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

%% PARSE FORMULA OF GUM (FOR TABLE SYNTAX)
function [M,T, param] = parse_formula(Tbl,fmla, param)
if ~ischar(fmla)
    error('if the first argument is a table, then the second must be a formula string');
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
OpenBracketPos = []; % position of opening brackets
CloseBracketPos = []; % position of closing brackets

option_list = {'sum','mean','tau','variance','binning','constraint', 'type', 'period','fit'}; % possible option types
TypeNames = {'linear','categorical','continuous','periodic', 'constant'}; % possible values for 'type' options
FitNames = {'all','none','scale','tau','ell','variance'}; % possible value for 'fit' options
basis_regexp = {'poly([\d]+)(', 'exp([\d]+)(', 'raisedcosine([\d]+)('}; % regular expressions for 'basis' options
lag_option_list = {'Lags','group','basis', 'split','placelagfirst'}; % options for lag operator

is_null_regressor = false;
inLag = false; % if we're within a lag operator
LagStruct = {};
CloseBrackPos = [];
separator = ';'; % separator for options


% check for mixture model
[mixture_type, fmla] = starts_with_word(fmla, {'multinomial','regression', 'hmm'});
withMixture = ~isempty(mixture_type);
if withMixture % check if number of mixture components is provided
    i = find(fmla=='(',1);
    assert(~isempty(i), sprintf('missing parenthesis after %s',mixture_type));
    if i>1
        nC = str2num(fmla(1:i-1));
        mixture_type = [  mixture_type fmla(1:i-1)];
        fmla(1:i) = [];
        %else
        %  nC = 0; % number of components are not defined, look for ;
    end
    mixture_regressors = '';
end

% parse the formula until there's nothing left to parse...
while ~isempty(fmla)

    % check for opening bracket
    if fmla(1)=='('
        OpenBracketPos(end+1) = length(O); % position of open bracket
        CloseBrackPos(length(OpenBracketPos)) = nan; % filler for corresponding position of close bracket
        fmla = trimspace(fmla(2:end));
    end


    % if starting with minus
    if fmla(1)=='-'
        V{end+1} = struct('variable',-1, 'type','constant','opts',struct());
        O(end+1) = '*';
        fmla = trimspace(fmla(2:end));
    end

    % chech for lag operator
    [transfo, fmla] = starts_with_word(fmla,{'lag('});
    if ~isempty(transfo)
        assert(~inLag, 'cannot define nested lag operators');
        inLag = true;
        LS = struct;
        LS.ini = length(O); % position of start of lag operator
    end

    transfo_label =  {'f(','cat(', 'flin(', 'fper(', 'poly[\d]+(', 'exp[\d]+(', 'raisedcosine[\d]+('};
    [transfo, fmla, transfo_no_number] = starts_with_word(fmla,transfo_label);
    if ~isempty(transfo) % for syntax f(..) or cat(...)

        opts = struct();
        switch transfo_no_number
            case {'f(', 'poly(','exp(', 'raisedcosine('}
                type = 'continuous';
                if ~strcmp(transfo, 'f(')
                    opts.basis = transfo(1:end-1);
                end
            case 'flin('
                type = 'linear';
            case 'fper('
                type = 'periodic';
            otherwise
                type = 'categorical';
        end
        fmla = trimspace(fmla);
        [v, fmla] = starts_with_word(fmla, VarNames);
        if isempty(v)
            error('''%s'' in formula must be followed by variable name',transfo);
        end
        v = string(v); % string array

        % check for multiple variables (i.e f(x,y)
        while fmla(1)==','
            fmla(1) = [];

            [new_v, fmla] = starts_with_word(fmla, VarNames);
            if isempty(new_v)
                error('''%s'' in formula must be followed by variable name',transfo);
            end
            v(end+1) = string(new_v);

            opts.sum = 'joint';
        end

        %% process regressor options


        % check for split variable first
        [opts, fmla] = process_split_variable(fmla, opts, VarNames);

        while fmla(1)~=')'
            if fmla(1) ~= separator
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
                case {'sum','mean'} % constraint type
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
                case 'basis'
                    [opts.basis, fmla] = starts_with_word(fmla, basis_regexp  );
                    assert(~isempty(opts.basis), 'incorrect basis type');

                case 'type' % define regressor type

                    [type, fmla] = starts_with_word(fmla, TypeNames);
                    if isempty(type)
                        error('incorrect regressor type for variable ''%s''', v);
                    end
                case 'fit'
                    [opts.HPfit, fmla] = starts_with_word(fmla, FitNames);
                    if isempty(opts.HPfit)
                        error('incorrect fit option for variable ''%s''', v);
                    end
                case 'period' % define period
                    i = find(fmla==')' | fmla==separator,1); % find next coma or parenthesis
                    if isempty(i)
                        error('incorrect formula, could not parse the value of period')
                    end
                    opts.period = eval(fmla(1:i-1)); % define period

                    fmla(1:i-1) = [];
                case 'constraint'
                    % i = 1;
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


    elseif fmla(1)=='[' % fixed regressor ([x])
        i = find(fmla==']',1);
        if isempty(i)
            error('incorrect formula: bracket ] was not found');
        end
        v = trimspace(fmla(1:i-1));
        fmla = trimspace(fmla(2:end));

        V{end+1} = struct('variable',v, 'type','none');

    else % variable name (linear weight)
        [v, fmla] = starts_with_word(fmla, VarNames);

        if ~isempty(v) %% linear variable

            opts = struct;

            % check for split variable first
            [opts, fmla] = process_split_variable(fmla, opts, VarNames);

            if iscategorical(Tbl.(v))
                type = 'categorical';
            else
                type = 'linear';
            end

            V{end+1} = struct('variable',v, 'type',type, 'opts', opts);

        elseif ~isnan(str2double(fmla(1))) %% numeric constant

            [Num, fmla] = starts_with_number(fmla);
            if Num==0 && (isempty(O) || any(O(end)=='+-'))% special case: 0 means do not include offset
                param.constant = 'off';
                is_null_regressor = true;

            else
                V{end+1} = struct('variable',Num, 'type','constant','opts',struct());
            end
        else
            error('could not parse formula at point ''%s''', fmla);

        end
    end

    if isempty(fmla)
        if is_null_regressor && ~isempty(O)
            O(end) = [];
        end
        break;
    end

    % check for closing bracket or separator (for lag operator)
    if any(fmla(1)==');')


        pp_bracket = find(isnan(CloseBrackPos),1,'last'); % check last open bracket which has not been closed yet
        assert(fmla(1)==')' || inLag, 'incorrect separator '';'' in formula');
        assert(~isempty(pp_bracket) || withMixture || inLag, 'there is no open parenthesis to close');

        if fmla(1)==';' || (inLag && isempty(pp_bracket)) || (inLag && LS.ini>=pp_bracket) % end of lag operator

            LS.end = length(O)+1; % position of operator


            while fmla(1)~=')'  % process lag options
                fmla(1) = []; % remove separator

                i = find(fmla=='=',1);
                if isempty(i)
                    error('missing ''='' sign for option in formula at ''%s''', fmla);
                end

                option = trimspace(fmla(1:i-1));
                if ~any(strcmpi(option, lag_option_list))
                    error('incorrect lag option type: %s',option);
                end
                fmla(1:i) = [];
                fmla = trimspace(fmla);


                switch lower(option)
                    case 'lags'
                        i = find(fmla==')' | fmla==separator,1);
                        assert(~isempty(i),'lag operator is not closed');
                        LS.lags = str2num(fmla(1:i-1));
                        fmla(1:i-1) = [];
                        if isnan(LS.lags)
                            error('Value for option ''%s'' should be numeric',option);
                        end
                    case 'split'
                        [ LS.splitvalue, fmla] = starts_with_boolean(fmla);
                        assert( ~isnan( LS.splitvalue), 'Value for option ''split'' should be boolean');
                    case 'placelagfirst'
                        [LS.placelagfirst, fmla] = starts_with_boolean(fmla);
                        assert( isnan(LS.placelagfirst), 'Value for option ''placelagfirst'' should be boolean');
                    case 'basis'
                        basis_regexp_nofinal = cellfun(@(x) x(1:end-1),basis_regexp,'UniformOutput',false);
                        [LS.basis, fmla] = starts_with_word(fmla, basis_regexp_nofinal  );
                        assert(~isempty(LS.basis), 'incorrect basis type');
                    case 'group'
                        [GroupVar, fmla] = starts_with_word(fmla, VarNames);
                        LS.group = Tbl.(GroupVar);
                end
                fmla = trimspace(fmla);

            end
            LagStruct{end+1} = LS; % add to other lag operators
            inLag = false; % we're officially out of lag operator

        elseif ~isempty(pp_bracket) % signalling the end of a bracket
            CloseBracketPos(pp_bracket) = length(O);
        end
        fmla = trimspace(fmla(2:end));
    end

    if isempty(fmla)
        break;
    end

    %% Now let's look for operations
    if ~any(fmla(1)=='+*-:;')
        error('Was expecting an operator in formula at point ''%s''', fmla);
    end


    if ~is_null_regressor || ~any(fmla(1)=='+-') % unless it's a plus or minus coming after 0, add operation
        O(end+1) = fmla(1);
        fmla(1) = [];
        fmla = trimspace(fmla);
    elseif fmla(1)=='+'
        fmla(1) = [];
        fmla = trimspace(fmla);
    end

    if fmla(1)==';' % separator for mixture models
        if ~withMixture
            error('; symbol in formula is restricted to mixture models')
        end
        [type, fmla] = starts_with_word(fmla, {'mixtureregressors=','mixturecategory='});
        if ~isempty(type) % now part of the formula that defines mixture regressors/category
            mixture_regressors = type;
            O(end) = 'M'; % change operator label
        end
    end
end

assert(~any(isnan(OpenBracketPos)), 'one or more parenthesis was not closed');


%% build the different regressors
for v=1:length(V)
    w = V{v}.variable;
    if ischar(w)
        w = string(w);
    end
    if isstring(w) % for f(x) or f(x,y,...)
        label = char(w(1));
        x = zeros(nObs,length(w));
        for idxV = 1:length(w)
            x(:,idxV) = Tbl.(w(idxV));
            if idxV>1
                label = [label ',' char(w(idxV))];
            end
        end

        %  elseif isstring(w) || ischar(w) % variable name
        %      x = Tbl.(w);
        %      label = w;
    else % numerical value
        x = w*ones(nObs,1);
        label = num2str(w);
    end

    opts_fields = fieldnames(V{v}.opts);
    opts_values = struct2cell(V{v}.opts);
    opts_combined = [opts_fields opts_values]';

    % condition regresor on value of split variable
    if isfield(V{v}.opts, 'split')
        split_var =  V{v}.opts.split;
        opts_combined(:,strcmp(opts_fields,'split')) = []; % remove split (not an option to rgressor constructor method)
    else
        split_var = [];
    end

    V{v} = regressor(x, V{v}.type, 'label',label, opts_combined{:});

    if ~isempty(split_var)
        V{v} = split(V{v}, Tbl.(split_var));
    end
end


%build predictor from operations between predictors
[M, idxC] = compose_regressors(V,O, OpenBracketPos, CloseBracketPos, LagStruct);

if  withMixture
    param.mixture = mixture_type;
    if ~all(idxC==1)
        param.indexComponent = idxC;
    end

    % locate mixture regressors
    i_mixt_reg = find(idxC==0);
    if ~isempty(i_mixt_reg)
        MixtReg = M(i_mixt_reg);
        M(i_mixt_reg) = [];
        assert(all([MistReg.nDim]==1),'mixture regressors can only be one-dimensional');
        MixtReg = MistReg.concatenate_regressors; % put all into single regressors
        MixtReg = MixtReg.Data; % extract design matrix
    else
        MixtReg = [];
    end
end

end

%% compose regressors into predictor
function [M, idxC] = compose_regressors(V,O, OpenBracketPos, CloseBracketPos, LagStruct)


%% process parentheses and lag operators first
while ~isempty(OpenBracketPos) || ~isempty(LagStruct) % as long as there are parenthesis or lag operators to process

    if ~isempty(OpenBracketPos)
        % select which parenthesis to process first
        i = 1;
        while length(OpenBracketPos)>i && OpenBracketPos(i+1)<CloseBracketPos(i)
            i = i+1; % if another bracket is opened before this one is closed, then process the inner bracket first
        end

        % make sure there's no lag operator within this bracket
        if ~isempty(LagStruct)
            LSini = cellfun(@(x) x.ini,LagStruct);
            i_lag = find(LSini>OpenBracketPos(i) & LSini<=OpenBracketPos(i),1);
            process_lag = ~isempty(i_lag);
        else
            process_lag = false;
        end

    else % if there's no more brackets to process, then we're left with processing lags
        process_lag = true;
        i_lag =1; % start with first lag operator
    end

    if process_lag
        %process lag
        LS = LagStruct{i_lag};
        idx = LS.ini+1 : LS.end; % indices of regressors into the bracket
        i = i_lag;
    else
        % process brackets
        idx = OpenBracketPos(i)+1:CloseBracketPos(i); % indices of regressors into the bracket
    end


    % recursive call: compose regressors within brackets
    [V{i}, this_idxC] = compose_regressors(V(idx),O(idx(1:end-1)), [], [],[]);
    assert( all(this_idxC==1), 'incorrect formula: cannot use mixture break within brackets');


    % if possible, concatenate regressors within parenthesis into single
    % regressor (useful e.g. for later multiplying with other term)
    if cat_regressor(V{i}, 1, 1)
        V{i} = cat_regressor(V{i}, 1);
        this_idxC(length(V{i})+1:end) = [];
    end

    if process_lag
        % process lag operator
        LS = rmfield(LS, {'ini','end'});
        if isfield(LS, 'lags')
            Lags = LS.lags;
            LS = rmfield(LS, 'lags');
        else
            Lags = 1;
        end
        opts_fields = fieldnames(LS);
        opts_values = struct2cell(LS);
        opts_combined = [opts_fields opts_values]';
        V{i} = laggedregressor(V{i},Lags, opts_combined{:});

        LagStruct(i) = [];
    else

        % remove elements already processed
        OpenBracketPos(i) = [];
        CloseBracketPos(i) = [];
    end

    V(idx(2:end)) = []; % regressors
    O(idx(1:end-1)) = []; % operators
end

%% build predictor from operations between predictors
current_idx = 1;
idxC = [];

while length(V)>1

    if any(O==':') % first process interaction interactions

        v = find(O==':',1);

        V{v} = V{v} : V{v+1}; % compose product

        % remove regressor and operation
        V(v+1) = [];
        O(v) = [];

    elseif any(O=='*') % then perform products
        v = find(O=='*',1);

        V{v} = V{v} * V{v+1}; % compose product

        % remove regressor and operation
        V(v+1) = [];
        O(v) = [];

    else % now we're left with additions, substractions
        % and break between mixtures (which here we treat just as addition)

        if isempty(idxC)
            idxC = ones(1,length(V));
        end

        if O(1)=='-' % substraction
            V{2} = V{2}*(-1);
        elseif O(1) ==';' % break between mixture
            idxC(length(V{1})*1:end) = idxC(length(V{1})*1:end) + 1; % raise mixture index for all regressors after break
        end

        V{1} = V{1} + V{2}; % compose addition

        % remove regressor and operation
        V(2) = [];
        O(1) = [];

    end

end

M = V{1};

end

%% check if string starts with any of possible character strings
function [word, str, word_no_number] = starts_with_word(str, WordList)
word = '';
word_no_number = '';

R = regexp(str,WordList);

% check whether there's a match (starting at first position)
WordMatch = cellfun(@(x) ~isempty(x)&&x(1)==1, R);

w = find(WordMatch);
if length(w)>1 % if more than one match

    % select longer string
    wLength = cellfun(@length, WordList(w));
    [~,i_maxLength] = max(wLength);
    w = w(i_maxLength);
end

%for w=1:length(WordList)
%    if ~isempty(R{w}) && R{w}(1)==1
if ~isempty(w)
    %we've got a match
    word = WordList{w};
    % wd = WordList{w};
    % if length(str)>=length(wd) && strcmp(str(1:length(wd)),wd)
    %   word = wd;

    str = trimspace(str);

    % check if word includes number (e.g. 'exp4')
    number_extent = regexp(word,'(\[\\d\]\+)','tokenExtents');

    word_no_number = word;


    if ~isempty(number_extent)
        % remove number

        number_extent = number_extent{1};
        word_no_number(number_extent(1):number_extent(end)) = [];



        % remove corresponding characters from string
        word_token = word;
        % word_token(find(word_token=='(',1)) = [];
        %             word_token(find(word_token==')',1)) = [];
        word_token = strrep(word_token,'([\d]+)','[\d]+'); % new!!

        word_token = ['(' strrep(word_token, '(','\(') ')'];
        word_extent = regexp(str,word_token,'tokenExtents');

        word = str(1:word_extent{1}(2));
        str(1:word_extent{1}(2)) = [];
    else
        str(1:length(word)) = []; % remove corresponding characters from string

    end
end
%end
end

%% check if string starts with a number
function  [Num, fmla] = starts_with_number(fmla)

% find longest string that corresponds to numeric value
Num = nan(1,length(fmla));
for i=1:length(fmla)
    if ~any(fmla(1:i)==',') % for some reason still gives non nan when there is a coma
        Num(i) = str2double(fmla(1:i));
    end
end
%i = find(~isnan(Num) & ~(fmla==','),1,'last');
i = find(~isnan(Num),1,'last');

if isempty(i)
    Num = nan;
else
    Num = Num(i);


    fmla(1:i) = [];
    fmla = trimspace(fmla);
end
end

%% check if string starts with a boolean
function [bool,fmla] = starts_with_boolean(fmla)

WordList = {'0','1','false','true','False','True'};
[bool,fmla] = starts_with_word(fmla, WordList);

bool = any(strcmp(bool,{'1','true','True'} ));
end


%% process split variable (TODO: should allow more than one splitting variable, e.g. x|z1,z2)
function [opts, fmla] = process_split_variable(fmla, opts, VarNames)

if ~isempty(fmla) && fmla(1)=='|'
    fmla = trimspace(fmla(2:end));

    [v_split, fmla] = starts_with_word(fmla, VarNames);
    if isempty(v_split)
        error('| in formula must be followed by variable name');
    end

    opts.split = v_split;
end
end

%% trim space at the beginning and end of a character string
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

%% mean weighted by observation weight
function m = weighted_mean(x, obs_weight)
if isempty(obs_weight)
    m = mean(x);
else
    m = sum(obs_weight.*x) / sum(obs_weight);
end
end

%% list of all possible metrics
function metrics =  metrics_list(varargin)
metrics = {'Dataset','LogPrior','LogLikelihood','LogJoint','AIC','AICc','BIC','LogEvidence',...
    'accuracy','ExplainedVariance','PredictorVariance','PredictorExplainedVariance','validationscore','r2','FittingTime'};
end

%% PLOT SINGLE WEIGHTS
function [h_nu, c, cm] = plot_single_weights(W, rk, c, cm, cols, colmaps)
if isa(W.plot, 'function_handle')
    %% custom-defined plot
    h_nu = W.plot(W);
    return;
end

title(W.label);

if isempty(W.scale)
    W.scale = 1:W.nWeight;
end
scale = W.scale';

% error bar values
U = W.PosteriorMean'; % concatenate over ranks
if isempty(W.PosteriorStd)
    se = nan(size(U));
else
    se = W.PosteriorStd';
end

%nU = size(U,1);
%nScale = length(scale);
%if size(U,2)>1
%else
%   nCurve = nU/nScale;
%   if nCurve>1
%       U = reshape(U, nScale,nCurve);
%       se = reshape(se, nScale,nCurve);
%   end
%end

if size(scale,2)>1
    if size(scale,2)>2
        warning('more than 2 dimensions for weights, ignoring extra dimensions');
    end

    [scale, U, se] = mesh_to_matrix(scale(:,1:2), U, se);
end
if ~iscell(scale)
    scale = {scale};
end


%% plotting options
plot_opt = {};

% fopts = fieldnames(W.plot);
% for ff=1:length(fopts)
%    fopts{end+1:end+2} = {fops{f}, W.plot.(fopts{f})};
% end
%  else

if W.nWeight < 8
    plot_opt = {'bar'};
end

% default color
plot_opt{end+1} = 'Color';

% select color map
twodmap = size(scale{1},2)>1;
nCurve = size(U,2);

if nCurve<5 % different colors for each curve

    plot_opt{end+1} = cols(mod(c+(1:nCurve),length(cols))+1);
    if rk ==1
        c = c + nCurve;
    end
else % use color map
    plot_opt{end+1} = colmaps{cm};
    cm = cm + 1;
end
imageplot = nCurve>9;

if ~isempty(W.plot)
    plot_opt = [plot_opt W.plot];
end

%% plot
if twodmap
    % color plot
    h_nu = colorplot(scale{1}(:,1),scale{1}(:,2), real(U));
elseif imageplot
    % image plot
    h_nu = imagescnan(scale{2},scale{1}, U);
else
    % cufve/bar plot with errors
    [~,~,h_nu] = wu([],U,se,scale,plot_opt{:});
end

%if iscell(scale)
%   [~,~,h_nu] = wu(U,se,{scale},plot_opt{:});
%    twodplot = 0;
%elseif isvector(scale)
%
%    [~,~,h_nu] = wu(scale, U,se,plot_opt{:});
%    twodplot = 0;
%else
%    if size(scale,2)>2
%      warning('more than 2 dimensions for weights, ignoring extra dimensions');
%    end
%    %2d plot for scale
%    twodplot = 1;
%    x_unq = unique(scale(:,1));
%    y_unq = unique(scale(:,2));
%    nX = length(x_unq);
%    nY = length(y_unq);
%    U2 = nan(nX,nY);
%    se2 = nan(nX,nY);
%    for iX=1:nX
%        for iY=1:nY
%        if isnumeric(scale)
%            bool = (scale(:,1) == x_unq(iX)) & (scale(:,2)==y_unq(iY));
%          else
%          bool = strcmp(scale(:,1),x_unq{iX}) & strcmp(scale(:,2),x_unq{iY});
%
%          end
%            if any(bool)
%                U2(iX,iY) = U(find(bool,1));
%                se2(iX,iY) = se(find(bool,1));
%            end
%        end
%    end

%h_nu = imagesc(y_unq,x_unq, U2);

%end

% add horizontal line for reference value
if ~imageplot && W.nWeight >= 8
    hold on;
    switch W.constraint
        case '1'
            y_hline = 1;
        case 'm'
            y_hline = 1/W.nWeight;
        otherwise
            y_hline = 0;
    end
    plot(xlim,y_hline*[1 1], 'Color',.7*[1 1 1]);
end


end

%% RECONSTRUCT MATRIX FROM MESH
function [S, U, se] = mesh_to_matrix(S, U, se, cutoff)
% [S, U, se] = mesh_to_matrix(S, U, se [,cutoff])
% to transform mesh data to matrix form (if the inoccupancy ratio is
% smaller than cutoff)

if nargin<4
    cutoff = 3;
end

% reconstruct weights as matrix
x_unq = unique(S(:,1));
y_unq = unique(S(:,2));
nX = length(x_unq);
nY = length(y_unq);

% compute ratio of total mesh points by provided points
ratio = nX*nY/size(S,1);

% if ratio is above threshold, keep data (i.e. they don't really form a
% mesh)
if ratio>cutoff
    return;
end

U2 = nan(nX,nY);
se2 = nan(nX,nY);
for iX=1:nX
    for iY=1:nY
        if isnumeric(S)
            bool = (S(:,1) == x_unq(iX)) & (S(:,2)==y_unq(iY));
        else
            bool = strcmp(S(:,1),x_unq{iX}) & strcmp(S(:,2),x_unq{iY});

        end
        if any(bool)
            U2(iX,iY) = U(find(bool,1));
            se2(iX,iY) = se(find(bool,1));
        end
    end
end
U = U2;
se = se2;
S = {x_unq y_unq};
end
