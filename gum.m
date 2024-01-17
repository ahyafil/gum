classdef gum

    % Defines a Generalized Unrestricted Model (GUM). Two possible
    % syntaxes:
    % * function M = gum(X, y, param)
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
    % - 'label': provide label to model
    % -'observations': 'binomial' (default), 'normal', 'poisson' or
    % 'neg-binomial'
    % - 'link': link function. Default is canonical link function ('logit'
    % for binomal observations; 'identity' for normal observations; 'log' for
    % Poisson and neg-binomial observations). The other possible value is 'probit' for
    % binomial observations.
    % -'w': is a weighting vector with length equal to the number of training examples
    % - 'split': splitting the model into an array of models for subset of
    % data based on the value of a categorical variable
    % - 'constant': whether a regressor of ones is included (to capture the
    % intercept). Possible values: 'on' [default], 'off'
    %
    % - 'mixture': for mixture of GUMs: 'multinomialK', 'hmmK', 'regressionK' where K is the number of models.
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
    % cat(x1):cat(x2) generates interaction terms, cat(x1)^cat(x2)
    % generates interaction terms and main effects
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
    % Complete list of methods:
    % OPERATIONS AND TESTS ON MODEL
    % -'extract_observations': extract model for only a subset of
    % observations
    % - 'isgam': whether model corresponds to a GAM
    % - 'isglm': whether model corresponds to a GLM
    % - 'isestimated': whether model weights have been estimated
    % - 'is_weight_set': whether model weights haven been assigned
    % - 'isfitted': whether hyperparameters have been fitted
    % - 'is_infinite_covariance': whether any regressor has infinite
    % covariance (i.e. no prior)
    % - 'number_of_regressors': number of regressors
    % - 'concatenate_weights': concatenates model weights into vector
    % - 'get_weight_structure': get structure for weight data
    % - 'get_hyperparameter_structure': get structure for hyperparameter data
    % - 'freeze_weights': freezes weights and parameters in the model
    % - 'knockout': remove set of regressors from model
    % - 'clear_data': clear raw data to make lighter object
    % - 'save': save model into .mat file
    %
    % DEALING WITH MULTIPLE MODELS
    % - 'split': split model depending on value of a vector
    % - 'concatenate_over_models': concatenate weights over models
    % - 'concatenate_score': concatenate score from array of models
    % - 'population_average': averages weights over models
    %
    % NUMERICAL METHODS
    % - 'compute_rho_variance' provide the variance of predictor in output structure
    % - 'vif': Variance Inflation Factor
    % - 'sample_weights_from_prior': assign weight values sampling from
    % prior
    % - 'sample_weights_from_posterior': assign weight values sampling from
    % posterior
    % - 'Predictor': compute the predictor for each observation
    % - 'ExpectedValue': computes the expected value for each observation
    % - 'LogPrior': computes the LogPrior of the model
    % - 'LogLikelihood': computes the LogLikelihood of the model
    % - 'Accuracy': accuracy of the model predictions at MAP weights
    % - 'LogJoint': computes the Log-Joint of the model
    % - 'Hessian': computes the Hessian of the negative LogLihelihood
    % - 'PosteriorCov': computes the posterior covariance
    % - 'IRLS': core step in inference
    % - 'boostrapping': generate bootstrap estimates of weight uncertainty
    % - 'Sample': generate observations from model using MAP weights
    % - 'Sample_Observations_From_Posterior': generate observations from
    % model sampling from weight posterior
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
    % See https://github.com/ahyafil/gum
    % version 0.1.2. Bug/comments: send to alexandre dot hyafil (AT) gmail dot com

    % DONE
    % - regressor: corrected issue with splitting dimensions
    % - added constraint "zero0"
    % - added 'ref' field for categorical regressor
    % - changed normalization for exponential basis functions
    % TODO
    % - wu: automatic permute of dimensions
    % - constraints: add custom field, i.e. "mean3"
    % - basis functions: change prior to ARD? allow for chnge (L2/ARD/none)
    % - plot_variance_vs_predictor (for normal and Poisson/lognorm)
    % - test if residual is linear in one predictor (normal, expand for
    % poisson)
    % - improve indexing in sparsearray
    % - Matern prior
    % - spectral trick (marginal likelihood in EM sometimes decreases - because of change of basis?)
    % - spectral trick for periodic: test; change prior (do not use Squared
    % Exp)
    % - link functions as in glmfit
    % - test rank again
    % - prior mean function (mixed effect; fixed effect: mean with 0 covar)
    % - use fitglme/fitlme if glmm model
    % - reset: weights for gum, clearing all scores and predictions
    % - allow parallel processing for crossvalid & bootstrap
    % - add mixture object (lapses; built-in: dirichlet, multiple dirichlet, HMM, multinomial log reg)
    % - allow EM for infinite covariance:  we should remove these from computing logdet, i.e. treat them as hyperparameters (if no HP attached)

    properties
        formula = ''
        label = ''
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
        function obj = gum(M, y, param)
            if nargin==0
                %% empty class
                return;
            end
            assert(nargin>=2, 'needs at least two arguments');
            % optional parameters
            if (nargin < 3)
                param = struct;
            end

            if istable(M)
                fmla = y;% formula
                [M, y, param] = parse_formula(M,fmla, param); % parse formula
                obj.formula = fmla;
            end

            obj.regressor = M;

            % convert M to structure if needed
            if isnumeric(M)
                M = regressor(M,'linear');
            end
            assert(isa(M,'regressor'),'first argument should either be a table, a numeric data or a regressor object');
            nMod = length(M); % number of modules

            %  n = prod(n); % number of observations

            % model label
            if isfield(param,'label')
                obj.label = char(param.label);
                param = rmfield(param,'label');
            end

            % compose formula
            if isempty(obj.formula)
                for i=1:nMod
                    obj.formula = [obj.formula M(i).formula ' + '];
                end
                obj.formula(1:end-3) = [];
            end

            %% check dependent variable
            if isvector(y)
                n = length(y);
                y = y(:);
                BinaryCountCode = 0;
            elseif ~ismatrix(y) || size(y,2)~=2
                error('T should be a column vector or a matrix of two columns');
            else
                n = size(y,1);
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
                obj.score.nObservations = sum(obj.ObservationWeight);
            else
                obj.score.nObservations = obj.nObs;
            end
            obj.score.isEstimated = 0;
            obj.score.isFitted = 0;
            obj.score.Dataset = "";

            %% transform two-column dependent variable into one-column
            if BinaryCountCode % if binary observations with one column for counts of value 1 and one column for total counts
                if any(y(:,1)<0)
                    error('counts should be non-negative values');
                end
                if any(y(:,1)>y(:,2))
                    error('values in the second column should be larger or equal to values in the first');
                end
                if ~isempty(obj.ObservationWeight)
                    error('two-column dependent variable is not compatible with observation weights');
                end

                w = [y(:,1) y(:,2)-y(:,1)]'; % first row: value 1, then value 0
                nRep = sum(w>0,1); % if there is observation both for 0 and/or for 1 (needs to replicate rows)

                y = repmat([1;0],1, n); % observation for each
                y = y(w>0); % only keep
                obj.ObservationWeight = w(w>0);

                n = sum(nRep);
                for m=1:nMod
                    RR = cell(1,ndims(M(m).Data));
                    RR{1} = nRep;
                    for d=2:ndims(M(m).Data)
                        RR{d} = 1;
                    end
                    M(m).Data = repelem(M(m).Data,RR{:});
                    M(m).nObs = n;
                end
                obj.nObs = n;

            end
            obj.T = y;

            %% parse parameters
            if isfield(param,'observations')
                obs = param.observations;
                obs = strrep(obs, 'count','poisson');
                obs = strrep(obs, 'binary','binomial');
                obs = strrep(obs, 'gaussian','normal');
                obs = strrep(obs, 'NB',  'neg-binomial');
                obs = strrep(obs, 'negative binomial',  'neg-binomial');
                assert(any(strcmp(obs, {'normal','binomial','poisson','neg-binomial'})), ...
                    'incorrect observation type: possible types are ''normal'',''binomial'', ''poisson'' and ''neg-binomial''');
            else
                obs = 'binomial';
            end
            if strcmp(obs,'binomial') && any(y~=0 & y~=1)
                error('for binomial observations, T values must be 0 or 1');
            elseif ismember(obs, {'poisson','neg-binomial'}) && any(y<0)
                error('for count observations, all values must be non-negative');
            end
            if all(y==0) || all(y==1)
                warning('T values are all 0 or 1, may cause problem while fitting');
            end
            obj.obs = obs;

            %% link function
            switch obs % default link function
                case 'normal'
                    obj.link = 'identity';
                case 'binomial'
                    obj.link = 'logit';
                case {'poisson','neg-binomial'}
                    obj.link = 'log';
            end

            if isfield(param,'link') && ~isempty(param.link)
                if strcmp(obs,'binomial') && strcmpi(param.link,'probit')
                    obj.link = 'probit';
                elseif ~strcmp(param.link, obj.link)
                    error('incorrect link function for %s observations: ''%s''', obs, obj.link);
                end

            end

            %% check if prior none
            if isfield(param, 'prior') && any(strcmpi(param.prior, {'none','off'}))
                M = M.disable_prior(); % disable prior
            end

            %% add constant bias ('on' by default)
            if ~isfield(param,'constant') || strcmpi(param.constant, 'on')
                % let's see whether we should add a prior or not on this
                % extra weight
                GCov = global_prior_covariance(compute_prior_covariance(M));
                if  isempty(GCov) || any(isinf(diag(GCov))) % if any other regressor has infinite variance
                    const_prior_type = 'none';
                else
                    const_prior_type = 'L2';
                end
                clear GCov;

                % create regressor
                Mconst = regressor(ones(n,1),'linear','label',"offset", 'prior',const_prior_type);

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

            obj.param = param;
            obj.regressor = M;

            obj = obj.compute_n_parameters_df; % compute number of parameters and degrees of freedom
            obj = obj.compute_n_free_hyperparameters; % compute number of free HPs
            obj.score.scaling = 1;
            if ismember(obs, {'normal','neg-binomial'}) % free dispersion parameter
                obj.score.scaling = nan;
            end

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

        %%% DISPLAY GUM OBJECT
        function print(obj)
            disp(obj);
        end

        function disp(obj)
            if numel(obj)>1 || isempty(obj)
                % disp(obj);
                return;
            end

            fprintf([repmat('=',1,80) '\n']);
            fprintf('Generalized Unrestricted Model (GUM) object');
            if ~isempty(obj.label)
                fprintf(': %s',obj.label);
            else
                fprintf(': unnamed model');
            end
            fprintf('\n%s\n', obj.formula);
            fprintf([repmat('-',1,80) '\n']);

            if isglm(obj)
                typestr = 'GLM';
            elseif isgam(obj)
                typestr = 'GAM';
            else
                typestr = 'GUM';
            end
            fprintf('%16s: %14s     %16s: %14s\n', 'Type',typestr, 'Observations', obj.obs);
            fprintf('%16s: %14d     %16s: %14d\n', 'nRegressorBlock',obj.nObs, 'ObservationBlock', obj.nMod);

            % add scores to summary
            Sc = export_scores_to_table(obj);
            cnt = 1;
            ScoreLabel = Sc.Properties.VariableNames;
            ScoreString = '';
            ScoreValues = {};
            Integer_scores = {'df', 'nObservations','nParameters','nFreeParameters','exitflag','nFreeHyperparameters','isEstimated','isFitted'};

            for s=1:length(ScoreLabel)
                this_score = Sc.(ScoreLabel{s});
                if isscalar(this_score) || ischar(this_score)
                    ScoreString = [ScoreString '%16s: %14'];
                    if ischar(this_score)
                        ScoreString(end+1) = 's';
                    elseif any(strcmp(ScoreLabel{s},Integer_scores))
                        ScoreString(end+1) = 'd';
                    else
                        ScoreString(end+1) = 'f';
                    end
                    if mod(cnt,2)
                        ScoreString = [ScoreString '     '];
                    else
                        ScoreString = [ScoreString '\n'];
                    end
                    ScoreValues{end+1} = ScoreLabel{s};
                    ScoreValues{end+1} = this_score;
                    cnt = cnt+1;
                end

            end
            if ~mod(cnt,2)
                ScoreString = [ScoreString '\n'];
            end
            fprintf(ScoreString,ScoreValues{:});

            %%

            fprintf([repmat('=',1,80) '\n']); % line of ====

            % add weights to summary
            if obj.isestimated()
                print_weights(obj);
            else
                fprintf('Weights: not estimated\n');
            end

            % add hyperparameters to summary
            fprintf([repmat('=',1,80) '\n']); % line of ====
            obj.print_hyperparameters;

        end

        %%% PRINT WEIGHTS
        function print_weights(obj)
            % print_weights(M) prints summary table for weights

            if obj.isestimated()
                W = export_weights_to_table(obj);
                label = string(W.label);
                nReg = height(W);

                if nReg>30
                    fprintf('\nToo many regressor values (%d) to print, use export_weights_to_table instead\n', nReg);
                elseif isgam(obj)
                    fprintf('%12s %12s %8s %8s %8s %8s\n', 'Regressor','Level','Mean','StdDev','T-stat','p-value');
                    for r=1:nReg
                        fprintf(['%12s %12' char_type(W.scale) ' %8f %8f %8f %8f\n'], label(r), W.scale(r), W.PosteriorMean(r), W.PosteriorStd(r), W.T(r), W.p(r));
                    end
                else
                    fprintf('%10s %12s %12s %8s %8s %8s %8s\n', 'RegressorId','Regressor','Level','Mean','StdDev','T-stat','p-value');
                    for r=1:nReg
                        fprintf(['%10s %12s %12' char_type(W.scale) ' %8f %8f %8f %8f\n'], string(W.regressor(r)), label(r), W.scale(r), W.PosteriorMean(r), W.PosteriorStd(r), W.T(r), W.p(r));
                    end
                end

            else
                fprintf('weights have not been estimated, cannot print summary\n');
            end
        end


        %%% PRINT HYPERPARAMETERS
        function print_hyperparameters(obj)
            % print_weights(M) prints summary table for weights

            H = export_hyperparameters_to_table(obj);
            nHP = height(H);
            if nHP==0
                fprintf('The model has no hyperparameter.\n');
                return;
            end
            if isgam(obj)

                fprintf('%20s %20s %8s %8s %8s %8s\n', 'Regressor','HP','value','fittable','LowerBnd','UpperBnd');
                for r=1:nHP
                    if iscell(H.label)
                        this_label = H.label{r};
                    else
                        this_label = H.label(r);
                    end
                    fprintf('%20s %20s %8f %8d %8f %8f\n', H.transform{r}, this_label, H.value(r), H.fittable(r), H.LowerBound(r), H.UpperBound(r));
                end
            else
                fprintf('%10s %20s %12s %8s %8s %8s %8s\n', 'RegressorId','Regressor','Level','value','fittable','LowerBnd','UpperBnd');

                for r=1:nHP
                    if iscell(H.label)
                        this_label = H.label{r};
                    else
                        this_label = H.label(r);
                    end
                    fprintf('%10d %20s %12s %8f %8d %8f %8f\n', H.RegressorId(r), H.transform(r), this_label, H.value(r), H.fittable(r), H.LowerBound(r), H.UpperBound(r));
                end
            end

        end


        %% SELECT SUBSET OF OBSERVATIONS FROM MODEL
        function obj = extract_observations(obj,subset)
            % M = extract_observations(M, subset) generates a model with
            % only observations provided by vector subset. subset is either
            % a vector of indices or a boolean vector indicating
            % observations to be included.

            obj.T = obj.T(subset);
            n_obs = length(obj.T);
            obj.nObs = n_obs;

            if ~isempty(obj.ObservationWeight)
                obj.ObservationWeight = obj.ObservationWeight(subset);
                obj.score.nObservations = sum(obj.ObservationWeight);
            else
                obj.score.nObservations = n_obs;
            end


            for m=1:length(obj.regressor) % for each module
                obj.regressor(m) = extract_observations(obj.regressor(m), subset);
                %  obj.regressor(m).Data =   extract_observations(obj.regressor(m).Data,subset); % extract for this module
                %  obj.regressor(m).nObs = n_obs;
            end

            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'rho') && ~isempty(obj.Predictions.rho)
                obj.Predictions.rho = obj.Predictions.rho(subset);
            end
            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'Expected') && ~isempty(obj.Predictions.Expected)
                obj.Predictions.Expected = obj.Predictions.Expected(subset);
            end
            if ~isempty(obj.Predictions) && isfield(obj.Predictions, 'sample') && ~isempty(obj.Predictions.rho)
                obj.Predictions.sample = obj.Predictions.rho(sample);
            end

            % update number of parameters and degrees of freedom
            obj = compute_n_parameters_df(obj);

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

                objS(v).score.Dataset = string(V(v)); % give label to dataset
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
            %
            % - 'gradient' (for crossvalidation fitting): whether to use gradients of CVLL to speed up search.
            % Gradients computation may suffer numerical estimation problems if
            % covariance are singular or close to singular. Possible values are: 'off'
            % [default], 'on' and 'check' (uses gradient but checks first that
            % numerical estimate is correct)
            %
            % - 'crossvalidation': a scalar defining the number of folds
            % (default:5), a cvpartition object or a cell array with two
            % columns and a number of rows defining the number of folds,
            % with the first column defining the indices of the training
            % set and the second column defining the indices of the test
            % set
            % -'CovPriorJacobian': provide gradient of prior covariance matrix w.r.t hyperparameters
            % to compute gradient of MAP LLH over hyperparameters.
            % - 'maxiter_HP': integer, defining the maximum number of iterations for
            % hyperparameter optimization (default: 200)
            % - 'TolFun': tolerance criterion for stopping optimization
            % - 'verbose':
            % display information for optimzation. Possible
            % values: 'full','on' [default for 'em'],'off','iter'[default for other methods]
            % - 'no_fitting': if true, does not fit parameters, simply provide as output variable a structure with LLH and
            % accuracy for given set of parameters (values must be provided in field
            % 'U')

            % default values
            HPfit = 'em'; % default algorithm
            verbose = 'iter'; % default verbosity
            use_gradient = 'on'; % use gradient for CV fitting
            maxiter = 200; % maximum number of iterations
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
                    % dd = M(m).nDim; % dimension in this module
                    M(m).ordercomponent = true;
                    %  M(m).ordercomponent = all(all(cc(:,1:dd)==cc(1,1:dd))); % default: reorder if all components have same constraints
                end
            end

            % level of verbosity
            if isfield(param, 'verbose')
                obj.param.verbose = param.verbose;
            else
                obj.param.verbose = 'on';
            end
            if isfield(param, 'verbose')
                verbose = obj.param.verbose;
            elseif isfield(param, 'display')
                verbose = obj.param.display;
            end


            %%  check fitting method
            if isfield(param, 'HPfit') % if specified as parameter
                HPfit = param.HPfit;
                assert(ischar(HPfit) && any(strcmpi(HPfit, {'em','cv','basic'})), 'incorrect value for field ''HPfit''');
            end
            if strcmp(verbose,'full') && ~strcmp(HPfit,'em')
                warning('''full'' is not an option for ''verbose'' for ''%s'' method, changing to ''iter''', HPfit);
                verbose = 'iter';
                obj.param.verbose = 'iter';
            end

            if isfield(param, 'gradient')
                use_gradient = param.gradient;
                assert(ischar(use_gradient) && any(strcmpi(use_gradient, {'on','off','check'})), 'incorrect value for field ''gradient''');
            end
            if strcmp(HPfit, 'cv') && ismember(use_gradient,{'on','check'})
                % check that there is no weights with basi functions and
                % constraint (gradient not coded for this case, should
                % change covariance_free_basis)
                W = [M.Weights];
                Basis_and_Constraint = ~cellfun(@isempty, {W.basis}) & ~cellfun(@(x) isequal(x,"free"), {W.constraint});
                if any(Basis_and_Constraint)
                    iW = find(Basis_and_Constraint,1);
                    warning('Gradient of CVLL not coded for weights with basis functions and constraints (%s), switching to gradient-less optimization',W(iW).label);
                    use_gradient = 'off';
                end
            end

            %%  check cross validation parameters
            if isfield(param, 'crossvalidation')
                obj.param.crossvalidation =  param.crossvalidation;
            elseif strcmpi(HPfit,'cv')
                % if asked for cross-validation but did not provide CVset
                obj.param.crossvalidation = .8; % use 10-fold CVLL with 80% data in training and 20% in test set
            end
            obj = check_crossvalidation(obj);

            %% process hyperparameters (which are to be fitted, bounds, initial values)
            HPall = [M.HP]; % concatenate HP structures across regressors
            HPini = [HPall.HP]; %hyperparameters initial values
            HP_LB = [HPall.LB]; % HP lower bounds
            HP_UB = [HPall.UB]; % HP upper bounds
            HP_fittable = logical([HPall.fit]);  % which HP are fitted
            if strcmpi(HPfit,'em')
                W = [M.Weights];
                basis = [W.basis];
                if any(contains([HPall.type],'basis') & HP_fittable) && ~isempty(basis) && any([basis.fixed])% if  any fittable HP parametrizes basis functions
                    warning('Log-evidence may not increase at every iteration of EM when fitting hyperparameters controlling basis functions. If log-evidence decreases, take results with caution and consider using crossvalidation instead.');
                end
            end

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
            if isfield(param, 'maxiter')
                maxiter = param.maxiter;
                param.maxiter_HP = param.maxiter;
                param = rmfield(param, 'maxiter');
            else
                param.maxiter_HP = maxiter;
            end
            if isfield(param, 'TolFun')
                HP_TolFun = param.TolFun;
                param.TolFun_HP = param.TolFun;
                param = rmfield(param, 'TolFun');
            else
                param.TolFun_HP = HP_TolFun;
            end
            if isfield(param, 'initialpoints')
                obj.param.initialpoints = param.initialpoints;
            end

            %% apply fitting method
            switch lower(HPfit)
                case 'basic' % grid search  hyperpameters that maximize marginal evidence

                    % clear persisitent value for best-fitting parameters
                    gum_neg_marg();

                    % run gradient descent on negative marginal evidence
                    errorscorefun = @(HP) gum_neg_marg(obj, HP, HPidx);
                    optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'display',verbose,'MaxIterations',maxiter);
                    HP = fmincon(errorscorefun, HPini,[],[],[],[],HP_LB,HP_UB,[],optimopt); % optimize

                    %% run estimation again with the optimized hyperparameters to retrieve weights
                    [~, obj] = errorscorefun(HP);

                case 'em' % expectation-maximization to find  hyperpameters that maximize marginal evidence
                    obj = em(obj, HPini,HPidx, use_gradient, maxiter, HP_TolFun);

                case 'cv' % gradient search to minimize cross-validated log-likelihood
                    % clear persistent value for best-fitting parameters
                    cv_score();

                    objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient
                    check_grad = strcmpi(use_gradient,'check'); % whether to use gradient

                    % run gradient descent
                    errorscorefun = @(P) cv_score(obj, P, HPidx, 0);
                    optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                        'CheckGradients',check_grad,'display',verbose,'MaxIterations',maxiter);
                    HP = fmincon(errorscorefun, HPini,[],[],[],[],HP_LB,HP_UB,[],optimopt); % optimize

                    %% run estimation again with the optimized hyperparameters to retrieve weights
                    obj =  cv_score(obj, HP, HPidx, 1);
            end

            obj.score.isFitted = 1;
        end

        %%% EM ALGORITHM
        function obj = em(obj, HPini,HPidx, use_gradient,maxiter, HP_TolFun)
            % M = em(M, HPini,HPidx, use_gradient,maxiter) runs the EM algorithm for
            % fitting. Call it through M = M.fit();

            if any([obj.regressor.rank]>1)
                error('EM not coded yet for rank larger than one');
            end

            old_logjoint = -Inf;
            logjoint_iter = zeros(1,maxiter);
            keepIterating = true;
            iter = 0;
            HP = HPini; % initial values for hyperparameters

            prm = obj.param;
            prm.HPfit = 'none';
            prm.crossvalidation = [];
            prm.originalspace = false;
            if ~isfield(prm,'initialpoints')
                prm.initialpoints = 10;
            end

            param_tmp = obj.param;

            obj.param = prm;
            vbs = prm.verbose;

            objective_grad = any(strcmpi(use_gradient, {'on','check'})); % whether to use gradient

            while keepIterating % EM iteration
                %% E-step (running inference with current hyperparameters)

                %  M = set_hyperparameters(M, HP, HPidx);

                obj.regressor = set_hyperparameters(obj.regressor, HP, HPidx);

                % evaluate covariances at for given hyperparameters
                % obj.regressor = compute_prior_covariance(M);
                obj.regressor = compute_prior_covariance(obj.regressor);

                check_grad = strcmpi(use_gradient,'check') && iter==0; % whether to use gradient (only at first iteration)


                % make sure that all weigths have real-valued priors (not infinite)
                if iter==0

                    if is_infinite_covariance(obj)
                        error('infinite covariance (probably some regressor have no prior), cannot use the ''em'' algorithm for fitting');
                        %% instead we should remove these from computing logdet, i.e. treat them as hyperparameters (if no HP attached)
                    end

                    if ~strcmp(vbs,'off')
                        fprintf('Initial weight inference (%d starting points):', prm.initialpoints);
                    end
                    switch vbs
                        case 'full'
                            prm.verbose = 'on';
                        case 'on'
                            prm.verbose = 'little';
                        otherwise
                            prm.verbose = 'off';
                    end
                elseif strcmp(vbs, 'full')
                    prm.verbose = 'little';
                else
                    prm.verbose = 'off';
                end

                if iter==1
                    prm.initialpoints = 1;
                end

                % inference
                obj = obj.infer(prm);

                % PP = projection_matrix_multiple(obj.regressor); % projection matrix for each dimension

                %% M-step (adjusting hyperparameters)
                regressorCounter = 0;
                HPidx_cat = [HPidx{:}]; % concatenate over components
                HPcounter = 1;
                for m=1:obj.nMod % for each regressor object
                    ss = obj.regressor(m).nFreeParameters; % size of each dimension

                    for d=1:obj.regressor(m).nDim % for each dimension
                        for r=1:size(HPidx{m},1)
                            this_HPidx = HPidx{m}{r,d}; % indices of hyperparameters for this set of weight
                            if ~isempty(this_HPidx)
                                if ~any(ismember(this_HPidx, [HPidx_cat{setdiff(1:obj.nMod,HPcounter)}])) %
                                    % if hyperparameters are not used in any other module

                                    % set hyperparameter values for this component
                                    HPs = obj.regressor(m).HP(r,d);
                                    HP_fittable = HPs.fit;
                                    HPs.HP(HP_fittable) = HP(this_HPidx); % fittable values

                                    % posterior mean and covariance for associated weights
                                    regressorIndex = (1:ss(r,d)) + sum(ss(:,1:d-1),'all') + sum(ss(1:r-1,d)) + regressorCounter; % index of regressors in design matrix

                                    this_cov =  obj.score.FreeCovariance(regressorIndex,regressorIndex) ; % free posterior covariance for corresponding regressor

                                    % if project on a
                                    % hyperparameter-dependent
                                    % basis, move back to original
                                    % space
                                    W = obj.regressor(m).Weights(d); % corresponding set of weight

                                    this_mean =  W.PosteriorMean(r,:);
                                    this_scale = W.scale;
                                    % this_P = PP{m}{r,d};
                                    ct = W.constraint;
                                    this_Prior = obj.regressor(m).Prior(r,d);
                                    this_PriorMean = this_Prior.PriorMean;
                                    if  ~isa(this_Prior.CovFun, 'function_handle') && ~iscell(this_Prior.CovFun) % function handle
                                        error('hyperparameters with no function');
                                    end

                                    B = W.basis; % here we're still in projected mode
                                    non_fixed_basis = any(contains(HPs.type,'basis') & HPs.fit) && ~all([B.fixed]); % if  any fittable HP parametrizes basis functions

                                    if ~isempty(B)
                                        if iscell(B)
                                            error('not coded yet for regressor with basis functions concatenated with another regressor');
                                        end

                                        if  non_fixed_basis %  if basis functions parametrized by fittable HP
                                            % working in original
                                            % space (otherwise working
                                            % in projected space)
                                            Bmat = B(1).B;
                                            this_mean = this_mean*Bmat;
                                            this_PriorMean = this_PriorMean*Bmat;
                                            if ~isequal(W.constraint,"free") && W.constraint.nConstraint>0
                                                this_cov = W.constraint.P'* this_cov *W.constraint.P;
                                            end
                                            this_cov = Bmat' * this_cov *Bmat;

                                            % define constraint in original
                                            % space
                                            dummyR = obj.regressor(m).project_from_basis().compute_projection_matrix;
                                            ct = dummyR.Weights(d).constraint;
                                            this_cov = covariance_free_basis(this_cov,ct);
                                            %  assert(W.constraint=="free", 'EM not coded for basis functions with constraint');

                                        end
                                        % not sure why this was here,
                                        % doesn't seem right
                                        %this_P = eye(length(this_cov));
                                    end
                                    this_cov = force_definite_positive(this_cov);

                                    if ~iscell(this_Prior.CovFun)
                                        iHP = contains(HPs.type, "cov"); % covariance hyperparameters
                                        [~, gg] = this_Prior.CovFun(this_scale, HPs.HP(iHP), B);
                                        %  withOptimizer = isstruct(gg)  && isequal(this_P, eye(ss(r,d)))  && ~non_fixed_basis;
                                        withOptimizer = isstruct(gg)  && isFreeStructure(ct)  && ~non_fixed_basis;
                                    else
                                        withOptimizer = false;
                                    end
                                    if  withOptimizer % function provided to optimize hyperparameters
                                        %  work on this to also take into account constraints (marginalize in the free base domain)

                                        HPnew = gg.EM(this_mean,this_cov); % find new set of hyperparameters
                                    else % find HP to maximize cross-entropy between prior and posterior

                                        optimopt = optimoptions('fmincon','TolFun',HP_TolFun,'SpecifyObjectiveGradient',objective_grad,...
                                            'CheckGradients',check_grad,'display','off','MaxIterations',1000);

                                        % find HPs that minimize negative
                                        % cross-entropy
                                        mstep_fun = @(hp) mvn_negxent(this_Prior.CovFun, this_PriorMean, this_scale, this_mean, this_cov,ct, hp, HPs, B);

                                        ini_val = mstep_fun(HP(this_HPidx));

                                        assert(~isinf(ini_val) && ~isnan(ini_val), ...
                                            'M step cannot be completed, probably because covariance prior is not full rank');

                                        % compute new set of
                                        % hyperparameters that
                                        % minimize
                                        HPnew = fmincon(mstep_fun,...
                                            HP(this_HPidx),[],[],[],[],HPs.LB(HP_fittable),HPs.UB(HP_fittable),[],optimopt);
                                    end
                                    HP(this_HPidx) = HPnew;  % select values corresponding to fittable HPs

                                else
                                    error('not coded: cannot optimize over various components at same time');
                                end
                            end
                            HPcounter = HPcounter+1;
                        end

                    end
                    regressorCounter = regressorCounter + sum(ss(1,:)) * rank(m); % jump index by number of components in module
                end

                % for decomposition in basis functions, convert weights back to
                % original domain
                obj.regressor = project_from_basis( obj.regressor);

                % has converged if improvement in LLH is smaller than epsilon
                iter = iter + 1; % update iteration counter;
                LogEvidence = obj.score.LogEvidence;
                if ~strcmp(vbs,'off')
                    fprintf('HP fitting: iter %d, log evidence %f\n',iter, LogEvidence);
                end
                % HP
                converged = abs(old_logjoint-LogEvidence)<HP_TolFun;
                old_logjoint = LogEvidence;
                logjoint_iter(iter) = LogEvidence;

                keepIterating = (iter<maxiter) && ~converged;

            end
            %  obj = obj;
            % obj.regressor = M2;
            obj.param = param_tmp;

            % allocate fitted hyperparameters to each module
            obj.regressor = obj.regressor.set_hyperparameters(HP);

        end


        %% %%%%% INFERENCE (ESTIMATE WEIGHTS) %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function   obj = infer(obj, param)
            % M = M.infer(); or M = M.infer(param);
            % INFERS (ESTIMATES) WEIGHTS OF GUM USING THE LAPLACE APPROXIMATION.
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
                    if ~isfield(param, 'verbose') || ~strcmp(param.verbose, 'off')
                        fprintf('Inferring weights for model %d/%d (%s)\n', i, numel(obj), obj(i).label);
                    end
                    obj(i) = obj(i).infer(param);
                    if ~isfield(param, 'verbose') || ~strcmp(param.verbose, 'off')
                        fprintf('\n');
                    end
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
            M = M.check_prior_covariance();
            if verbose
                fprintf('done\n');
            end

            %% compute prior mean and initialize weight
            M = M.compute_prior_mean();
            M = M.initialize_weights(obj.obs);
            obj.regressor = M;
            obj = obj.compute_n_parameters_df;

            singular_warn = warning('off','MATLAB:nearlySingularMatrix');

            % check hyper prior gradient matrix has good shape
            if do_grad_hyperparam
                % nParamTot = sum([obj.regressor.nTotalParameters]);  % total number of weights
                nParamTot = obj.score.nFreeParameters;
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
                    variance{m} = zeros(nSet,M(m).rank); % variance across observations for each component

                end
                CVLL = zeros(1,nSet); % CVLL for each
                validationscore = zeros(1,nSet); % score for each (CVLL normalized by number of observations)
                accuracy = zeros(1,nSet); % proportion correct
                exitflag_CV = zeros(1,nSet); % exit flag (converged or not)
                U_CV = zeros(obj.score.nParameters,nSet);

                if do_grad_hyperparam
                    grad_hp = zeros(size(CovJacob,3),nSet); % gradient matrix (hyperparameter x permutation)

                    PP = projection_matrix_multiple(M,'all'); % global transformation matrix from full parameter set to free basis
                else
                    PP = []; % just for parfor
                end

                warning('off','MATLAB:nearlySingularMatrix');

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
                    %obj_train = IRLS(extract_observations(obj,trainset));
                    obj_train = obj.extract_observations(trainset).IRLS();

                    exitflag_CV(p) = obj_train.score.exitflag;

                    allU = concatenate_weights(obj_train);
                    U_CV(:,p) = allU;

                    score_train = obj_train.score;
                    s = score_train.scaling; % dispersion parameter

                    %compute score on testing set (mean log-likelihood per observation)
                    obj_test = obj.extract_observations(validationset); % model with test set
                    obj_test = obj_test.set_weights_from_model(obj_train); % set weights computed from training data
                    obj_test.Predictions.rho = [];
                    obj_test.score.scaling = s; % dispersion parameter estimated in training set

                    [obj_test,CVLL(p),grad_validationscore] = obj_test.LogLikelihood(); % evaluate LLH, gradient of LLH
                    accuracy(p) = Accuracy(obj_test); % and accuracy

                    nValidation = obj_test.score.nObservations; % number of observations in test set

                    validationscore(p) = CVLL(p)/ nValidation; % normalize by number of observations

                    % compute gradient of score over hyperparameters
                    if do_grad_hyperparam

                        % add contribution of free dispersion parameter to
                        % gradient of CVLL w.r.t weights
                        free_dispersion = ismember(obj.obs, {'normal','neg-binomial'});

                        PPP = PP;

                        dY_drho = inverse_link_function(obj_train,obj_train.Predictions.rho, 1); % derivative of inverse link

                        if free_dispersion
                            PPP(end+1,end+1) = 1;

                            % add derivative of test data LLH w.r.t.
                            % dispersion parameter
                            obj_test = obj_test.compute_r_squared;
                            grad_validationscore(end+1) = -nValidation/2/s + obj_test.score.SSE/2/s^2;

                            %  log-joint derived w.r.t to dispersion param and basis function hyperparameters
                            gradgrad_dispersion = compute_gradient_LLH_wrt_basis_fun_hps(obj_train.regressor, dY_drho.*prediction_error(obj_train), 1)/s^2;

                        end

                        grad_validationscore = grad_validationscore / nValidation; % gradient w.r.t each weight

                        % posterior covariance computed from train data (include dispersion parameter if free)
                        PCov = PosteriorCov(obj_train, free_dispersion);

                        % compute LLH derived w.r.t to U (free basis) and basis fun hyperparameters
                        err_train = prediction_error(obj_train); % training model prediction error
                        gradgrad_basisfun_hp = compute_gradient_LLH_wrt_basis_fun_hps(obj_train.regressor, err_train,0, dY_drho)/s;

                        this_gradhp = zeros(1,size(CovJacob,3));
                        for q=1:size(CovJacob,3) % for each hyperparameter
                            %   gradgrad = PP*CovJacob(:,:,q)'*allU'; % log-joint derived w.r.t to U (free basis) and covariance hyperparameters
                            gradgrad = CovJacob(:,:,q)'*(PP*allU'); % log-joint derived w.r.t to U (free basis) and covariance hyperparameters
                            % gradgrad = gradgrad - PP*gradgrad_basisfun_hp(:,q);  % add contribution of basis function HP (finite differences say sth different)
                            gradgrad = gradgrad - gradgrad_basisfun_hp(:,q);  % add contribution of basis function HP (finite differences say sth different)


                            if free_dispersion % add log-joint derived w.r.t to dispersion param and hyperparameters (null for cov HPs)
                                gradgrad(end+1,:) = gradgrad_dispersion(:,q);
                            end

                            gradU = - PPP' * PCov * gradgrad;% derivative of inferred parameter U w.r.t hyperparameter (full parametrization)

                            this_gradhp(q) = grad_validationscore * gradU; % derivate of score w.r.t hyperparameter
                        end

                        % add direct contribution of basis function
                        % hyperparameters
                        this_grad_basisfun_hp = compute_gradient_LLH_wrt_basis_fun_hps(obj_test.regressor, prediction_error(obj_test), 1)/s;
                        this_gradhp = this_gradhp + this_grad_basisfun_hp/ nValidation;

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

                Sfit.CrossValidatedLogLikelihood = mean(CVLL);
                Sfit.CrossValidatedLogLikelihood_all = CVLL;
                Sfit.validationscore = mean(validationscore);
                Sfit.validationscore_all = validationscore;
                Sfit.accuracy_validation = mean(accuracy);
                Sfit.accuracy_all = accuracy;
                Sfit.exitflag_CV = exitflag_CV;
                Sfit.converged_CV = sum(exitflag_CV>0); % number of permutations with convergence achieved
                if isfield(obj.param, 'testset')
                    Sfit.testscore = testscore;
                    Sfit.accuracy_test = accuracy_test;
                end
                if do_grad_hyperparam
                    obj.grad = mean(grad_hp,2); % gradient is mean of gradient over permutations
                end

            elseif do_grad_hyperparam % optimize parameters directly likelihood of whole dataset (not recommended)

                PP = projection_matrix_multiple(M,'all'); % projection matrix for each dimension

                %compute score on whole dataset (mean log-likelihood per observation)
                [obj,validationscore,grad_validationscore] = LogLikelihood(obj); % evaluate LLH
                Sfit.accuracy = Accuracy(obj); % and accuracy

                % explained variance
                nWeightedObs = obj.score.nObservation;
                % if isempty(obj.ObservationWeight)
                %     nWeightedObs = obj.nObs;
                % else
                %     nWeightedObs = sum(obj.ObservationWeight);
                % end
                Sfit.validationscore = validationscore/ nWeightedObs; % normalize by number of observations
                grad_validationscore = grad_validationscore / nWeightedObs; % gradient w.r.t each weight

                allU = concatenate_weights(M);

                PCov = PosteriorCov(obj); % posterior covariance computed from train data
                grd = zeros(size(CovJacob,3),1);
                s = obj.score.scaling; % dispersion parameter

                for q=1:size(CovJacob,3) % for each hyperparameter
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
            P = projection_matrix_multiple(M,'all'); % projection matrix for each dimension
            Sfit.covb = P'*Sfit.FreeCovariance* P; % see covariance under constraint Seber & Wild Appendix E
            % invHinvK = P'*invHinvK*P;

            % standard error of estimates
            all_se = sqrt(diag(Sfit.covb))';

            % T-statistic for the weights
            allU = concatenate_weights(M);
            allPriorMean = concatenate_weights(M,0,'PriorMean');
            all_T = (allU-allPriorMean) ./ all_se;

            % p-value for significance of each coefficient
            %all_p = 1-chi2cdf(all_T.^2,1);
            all_p = 2*normcdf(-abs(all_T));

            % distribute values of se, T, p and V to different regressors
            M = M.set_weights(all_se,[],'PosteriorStd');
            M = M.set_weights(all_T,[],'T');
            M = M.set_weights(all_p,[],'p');
            M = M.set_posterior_covariance(Sfit.covb);
            M = M.set_posterior_covariance(invHinvK,'invHinvK');

            %             midx = 0;
            %             mfidx = 0;
            %             for m=1:obj.nMod
            %                 rr = rank(m);
            %
            %                 for d=1:M(m).nDim
            %                     W = M(m).Weights(d);
            %                     nW = W.nWeight;
            %                     nFW = M(m).nFreeParameters(d);
            %                   %  W.PosteriorStd = zeros(M(m).rank, nW);
            %                   %  W.T = zeros(M(m).rank,nW);
            %                   %  W.p = zeros(M(m).rank, nW);
            %                     W.PosteriorCov =  zeros(nW, nW, M(m).rank);
            %                     if any(strcmp(W.type, {'continuous','periodic'}))
            %                         W.invHinvK =  zeros(nFW, nFW, M(m).rank);
            %                     end
            %                     for r=1:M(m).rank
            %                         idx = (1:nW) + (r-1)*nW + rr*sum([M(m).Weights(1:d-1).nWeight]) + midx; % index of regressors in design matrix
            %                       %  W.PosteriorStd(r,:) = all_se(idx);
            %                       %  W.T(r,:) = all_T(idx);
            %                       %  W.p(r,:) = all_p(idx);
            %                         W.PosteriorCov(:,:,r) = Sfit.covb(idx,idx);
            %                         if any(strcmp(W.type, {'continuous','periodic'}))
            %                                                     fidx = (1:nFW) + (r-1)*nFW + rr*sum([M(m).nFreeParameters(1:d-1)]) + mfidx; % index of regressors in design matrix
            %
            %                             W.invHinvK(:,:,r) = invHinvK(fidx,fidx);
            %                         end
            %                     end
            %                     M(m).Weights(d) = W;
            %                 end
            %
            %                 midx = midx + M(m).nTotalParameters; % jump index by number of components in module
            %                                 mfidx = mfidx + M(m).nTotalFreeParameters; % jump index by number of components in module
            %
            %              %   midx = midx + sum([M(m).Weights.nWeight]) * rr; % jump index by number of components in module
            %
            %             end

            Sfit.exitflag = Sfit.exitflag;
            Sfit.exitflag_allstarting = Sfit.exitflag_allstarting;

            %% log-likelihood and approximation to log-evidence

            % LLH at inferred params
            [obj,Sfit.LogLikelihood] = LogLikelihood(obj);
            obj = predictor_variance(obj);

            % number of free parameters
            nFreePar = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));
            obj.score.nFreeParameters = nFreePar;

            % model evidence using Laplace approximation (Bishop - eq 4.137)  -
            % requires that a prior has been defined - Eq 16
            LD = logdet(B);
            Sfit.LogEvidence = Sfit.LogLikelihood - LD/2;

            PP = projection_matrix_multiple(M);
            for m=1:nM % add part from prior
                for d=1:M(m).nDim
                    for r=1:M(m).rank % assign new set of weight to each component
                        this_P = PP{m}{r,d};
                        dif =  (M(m).Weights(d).PosteriorMean(r,:) - M(m).Prior(r,d).PriorMean)*this_P'; % distance from prior mean (projected)
                        this_cov = M(m).Prior(r,d).PriorCovariance;
                        %  this_cov = this_P * M(m).Prior(r,d).PriorCovariance * this_P'; % corresponding covariance prior
                        inf_var = isinf(diag(this_cov)); % do not include weights with infinite prior variance
                        if any(~inf_var)
                            dif = dif(~inf_var);
                            if sum(inf_var)<100 || ~issparse(dif) % faster
                                MatOptions = struct('POSDEF',true,'SYM',true); % is symmetric positive definite
                                try
                                    CovDif = linsolve(full(this_cov(~inf_var,~inf_var)),full(dif)',MatOptions);
                                catch
                                    CovDif = this_cov(~inf_var,~inf_var) \ dif';
                                end
                            else
                                CovDif = this_cov(~inf_var,~inf_var) \ dif';
                            end
                            Sfit.LogEvidence = Sfit.LogEvidence - dif*CovDif/2; % log-prior for this weight
                        end
                    end
                end
            end

            Sfit.BIC = nFreePar*log(obj.score.nObservations) -2*Sfit.LogLikelihood; % Bayes Information Criterion
            Sfit.AIC = 2*nFreePar - 2*Sfit.LogLikelihood; % Akaike Information Criterior
            Sfit.AICc = Sfit.AIC + 2*nFreePar*(nFreePar+1)/(obj.score.nObservations -nFreePar-1); % AIC corrected for sample size
            Sfit.LogJoint_allstarting = Sfit.LogJoint_allstarting; % values for all starting points

            % compute SSE and r2 for normal observations
            obj = obj.compute_r_squared;

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

            obj.score.isEstimated = true;
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
            % -'verbose': 'on' or 'off',
            % an '*' on the command line indicates every new bootstrap completed.
            %
            % [M, bootsam] = M.boostrapping(nBootstrap) provide the bootstrap sample indices,
            % returned as an n-by-nBootstrap numeric matrix, where n is the number of observation values in M.
            %
            % Note: if multiple GUM models with same number of observations are provided as input, the same bootstraps are used for all models
            %
            % See also bootci, bootstrp
            assert(isscalar(nBootstrap) && nBootstrap>0, 'nBootstrap should be a positive scalar');

            verbose = 'on';
            alpha = 0.05;
            type = 'per';

            assert(mod(length(varargin),2)==0, 'Name-Value arguments should be provided in pairs');
            for v=1:2:length(varargin)
                switch lower(varargin{v})
                    case 'alpha'
                        alpha = varargin{v+1};
                        assert(isscalar(alpha) && alpha>=0 && alpha<=1,...
                            'alpha must be a scalar between 0 and 1');
                    case 'type'
                        type = varargin{v+1};
                        assert(ischar(type) && (ismember(type,{'norm','per'})),...
                            'value for type is either ''per'' or ''norm''');
                    case 'verbose'
                        verbose = varargin{v+1};
                end
            end

            n = obj(1).nObs;
            if length(obj)>1
                assert([obj.nObs]==n, 'all models must have the same number of observations');
            end

            if strcmp(verbose,'on')
                fprintf('Computing %d boostraps for %d model(s): ',nBootstrap, length(obj));
            end

            if nargout>1
                bootsam = zeros(n, nBootstrap);
            end

            % pre-allocate boostrap weights
            U_bt = cell(1,numel(obj));
            for i=1:numel(obj)
                U_bt{i} = zeros(obj(i).score.nParameters,nBootstrap);
            end

            for p=1:nBootstrap % for each permutation

                bt_set = randi(n,1,n); % generating boostrap (sampling with replacement)
                if nargout>1
                    bootsam(:,p) = bt_set;
                end

                %fit on bootstrap data set
                vbs = obj(i).param.verbose;
                obj(i).param.verbose = 'off';
                for i=1:numel(obj)
                    obj_bt = obj(i).extract_observations(bt_set).IRLS();

                    Ucat = concatenate_weights(obj_bt);
                    U_bt{i}(:,p) = Ucat;
                end
                obj(i).param.verbose = vbs;

                if strcmp(verbose,'on')
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

            if strcmp(verbose,'on')
                fprintf('\n done!\n');
            end

        end

        %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% IMPLEMENT THE LOG-JOINT MAXIMIZATION ALGORITHM (modified IRLS)
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function  obj = IRLS(obj)

            nM = obj.nMod; % number of modules
            M = obj.regressor;
            nD = [M.nDim]; % number of dimension for each module
            D = max([M.nFreeDimensions]);
            assert(D>0, 'no free weights');

            if isempty(obj.mixture)
                nC = 1; % number of components
                idxComponent = ones(1,nM); % indices of component
            else
                nC = obj.mixture.nComponent;
                idxComponent = obj.mixture.idxComponent;
            end

            rank = zeros(1,nM);
            for m=1:nM
                rank(m) = M(m).rank;
            end

            % order of updates for each dimensions
            UpdOrder = UpdateOrder(M);

            PP = projection_matrix_multiple(M); % free-to-full matrix conversion

            % projection matrix and prior covariance for set of weights
            P = cell(nC,D); % free-to-full matrix conversion
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
                    size_mod(m,d) = [M(m).Weights(this_d(m)).nWeight];
                    nReg(d) = nReg(d) + rank(m)*size_mod(m,d); % add number of regressors for this module
                end

                for cc=1:nC
                    %idxC = idxComponent==cc;
                    % Lambda{cc,d} = P{cc,d}*Lambda{cc,d}*P{cc,d}'; %
                    % project onto free basis (done already, corrected)
                    % Lambda{cc,d} = symmetric_part(Lambda{cc,d});

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
                Pall = projection_matrix_multiple(M,'all'); % full-to-free basis

                % project onto free basis (already done!)
                %  if issparse(Kall) && issparse(Pall)
                %      Kall =  Pall*Kall*Pall'; % not even sure it wouldn't faster if we convert to full
                %  else
                %      Pall = full(Pall); % in case Pall is sparse, faster this way
                %      Kall =  Pall*Kall*Pall';
                %  end

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

                if initialpoints>1 && any(strcmp(obj.param.verbose, {'on','full'}))
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
                obj = obj.Predictor().LogJoint([],true);
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

                % dispersion parameter from exponential family
                FixedDispersion = ~ismember(obj.obs, {'normal','neg-binomial'});
                for cc=1:nC
                    pseudo_rho =  sum(weighted_fun(obj.T,cc)) / obj.score.nObservations; % mean value
                    if strcmp(obj.obs, 'neg-binomial')
                        pseudo_rho = log(pseudo_rho);
                    end
                    [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion, pseudo_rho);
                end

                FullHessianUpdate = false(1,nC);
                logFullHessianStep = 0; % determine size of full Hessian step (https://en.wikipedia.org/wiki/Backtracking_line_search)

                %% loop weight update until convergence
                while not_converged

                    old_logjoint = logjoint;

                    for cc=1:nC
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
                                end
                            elseif d==1 || ~FullHessianUpdate(cc)
                                this_rho = Phi*UU'; % predictor
                            end
                            obj.Predictions.rho(:,cc) = this_rho;

                            switch obj.link
                                case 'logit'
                                    Y = 1 ./ (1 + exp(-this_rho)); % expected mean
                                    R = Y .* (1-Y) ; % derivative wr.t. predictor
                                case 'log'
                                    Y = exp(this_rho); % rate
                                    R = Y; % for Poisson observations
                                    if strcmp(obj.obs,'neg-binomial')
                                        R = R .* (1+s*obj.T) ./(1+s*Y).^2;
                                    end
                                    R = min(R,1e10); % to avoid badly scaled hessian
                                case 'identity'
                                    Y = this_rho;
                                    R = ones(obj.nObs,1);
                                case 'probit'

                                    Y = normcdf(this_rho);

                                    norm_rho = normpdf(this_rho);
                                    sgn = sign(obj.T-.5); % convert to -1/+1
                                    Ysgn = Y;
                                    Ysgn(sgn==-1) = 1-Ysgn(sgn==-1);
                                    R = (norm_rho./Ysgn).^2 + sgn.*this_rho.*norm_rho./Ysgn;
                            end

                            % remove constant parts from projected activation
                            [rho_tilde, UconstU] = remove_constrained_from_predictor(M(idxC), this_d(idxC), this_rho, Phi, UU);

                            % compute gradient
                            %err = obj.T-Y;
                            err = prediction_error(obj,Y, this_rho);
                            if ~FullHessianUpdate(cc)
                                G = weighted_fun(R .* rho_tilde + err,cc); % inside equation 12
                            else
                                G =  weighted_fun(err,cc);
                            end

                            %  Rmat = spdiags(weighted_fun(R,cc), 0, n, n);
                            R = weighted_fun(R,cc);

                            if ~SpCode
                                Psi = Phi*P{cc,d}';
                            end

                            if FullHessianUpdate(cc)
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


                            if ~FullHessianUpdate(cc) %% update weights just along that dimension

                                % Hessian matrix on the free basis (eq. 12)
                                if SpCode
                                    sparseness = nnz(Phi)/numel(Phi);
                                    if sparseness>.1 || (sparseness>.03 && size(Phi,2)<100)
                                        % in this case this runs faster
                                        % using full representations
                                        PhiRPhi = full(Phi)'*(R.*full(Phi));
                                    else
                                        PhiRPhi = Phi'*(R.*Phi);
                                    end
                                    if ~inf_cov %finite covariance matrix
                                        H = KP{cc,d}*PhiRPhi*P{cc,d}' + s*eye(nFree(cc,d)); % should be faster this way
                                    else % any infinite covariance matrix (e.g. no prior on a weight)
                                        H = P{cc,d}*PhiRPhi*P{cc,d}' + s*precision{cc,d};
                                    end

                                else
                                    PsiRPsi = Psi'*(R.*Psi);
                                    if ~inf_cov %finite covariance matrix
                                        H = Lambda{cc,d}*PsiRPsi  + s*eye(nFree(cc,d)); % Hessian matrix on the free basis (equation 12)
                                    else
                                        H = PsiRPsi + s*precision{cc,d};
                                    end
                                end

                                % new set of weights eq.12 (projected back to full basis)
                                MatOptions = struct('POSDEF',inf_cov,'SYM',inf_cov); % is positive definite, symmetric only in one definition
                                xi = linsolve(H,B,MatOptions)' * P{cc,d};

                                %  while strcmp(obs,'poisson') && any(Phi*(Unu+[repelem(U_const(1:rank),m(d)) zeros(1,m(D+1))])'>500) % avoid jump in parameters that lead to Inf predicted rate

                                % we set the old values of regressor just to
                                % recompute the log-joint prior to changing
                                % weights
                                obj.regressor(idxC) = set_weights(M(idxC), UU, this_d(idxC));

                                for m= find(idxC)
                                    d2 = this_d(m);
                                    logprior{m}(:,d2) = LogPrior(obj.regressor(m),d2); % log-prior for this weight
                                end

                                [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion);

                                % compute log-joint
                                interim_logjoint = logjoint;
                                [obj,logjoint] = LogJoint(obj,logprior, true);

                                % step halving if required (see glm2: Fitting Generalized Linear Models
                                %    with Convergence Problems - Ian Marschner)
                                obj.Predictions.rho(:,cc) = Phi*(xi+UconstU)';
                                while iter<4 && strcmp(obj.link,'log') && any(abs(obj.Predictions.rho(:,cc))>500) % avoid jump in parameters that lead to Inf predicted rate
                                    xi = (UU+xi)/2;  %reduce step by half
                                    obj.Predictions.rho(:,cc) = Phi*(xi+UconstU)';
                                end


                                compute_logjoint = true;


                                cnt = 0;
                                diverged = false;

                                while compute_logjoint
                                    % add new set of weights to regressor object
                                    obj.regressor(idxC) = set_weights(M(idxC),xi+UconstU, this_d(idxC));

                                    for m=find(idxC)
                                        d2 = this_d(m);
                                        logprior{m}(:,d2) = LogPrior(obj.regressor(m),d2); % log-prior for this weight
                                    end

                                    [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion);

                                    % compute log-joint
                                    [obj,logjoint] = LogJoint(obj,logprior, true);

                                    compute_logjoint = (logjoint<interim_logjoint-1e-3);
                                    if compute_logjoint % if log-joint decreases,
                                        xi = (UU-UconstU+xi)/2;  %reduce step by half
                                        obj.Predictions.rho(:,cc) = Phi*(xi+UconstU)';

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

                                M(idxC) = set_weights(M(idxC), xi+UconstU, this_d((idxC)));
                                %  M = obj.regressor;
                            else
                                %% prepare for Newton step in full weight space
                                B_dummy(idxC) = set_free_weights(Udc_dummy(idxC),B', B_dummy(idxC), this_d(idxC));
                                Udc_dummy(idxC) = set_weights(Udc_dummy(idxC), UconstU, this_d(idxC));

                            end
                        end

                        %% DIRECT NEWTON STEP ON WHOLE PARAMETER SPACE
                        if FullHessianUpdate(cc)

                            UU = concatenate_weights(M(idxC)); % old set of weights
                            UconstU = concatenate_weights(Udc_dummy(idxC)); % fixed weights
                            % B = concatenate_weights(B_dummy); % gradient over all varialves
                            B = cellfun(@(x) [x{:}], B_dummy(idxC),'unif',0);
                            B = [B{:}]';

                            % compute full Hessian of Log-Likelihood (in unconstrained space)
                            Hess = Hessian(obj, c_fpd);

                            % compute hessian matrix of log-joint on the unconstrained basis
                            if ~inf_cov %finite covariance matrix
                                B = B + Kall*(Hess*(Pall*UU'));
                                H = Kall* Hess + s*eye(nFree_all(cc)); % K * Hessian matrix on the free basis
                            else % any infinite covariance matrix (e.g. no prior on a weight)
                                B = B + Hess*(Pall*UU');
                                H = Hess + s*precision_all; % Hessian matrix on the free basis
                            end

                            MatOptions = struct('POSDEF',inf_cov,'SYM',inf_cov); % is positive definite, symmetric only in one definition
                            xi = linsolve(H,B,MatOptions)' * Pall; % new set of weights (projected back to full basis)

                            Unoconst = UU-UconstU;
                            if logFullHessianStep~=0
                                % if not doing full Newton update but only
                                % fraction in the direction
                                xi = Unoconst + exp(logFullHessianStep)*(xi-Unoconst);
                            end

                            % check that we did improve log-joint
                            compute_logjoint = -1;
                            while compute_logjoint ~=0

                                if compute_logjoint>0 % first loop: do not enter
                                    % run smaller step (pseudo Armijo rule)
                                    logFullHessianStep = logFullHessianStep-1;

                                    xi = Unoconst + exp(-1)*(xi-Unoconst);
                                end

                                obj.regressor(idxC) = set_weights(M(idxC),xi+UconstU);
                                obj = Predictor(obj); % compute rho
                                [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion);
                                [obj,logprior] = LogPrior(obj); % compute log-prior
                                [obj,logjoint] = LogJoint(obj,[],true);

                                % compute again with smaller step if failed
                                % to improve log-joint and step is larger
                                % in norm than previous step (probably
                                % without full Hessian)
                                compute_logjoint = (logjoint<old_logjoint) && norm(xi-Unoconst)>norm(dU)/10;
                            end

                            if (logjoint<old_logjoint-1e-6) % full step didn't work: go back to previous weights and run
                                FullHessianUpdate = 0;
                                obj.regressor(idxC) = M(idxC);
                                obj = Predictor(obj); % compute rho

                                [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion);
                                [obj,logprior] = LogPrior(obj); % compute log-prior
                                [obj,logjoint] = LogJoint(obj,[],true);

                                old_logjoint = logjoint - 2*TolFun; % just to make sure it's not flagged as 'converged'

                                % change thresholds for switching to full
                                % hessian update to make it less likely
                                if c_fpd>0
                                    c_fpd = 10*c_fpd; % increase chances that Hessian will really be def positive
                                else
                                    c_fpd = 1e3*eps;
                                end

                            else % full Hessian step did work
                                M(idxC) = obj.regressor(idxC);

                                % try larger step for next iteration (pseudo Armijo rule)
                                logFullHessianStep = min(logFullHessianStep+1,0);
                            end
                        end

                        %  Eq 13: update scales (to avoid slow convergence) (only where there is
                        % more than one component without constraint
                        % inforce constraint during optimization )
                        % if iter<5
                        recompute = false;
                        for m=find(idxComponent==cc & nD>1)
                            free_weight_set = isFreeWeightSet(M(m));
                            for r=1:rank(m)
                                alpha = zeros(1,nD(m));
                                n_freeweights = sum(free_weight_set(r,:));
                                if n_freeweights>1
                                    this_nLP = -logprior{m}(r,free_weight_set);
                                    mult_prior_score = prod(this_nLP )^(1/n_freeweights);
                                    if mult_prior_score>0
                                        alpha(free_weight_set) = sqrt( mult_prior_score./ this_nLP); % optimal scaling factor

                                        for d=find(free_weight_set)
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
                            obj = Predictor(obj,cc);
                            [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion);
                            % if ~FixedDispersion
                            %     s = unbiased_mse(obj.T - obj.Predictions.rho(:,cc)); %  update scaling parameter: unbiased mse
                            %     obj.score.scaling = s;
                            % end

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
                        if ~FullHessianUpdate(cc) && ...
                                cos_successive_updates(iter)>thr_FHU(1) && consistency>thr_FHU(2) && consistency<thr_FHU(3)
                            FullHessianUpdate(cc) = true;
                        end
                        FHU(cc,iter) = FullHessianUpdate(cc);
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
                for m=1:nM

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
                if  ~extflg
                    if iter>maxiter
                        extflg = -2; % maximum number of iterations
                        msg = '\nreached maximum number of iterations (%d), log-joint:%f\n';
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
                    case {'on','full'}
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
        function obj = sample_weights_from_prior(obj)
            % M = M.sample_weights_from_prior();

            for m=1:length(obj)
                obj(m).regressor = obj(m).regressor.sample_weights_from_prior;
                return;
            end
        end

        %% sample weights from posterior (approximate)
        function obj = sample_weights_from_posterior(obj)
            % M = M.sample_weights_from_posterior();
            for m=1:length(obj)
                obj(m).regressor = obj(m).regressor.sample_weights_from_posterior;
                return;
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

            assert(obj.is_weight_set,'model weights are not defined/set/estimated');

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
                nReg = nReg + M(m).rank* M(m).Weights(D(m)).nWeight; % add number of regressors for this module
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
                bool = all(strcmp(Wtype,'linear') | strcmp(Wtype,'categorical') | strcmp(Wtype,'constant'));
            end
        end

        %% TEST IF MODEL WEIGHTS HAVE BEEN ESTIMATED
        function bool = isestimated(obj)
            % bool = isestimated(M)
            % tests if weights in model M have been estimated

            bool = false(size(obj));
            for m=1:numel(obj)
                bool(m) = obj.score.isEstimated;
            end
        end

        %% TEST IF MODEL WEIGHTS HAVE BEEN ASSIGNED (OR INFERRED)
        function bool = is_weight_set(obj)
            % bool = isestimated(M)
            % tests if weights in model M have been set (estimated or
            % pre-assigned)

            bool = false(size(obj));
            for m=1:numel(obj)
                W = [obj(m).regressor.Weights];
                bool(m) = ~any(cellfun(@isempty,{W.PosteriorMean}));
            end
        end

        %% TEST IF MODEL HYPERPARAMETERS HAVE BEEN FITTED
        function bool = isfitted(obj)
            % bool = isfitted(M)
            % tests if hyperparameters in model M have been fitted

            bool = false(size(obj));
            for m=1:numel(obj)
                bool(m) = obj.score.isFitted;
            end
        end

        %% TEST IF ANY REGRESSOR HAS INFINITE COVARIANCE
        function bool = is_infinite_covariance(obj)
            % bool = is_infinite_covariance(M) tests if any regressor in
            % model M has infinite prior covariance

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
        function f = inverse_link_function(obj,rho,isDerivative)
            % f = inverse_link_function(M) returns handle to inverse link
            % function.
            % a = inverse_link_function(M, rho) returns values of inverse
            % link function evaluated at points rho
            %
            %  a = inverse_link_function(M, rho, 1) returns the derivative
            %  of the inverse link w.r.t rho instead

            if nargin<3 || ~isDerivative
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
            else
                % return derivative instead
                switch obj.link
                    case 'identity'
                        f = @(x) ones(size(x));
                    case 'logit'
                        f = @(x) exp(-x)./(1+exp(-x)).^2;
                    case 'log'
                        f = @exp;
                        assert(~strcmp(obj.obs, 'neg-binomial'), 'need to adapt the formula to prediction error for NB');

                    case 'probit'
                        f = @normpdf;
                        error( 'need to adapt the formula to prediction error for probit');
                end

            end

            if nargin>1 && ~isempty(rho) % evaluate at data points
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
                        R = Y .* (1-Y) ; % derivative wr.t. predictor
                    case 'probit'
                        % see e.g. Rasmussen 3.16
                        n = normpdf(rho);
                        sgn = sign(obj.T-.5); % convert to -1/+1

                        Ysgn = Y;
                        Ysgn(sgn==-1) = 1-Ysgn(sgn==-1);
                        R = (n./Ysgn).^2 + sgn.*rho.*n./Ysgn;

                    case 'log'
                        R = Y; % Poisson observation
                        if strcmp(obj.obs,'neg-binomial')
                            s = obj.score.scaling;
                            R = R .* (1+s*obj.T) ./(1+s*Y).^2;
                        end
                    case 'identity'
                        R = ones(obj.nObs,1);
                end
            end

            obj.Predictions.Expected = Y;
        end

        %% COMPUTE SUM OF SQUARED ERRORS AND R-SQUARED (FOR NORMAL OBSERVATIONS)
        function  obj = compute_r_squared(obj)

            % M = M.compute_r_squared
            % compute SSE and r2 for normal observations
            if ~strcmp(obj.obs, 'normal')
                return;
            end
            if isempty(obj.ObservationWeight)
                SSE = sum((obj.T-obj.Predictions.rho).^2); % residual error
                SST = (obj.nObs-1)*var(obj.T); % total variance
            else
                SSE = sum(obj.ObservationWeight .* (obj.T-obj.Predictions.rho).^2);
                SST = (sum(obj.ObservationWeight)-1) * var(obj.T, obj.ObservationWeight);
            end
            obj.score.SSE = SSE;
            obj.score.r2 = 1- SSE/SST; % r-squared

        end

        %% UPDATE DISPERSION PARAMETER
        function [obj,s] = compute_dispersion_parameter(obj, cc, FixedDispersion, rho)
            s = obj.score.scaling;

            if FixedDispersion
                return;
            end
            if nargin<4
                rho = obj.Predictions.rho;
            end

            % weighting by observations weights
            if isempty(obj.ObservationWeight) && isempty(obj.mixture)
                w  = [];
            elseif nC ==1
                w = obj.ObservationWeight;
            elseif isempty(obj.ObservationWeight)
                w = obj.mixture.Posterior(:,cc);
                warning('move down??');
            else
                w = obj.ObservationWeight .*obj.mixture.Posterior(:,cc);
            end


            if strcmp(obj.obs,'neg-binomial')
                % neg-binomial observations
                Y = exp(rho(:,cc));
                s(cc) = compute_dispersion_parameter_neg_binomial(obj.T, Y, w, s(cc));
            else
                %% gaussian observations

                err  = obj.T - rho(:,cc); % model error

                P = [obj.regressor.Prior];
                if all(strcmp({P.type},'none'))
                    % if no prior, we use degrees of freedom to obtained
                    % unbiased mse
                    norm_factor = obj.score.df;
                else % otherwise max likelihood
                    norm_factor = obj.score.nObservations;
                end

                % weighting by observations weights
                if isempty(w)
                    SSE = sum(err.^2);
                else
                    SSE = sum((x.*w).^2);
                end

                s(cc) =  SSE/norm_factor; %  update scaling parameter: unbiased mse
            end
            obj.score.scaling = s;
        end

        %% GENERATE SAMPLE FROM MODEL
        function [obj, smp] = Sample(obj)
            % M = M.Sample();
            % sample observations from model, using inferred weights (Posterior
            % Mean, i.e. MAP weights)
            %
            % [M, smp] = Sample(M) outputs the sample

            if numel(obj)>1
                smp = cell(size(obj));
                for i=1:numel
                    [obj(i), smp{i}] = Sample(obj(i));
                end
                return;
            end

            % retrieve expected value
            if isempty(obj.Predictions) || isempty(obj.Predictions.Expected)
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
                case 'neg-binomial'
                    r =  1 / obj.score.scaling; % p parameter of NB is inverse of dispersion
                    p = 1./(1 + obj.score.scaling*Y);
                    smp = nbinrnd(r,p);
                case 'normal'
                    smp = Y + sqrt(obj.score.scaling)*randn(obj.nObs,1); % generate from normal distribution
            end
            obj.Predictions.sample = smp;

        end

        %% GENERATE SAMPLE FROM MODEL
        function [obj, smp] = Sample_Observations_From_Posterior(obj, nRepetitions)
            % M = M.Sample_Observations_From_Posterior();
            % sample observations from model, using weights sampled from the posterior
            %
            % M = M.Sample_Observations_From_Posterior(nRepetitions) to repeat
            % the sampling process nRepetitions time (each with a new sample
            % from the posterior)
            %
            % [M, smp] = Sample_Observations_From_Posterior(M) outputs the sample

            if nargin<2
                nRepetitions=1;
            end

            if numel(obj)>1
                for i=1:numel(obj)
                    [obj(i), smp{i}] = Sample_Observations_From_Posterior(obj(i), nRepetitions);
                end
                return;
            end

            % preallocate matrix of sample (nRepetitions x nWeights)
            smp = zeros(nRepetitions, obj.nObs);
            for r=1:nRepetitions
                % sample weights from posterior
                obj2 = obj.sample_weights_from_posterior;

                % sample observations with given set of weights
                [~, smp(r,:)] = obj2.Sample();
            end

            obj.Predictions.sample_from_posterior = smp;

        end

        %% COMPUTE LOG-LIKELIHOOD
        function  [obj,LLH, grad] = LogLikelihood(obj,fit_mixture_weights)
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
            s = obj.score.scaling;

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
                    lh = obj.T.*log(Y) - Y - gammaln(obj.T+1);
                case 'neg-binomial'
                    % from Generalized Linear Models and Extensions, by
                    % Hardin & Hilbe
                    s =  obj.score.scaling;
                    one_sY = 1+s*Y;
                    r = 1/s; % r-parameter is inverse of dispersion
                    lh = obj.T.*log(s*Y./one_sY) - r*log(one_sY) + gammaln(obj.T+r) - gammaln(obj.T +1) - gammaln(r);
                case 'normal'
                    %  if isfield(obj.score, 'scaling')
                    %      s = obj.score.scaling;
                    %  else
                    %      s = var(Y);
                    %  end
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

                % prediction error: difference between target and predictor
                % (different for non-canonical link funcion)
                err = prediction_error(obj,Y, obj.Predictions.rho)';
                if ~isempty(obj.ObservationWeight)
                    err = obj.ObservationWeight' .* err;
                end

                Phi = design_matrix(obj.regressor,[], 0,false);                 % full design matrix
                grad = err * Phi / s;
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
                case {'poisson','neg-binomial'} % poisson
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


        %% COMPUTE HESSIAN OF NEG LOG-LIKELIHOOD (IN UNCONSTRAINED SPACE) - NOT NORMALIZED BY DISPERSION PARAMETER
        function [H,P] = Hessian(obj, c_fpd)
            %  H = Hessian(obj) computes the Hessian of the negative
            %  Log-Likelihood in unconstrained space of weights
            %
            % [H,P] = Hessian(obj)
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

            P = projection_matrix_multiple(M,'all');  %  projection matrix from full to unconstrained space
            P = full(P);
            H = P*H_full*P';
            H = symmetric_part(H); % ensure that it's symmetric (may lose symmetry due to numerical problems)

            if c_fpd>0
                H = force_definite_positive(H, c_fpd);
            end
        end

        %% COMPUTE POSTERIOR COVARIANCE
        function [V, B, invHinvK] = PosteriorCov(obj, withScaling)
            % V = PosteriorCov(M) computes the posterior covariance in free basis for
            % model M

            if nargin<2
                withScaling = 0;
            end

            % compute Hessian of likelihood
            [H,P] = Hessian(obj);
            s = obj.score.scaling;
            H = H/s; % re-scale with dispersion parameter

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
                inf_no_hp = cellfun(@(x) full(any(isinf(x(:)))), K) & cellfun(@isempty, {hp.HP});
                for i=inf_no_hp
                    %% !! finish this
                end
            end

            K = blkdiag(K{:}); % prior is block-diagonal

            %  Kfree = P*K*P'; % project on free basis

            % matrix multiplication messes up with infinite cov, so let's
            % correct that
            % inf_prior_weights = isinf(diag(K));
            % inf_prior_free_weights = any(P(:,inf_prior_weights),1); % free weights with infinite prior cov
            % Kfree(inf_prior_free_weights,inf_prior_free_weights) = diag(inf(1,sum(inf_prior_free_weights)));

            K_noinf = blkdiag(K_noinf{:});
            %  K_noinf_free = P*K_noinf*P'; % prior covariance in free basis
            K = symmetric_part(K);  % often not completely symmetric due to numerical errors

            % not sure I can use this if dim>1 because the
            %likelihood may no longer be convex, so this can have neg
            %eigenvalues
            %sqW = sqrtm(W);
            %B = eye(free_idx(end)) + sqW*Kfree*sqW; % formula to computed marginalized evidence (eq 3.32 from Rasmussen & Williams 2006)
            nFreeParameters = sum([M.nTotalFreeParameters]);
            %nFreeParameters = sum(cellfun(@(x) sum(x(:)), {M.nFreeParameters}));%nFreeParameters = sum(cellfun(@(x) sum(x,'all'), {M.nFreeParameters}));
            B = K*H + eye(nFreeParameters); % formula to compute marginalized evidence

            %       Wsqrt = sqrtm(full(H)); % matrix square root
            %   B = Wsqrt*Kfree*Wsqrt / obj.score.scaling + eye(nFreeParameters); % Rasmussen 3.26

            % I am not applying it because B is not symmetric - perhaps we
            % should if we would use the definition with Wsq (from Rasmussen)
            %   B = force_definite_positive(B);


            % compute posterior covariance
            compute_posterior_covariance = inf_terms || all([M.nFreeDimensions]<2) || withScaling;
            if compute_posterior_covariance

                % use this if no prior on some variable, or if non convex
                % likelihood, or if includes scaling parameter

                wrn = warning('off', 'MATLAB:singularMatrix');
                HH = H + inv(K);

                % extend matrix to include dispersion parameter
                if withScaling
                    obj = obj.compute_r_squared;
                    d2logjoint_ds2 = obj.score.nObservations/2/s^2 - obj.score.SSE/s^3; % second derivate of log-joint w.r.t dispersion
                    d2logjoint_ds_dU = - prediction_error(obj)' * obj.regressor.design_matrix * P' / s^2; % second derivate of LJ w.r.t dispersion and free weights

                    HH = [HH -d2logjoint_ds_dU'; -d2logjoint_ds_dU -d2logjoint_ds2]; % add to Hessian matrix
                end

                V = inv(HH);
            else
                V = B \ K; %inv(W + inv(Kfree));
                %  V = Kfree - Kfree*Wsqrt /inv(B)*Wsqrt*Kfree; % Rasmussen 3.27
            end
            V = full(V);

            % check that covariance is symmetric
            if norm(V-V')>1e-3*norm(V)
                if any(cellfun(@(MM) any(sum(isFreeWeightSet(MM),2)>1), num2cell(M)))
                    warning('posterior covariance is not symmetric - this is likely due to having two free dimensions in our regressor - try adding constraints');
                else
                    warning('posterior covariance is not symmetric - dont know why');
                end
            end

            V = symmetric_part(V); % may be not symmetric due to numerical reasons

            if nargout>2
                %% compute inv(Hinv+K) - used for predicted covariance for test datapoints
                invHinvK = inv(inv(H) + K);
            end
            if compute_posterior_covariance
                warning(wrn.state, 'MATLAB:singularMatrix');
            end
        end


        %% COMPUTE NUMBER OF PARAMETER
        function obj = compute_n_parameters_df(obj)
            % M = compute_n_parameters(M)

            % we count parameter in projected space
            M = obj.regressor.project_to_basis;

            obj.score.nParameters = sum([M.nTotalParameters]); % total number of parameters
            obj.score.nFreeParameters = sum([M.nTotalFreeParameters]); % total number of free parameters

            if isempty(obj.ObservationWeight)
                obj.score.df = obj.nObs - obj.score.nFreeParameters; % degree of freedom
            else
                obj.score.df = sum(obj.ObservationWeight) - obj.score.nFreeParameters; % degree of freedom
            end
        end

        %% COMPUTE NUMBER OF FREE HYPERPARAMETERS
        function obj = compute_n_free_hyperparameters(obj)

            HP = [obj.regressor.HP];
            isFreeHP = [HP.fit];

            obj.score.nFreeHyperparameters = sum(isFreeHP);
        end

        %% GET WEIGHT STRUCTURE
        function W = get_weight_structure(obj,label)
            % W = get_weight_structure(M) or
            % W = get_weight_structure(M, label) to get weight structure.
            % If label is a non-scalar string or label is not specified, W
            % is a structure column array with one element for each set of
            % weight.
            % If M is an array of GUM objects, W is a 2D structure array
            % where each row is for a different model.
            for i=1:numel(obj)
                if nargin>1
                    W(i,:) = get_weight_structure(obj.regressor,label);
                else
                    W(i,:) = get_weight_structure(obj.regressor);
                end
            end
        end

        %% GET HYPERPARAMETER STRUCTURE
        function H = get_hyperparameter_structure(obj,label)
            % H = get_hyperparameter_structure(M) or
            % H = get_hyperparameter_structure(M, label) to get hyperparameter structure.
            % If label is a non-scalar string or label is not specified, H
            % is a structure column array with one element for each set of
            % hyperparameters.
            % If M is an array of GUM objects, H is a 2D structure array
            % where each row is for a different model.
            for i=1:numel(obj)
                if nargin>1
                    H(i,:) = get_hyperparameter_structure(obj.regressor,label);
                else
                    H(i,:) = get_hyperparameter_structure(obj.regressor);
                end
            end
        end

        %% FREEZE WEIGHTS
        function obj = freeze_weights(obj, varargin)
            %  M = freeze_weights(M) freezes all weights and
            %  hyperparameters in model M.
            %
            % M = M.freeze_weights(regressor_label);
            % freezes weights for regressor with corresponding label
            for i=1:numel(obj)
                obj(i).regressor = freeze_weights(obj(i).regressor, varargin{:});

                % update d.f., number of parameters and HPs
                obj(i) = obj(i).compute_n_parameters_df;
                obj(i) = obj(i).compute_n_free_hyperparameters;
            end
        end

        %% KNOCK_OUT REGRESSORS
        function obj = knockout(obj, index)
            % M = M.knockout(idx);
            % "knocks-out" regressor(s) at indices in vector ind from model, i.e. removes it
            % from the model.
            %
            % M = M.knockout(label1) or M = M.knockout([label1,label2,...])
            % knocks-out regressors based on their label
            %
            % M = M.knockout('all') creates an array of models, where in
            % each model one of the regressor is knocked-out

            % M.knockout("all") syntax
            if isequal(index,'all') || isequal(index,"all")
                obj = repmat(obj,1,obj.nMod);
                for i=1:obj(1).nMod
                    obj(i) = obj(i).knockout(i); % remove regressor i
                end
                return;
            end

            if ischar(index) || isstring(index)
                I = find_weights(obj.regressor, index);
                index = unique(I(1,:)); % regressors that appear at list once in list of weights
            end

            if length(index) ==1
                obj.label = obj.regressor(index).formula + " knock-out";
            end

            % remove regressors
            obj.regressor(index) = [];
            obj.nMod = obj.nMod - 1;

            % reset metrics
            obj.score.isEstimated = 0;
            obj.score.isFitted = 0;
            obj = obj.compute_n_parameters_df; % compute number of parameters and degrees of freedom
            obj = obj.compute_n_free_hyperparameters; % compute number of free HPs
        end

        %% SET WEIGHTS AND HYPERPARAMETERS FROM ANOTHER MODEL
        function [obj,I] = set_weights_and_hyperparameters_from_model(obj, varargin)
            % M = M.set_weights_and_hyperparameters_from_model(M2);
            % sets weights and hyperparameters of model M at values of model
            % M2 (for regressors shared between the two models)
            [obj.regressor,I] = set_weights_and_hyperparameters_from_model(obj.regressor, varargin{:});
        end

        %% SET HYPERPARAMETERS FROM ANOTHER MODEL
        function [obj,I] = set_hyperparameters_from_model(obj, varargin)
            % M = M.set_hyperparameters_from_model(M2);
            % sets  hyperparameters of model M at values of model
            % M2 (for regressors shared between the two models)
            [obj.regressor,I] = set_hyperparameters_from_model(obj.regressor, varargin{:});
        end

        %% SET WEIGHTS FROM ANOTHER MODEL
        function [obj,I] = set_weights_from_model(obj, varargin)
            % M = M.set_weights_from_model(M2);
            % sets weights of model M at values of model
            % M2 (for regressors shared between the two models)
            [obj.regressor,I] = set_weights_from_model(obj.regressor, varargin{:});
        end

        %% CONCATENATE ALL WEIGHTS
        function U = concatenate_weights(obj)
            % U = concatenate_weights(M)
            % concatenates all weights from model M into a single vector
            U = concatenate_weights(obj.regressor);
        end

        %% COMPOSE MULTIDIMENSIONAL SET OF WEIGHTS
        function obj = compose_weights(obj,varargin)
            % M = M.compose_weights(lbl);
            % to compose multidimensional weights with label lbl, i.e. replaces separable
            % weight vectors U_1, U_2 ... by weight matrix/array U_1 * U_2
            % * ... (where * represents the tensor product)
            %
            %  M = M.compose_weights(); to compose all multidimensional
            %  sets of weights
            obj.regressor = compose_weights(obj.regressor, varargin{:});
        end

        %% CONCATENATE WEIGHTS OVER POPULATION
        function obj = concatenate_over_models(obj, place_first)
            % M = concatenate_over_models(M)
            % concatenates weights from array of models (e.g. same model for different datasets) into a single model object (usually for plotting
            % and storing)
            %
            % M = concatenate_over_models(M, false) to place model index as
            % last dimension in weights (default: first dimension)
            if nargin==1 % whether we put model index as first dimension in weights
                place_first = true;
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
                    obj(1).regressor(m).Weights(d).dimensions = string(obj(1).regressor(m).Weights(d).dimensions); % shouldn't be useful
                    obj(1).regressor(m).Weights(d).dimensions(end+1) = ""; % dataset

                    % check if different scales used across models
                    DifferentScale = ~all(cellfun(@(x) isequal(x,scale{1,d}) ,scale(2:end,d)));
                    DifferentScale = DifferentScale && ~all(cellfun(@(x) all(isnan(x),'all'), scale(:,d))); % if scale of nans
                    if DifferentScale && isnumeric(scale{1,d})

                        % just in case same scale with some numerical imprecisions
                        DifferentScale = ~all(cellfun(@(x) isequal(size(x),size(scale{1,d})) ,scale(2:end,d))) || ...
                            ~all(cellfun(@(x) all(abs(x(:)-scale{1,d}(:))<1e-15) ,scale(2:end,d)));
                    end
                    if  DifferentScale

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

            %% concatenate labels
            obj(1).label = {obj.label};

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

                if strcmp(mt, 'Dataset')
                    for m=1:n
                        all_score{m}.Dataset = string(all_score{m}.Dataset);
                    end
                    % pre-allocate values for each model
                    X = strings(length(all_score{1}.(mt)),length(obj));
                else
                    % pre-allocate values for each model
                    X = nan(length(all_score{1}.(mt)),length(obj));
                end



                % add values from each model where value is present
                for m=1:n
                    if ~isempty(all_score{m}) && isfield(all_score{m}, mt) && ~isempty(all_score{m}.(mt))
                        xx = all_score{m}.(mt);
                        X(:,m) =  xx;
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
            NoIncludeField =  {'PredictorVariance', 'PredictorExplainedVariance','exitflag_allstarting','LogJoint_allstarting','LogJoint_hist_allstarting','FreeCovariance','covb'};
            NoIncludeField = NoIncludeField(ismember(NoIncludeField, fieldnames(S)));
            S = rmfield(S,NoIncludeField);

            fn = fieldnames(S);
            for f = 1:length(fn) % make sure they're all row vectors
                X = S.(fn{f});
                if isvector(X) && ~ischar(X)
                    if numel(X)==numel(S.nObservations)
                        S.(fn{f}) = X(:);
                    else
                        S = rmfield(S, fn{f});
                    end
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

                    W.PosteriorMean = mean(X,dd,'omitnan'); % population average
                    W.PosteriorStd = std(X,[],dd,'omitnan')/sqrt(n); % standard error of the mean
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
                end
            end

            obj.score.PredictorExplainedVariance = PEV;
        end

        %% COMPUTE ESTIMATED VARIANCE OF PREDICTOR
        function obj = compute_rho_variance(obj)
            % M = compute_rho_variance(M)
            % computes estimated variance of predictor for each datapoint.

            n = obj.nObs;
            sigma = obj.score.covb;

            if any(isinf(sigma(:))) % if no prior defined on any dimension, cannot compute it
                obj.Predictions.rhoVar = nan(n,1);
                return;
            end

            nSample = 1000; % number of sample to compute variance
            U = concatenate_weights(obj);             % vector of posterior weights mean
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
                obj.regressor = obj.regressor.set_weights(Us);

                % compute predictor and store it
                obj = Predictor(obj);
                rho_sample(:,i) = obj.Predictions.rho;
            end

            % compute variance over samples
            obj.Predictions.rhoVar = var(rho_sample,0,2);
        end

        %% CHECK FORMAT OF CROSS-VALIDATION
        function obj = check_crossvalidation(obj)
            % M = M.check_crossvalidation;
            % process cross-validation set

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
                n = obj.nObs;

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
            % If using basis functions, VIF is computed in the space of
            % basis functions
            %
            %[V, V_free] = vif(obj,D)

            if nargin<2 % by default, project on dimension 1
                D = ones(1,obj.nMod);
            end

            % work in projected space
            obj.regressor = obj.regressor.project_to_basis();

            Phi = design_matrix(obj.regressor,[],D);

            PP = projection_matrix_multiple(obj.regressor); % free-to-full matrix projection
            P = blkdiag_subset(PP, D(:)); % projection matrix for constraints
            P = P{1};
            Phi = Phi*P';

            % remove constant column (no VIF associated)
            isConstantReg = all(Phi==Phi(1,:),1);
            if sum(isConstantReg)>1 % more than one constant column: everything is colinear
                V_free = Inf(1,size(P,1));
                V = Inf(1,size(P,2));
                return;
            end
            Phi(:,isConstantReg) = []; % remove corresponding regressor

            R = corrcoef(Phi); % correlation matrix

            V_free(~isConstantReg) = diag(inv(R))'; % VIF in free basis
            V_free(isConstantReg) = 0;

            % project back to full basis and normalize
            V = (V_free*P) ./ sum(P,1);
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
            %
            % See gum.plot_vif

            h.Axes = [];
            h.Objects = {};

            % compute design matrix
            [Phi, nReg, dims] = design_matrix(obj.regressor, varargin{:});

            M = obj.regressor;

            hold on;
            colormap(gray);
            h_im = imagesc(Phi);
            h.Objects = [h.Objects h_im];
            axis tight;

            % add vertical lines between sets of regressor
            nRegCum = cumsum([0 nReg]);
            for m=1:obj.nMod-1
                h.Objects(end+1) = plot((nRegCum(m+1)+.5)*[1 1], ylim, 'b','linewidth',2);
            end

            set(gca, 'ydir','reverse');
            f = 1;
            with_subsets = false;
            for m=1:obj.nMod
                h_txt = [];
                W = [M(m).Weights(dims{m})]; %weights structure
                for d=1:length(W)
                    if isscalar(W(d).label)
                        % add regressor label (unless there are more than
                        % one -> concatenated regressors)
                        h_txt(d) = text( mean(nRegCum(m:m+1))+.5, 0.5, W(d).label,...
                            'verticalalignment','bottom','horizontalalignment','center');
                        if nReg(f)<.2*sum(nReg)
                            set(h_txt(d),'Rotation',90,'horizontalalignment','left');
                        end
                    end

                    scl = W(d).scale;
                    if ~isempty(scl) && ~isrow(scl) && length(unique(scl(2,:)))<=size(scl,2)/2
                        % add thiner vertical lines between subsets of regressors
                        subset_change = find(scl(2,1:end-1)~=scl(2,2:end));
                        xval = nRegCum(f)+ .5 + subset_change'; % find changes of values in second line
                        plot(xval*[1 1], ylim, 'b','linewidth',1);

                        if length(h_txt)>=d
                            set(h_txt(d),'Position', get(h_txt(d),'Position') - [0 diff(ylim)/30 0]); % move main text vertically
                        end

                        subset_label = unique(scl(2,:));
                        if W(d).dimensions(end)=="index" % concatenated regressors
                            subset_label = W(d).label(subset_label);
                        end
                        xxval = [nRegCum(f)+.5 xval nRegCum(f+1)+.5];
                        for ss=1:length(subset_label)
                            text( mean(xxval(ss:ss+1)), 0.5, string(subset_label(ss)),...
                                'verticalalignment','bottom','horizontalalignment','center');
                        end

                    end
                    f = f+1;
                end


                h.Objects = [h.Objects h_txt];
            end

            axis off;

        end

        %% PLOT VIF
        function h = plot_vif(obj,D)
            % plot_vif(M) to plot Variance Inflation Factor.
            % plot_vif(M,D) to specify which dimension to use
            %
            % See gum.vif

            if nargin<2
                D = ones(1,length(obj.regressor));
            end
            obj.regressor = obj.regressor.project_to_basis;
            V= vif(obj,D);

            % put VIF values as dummy weights (and std to nan values)
            obj.regressor = obj.regressor.initialize_weights(obj.obs); % this is needed for fixed regressors
            obj.regressor = set_weights(obj.regressor, V,D);
            obj.regressor = set_weights(obj.regressor, nan(size(V)),D,'PosteriorStd');

            idx = [1:length(obj.regressor); D]; % 2-rows columns for which regressor to plot

            % now plot
            obj.score.isEstimated = true; % to allow plotting
            h = obj.plot_weights(idx);

            set_figure_name(gcf, obj, 'VIF'); % figure name
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
            if ~all(cellfun(@isempty,[M.label]))
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
            set_figure_name(gcf, obj, 'Posterior Covariance'); % figure name
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
            if ~all(is_weight_set(obj))
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
            if isnumeric(U2) && isrow(U2)
                % only providing index of regressor without dimensionality
                nDim = [obj.regressor(U2).nDim];
                U2 = repelem(U2 ,1, nDim);
                for i=1:obj.nMod
                    U2(2,U2(1,:)==i) = 1:obj.regressor(i).nDim;
                end
            end

            if length(obj)>1 % array of models

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

                % plot each model use recursive calls
                for mm=1:nObj
                    if nsub>1 % more than one subplot per model: subplots with one row per model, one column per regressor
                        for i=1:nsub
                            idx = (mm-1)*nsub + i;
                            this_h(i) = subplot(nObj, nsub, idx);
                        end
                    else % one subplot per model: all models in a grid
                        this_h = subplot2(nObj, mm);
                    end
                    h(mm) = plot_weights(obj(mm).regressor, U2, this_h);

                    % remove redundant labels and titles
                    if mm>1 && nsub>1
                        title(this_h,'');
                    end
                    if mm<nObj && nsub>1
                        xlabel(this_h, '');
                    end
                end
                set(gcf,'name',"Weights "+numel(obj)+ "models");
                return;
            end

            % if more than one dataset, pass it as extra possible label
            Dataset = string(obj.score.Dataset);
            if length(Dataset)>1 && ~all(Dataset=="")
                varargin{end+1} = 'Dataset';
                varargin{end+1} = string(obj.score.Dataset);
            elseif iscell(obj.label)
                varargin{end+1} = 'Dataset';
                varargin{end+1} = string(obj.label);
            end

            % plot regressor weights
            h = plot_weights(obj.regressor,U2, varargin{:});

            set_figure_name(gcf, obj, 'Weights');
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
                        M(m).Weights(d).label = "HP" + m +"_"+d; % default label
                    end
                    title(M(m).Weights(d).label);

                    [~,~,h_nu] = wu(M(m).HP(d).HP',[],{M(m).HP(d).label},'bar');

                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end

            set_figure_name(gcf, obj, 'Hyperparameters');
        end

        %% PLOT DATA GROUPED BY VALUES OF PREDICTOR
        function [h,Q] = plot_data_vs_predictor(obj, Q)
            % h = plot_data_vs_predictor(M) plots the predicted vs actual
            % average of the dependent variable per bins of the predictor
            % rho (i.e. posterior predictive checks). For example for
            % binary observations, the predicted relationship is the
            % logistic sigmoid function, while the actual value is the
            % number of observations equal to 1 in each bin of rho.
            %
            % h = plot_data_vs_predictor(M, Q) defines the number of
            % quantiles Q used to group values of rho. Alternatively, Q can
            % be a vector of cumulative probability values (increasing
            % values between 0 and 1) - see function quantile.
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

            H = [-inf unique(H) inf];
            nQ = length(H)-2;

            if isempty(obj.ObservationWeight)
                w = ones(obj.nObs,1);
            else
                w = obj.ObservationWeight;
            end

            M = zeros(1,nQ+1);
            sem = zeros(1,nQ+1);
            for q=1:nQ+1
                idx = rho>H(q) & rho<=H(q+1);
                nobs = sum(w(idx));
                M(q) = sum(obj.T(idx) .* w(idx)) /  nobs;
                if strcmp(obj.obs,'binomial')
                    sem(q) = sqrt( M(q)*(1-M(q)) / nobs);
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

            set_figure_name(gcf, obj, 'Data vs Model');
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
            % plot_score(M, score, ref)
            % ref provides an index to the reference model (its value is set to 0)
            %
            % plot_score(M, score, ref, labels)
            % provides labels to each model - if not provided during model
            % definition
            %
            % plot_score(M, score, ref, 'scatter') to use scatter plot
            % instead of bar plots
            %
            %  h = plot_score(...) provides graphical handles

            if nargin<3
                ref = [];
            end

            if nargin<4
                labels = {obj.label};
                cnt = 1;
                for m=1:numel(obj)
                    if isempty(labels{m})
                        labels{m} = ['unnamed' num2str(cnt)];
                        cnt = cnt+1;
                    end
                end
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
            %plot_basis_functions(M, label) to specify which regressor to
            %select
            %
            % plot_basis_functions(M, label,'normalize') or plot_basis_functions(M, [],'normalize')
            % normalizes basis functions
            %
            %h = plot_basis_functions(...) provides graphical handles
            h = plot_basis_functions(obj.regressor, varargin{:});

            set_figure_name(gcf, obj, 'basis functions');
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

for m=1:nM % for each regressor object
    if all(isFixedWeightSet(M(m))) % if all fixed weights
        UpOrder(m,:) = ones(1,D); % then we really don't care
    else
        fir = first_update_dimension(M(m)); % find first update dimension
        no_fixed_dims = find(any(~isFixedWeightSet(M(m)),1)); % find all dimensions whose weights aren't completely fixed
        fir = find(no_fixed_dims== fir);
        UpOrder(m,:) = 1 + mod(fir-1+(0:D-1),length(no_fixed_dims)); % update dimension 'fir', then 'fir'+1, .. and loop until D
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

%% check size of regressors and resize if multiple dimension observations
function M = checkregressorsize(M,n)
if M.nObs ~=n
    error('number of observations do not match between observable and regressors');
end
end

%% prediction error (to compute gradient of LLH w.r.t weights)
function err = prediction_error(obj,Y, rho)

if nargin<2
    Y = obj.Predictions.Expected;
end
if nargin<3
    rho = obj.Predictions.rho;
end

if strcmp(obj.link,'probit')
    % for probit regression
    T_signed = 2*obj.T-1; % map targets to -1/+1
    Y_signed = Y;
    Y_signed(T_signed==-1) = 1-Y_signed(T_signed==-1);
    err = T_signed .* normpdf(rho) ./ Y_signed; % see e.g. Rasmussen 3.16
elseif strcmp(obj.obs, 'neg-binomial')
    % NB regression
    s = obj.score.scaling;
    err = (obj.T-Y)./(1 + s*Y);
else
    % canonical link function, so this is truly the prediction error
    err = obj.T-Y;
end

end

%% negative marginalized evidence (for hyperparameter fitting)
function [negME, obj] = gum_neg_marg(obj, HP, idx)

persistent UU fval;
%% first call with no input: clear persistent value for best-fitting parameters
if nargin==0
    fval = [];
    UU = [];
    return;
end
first_eval = isempty(fval);
if first_eval
    fval = Inf;
end

M = obj.regressor;
if ~isempty(UU)
    M = M.set_weights(UU);
end

param = obj.param;
param.originalspace = false;
if first_eval
    switch param.verbose
        case 'iter'
            param.verbose = 'on';
        case 'on'
            param.verbose = 'little';
        otherwise
            param.verbose = 'off';
    end
elseif strcmp(param.verbose,'iter')
    param.verbose = 'little';
else
    param.verbose = 'off';
end

if  ~first_eval
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
end


%% CROSS-VALIDATED LLH SCORE
function [errorscore, grad] = cv_score(obj, HP, idx, return_obj)

persistent UU fval;
%% first call with no input: clear persistent value for best-fitting parameters
if nargin==0
    fval = [];
    UU = [];
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
if first_eval
    switch param.verbose
        case 'iter'
            param.verbose = 'on';
        case 'on'
            param.verbose = 'little';
        otherwise
            param.verbose = 'off';
    end
elseif strcmp(param.verbose,'iter')
    param.verbose = 'little';
else
    param.verbose = 'off';
end

if ~first_eval
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
    grad = obj.grad;
end

n = obj.nObs;
errorscore = -obj.score.validationscore * n; % neg-LLH (i.e. cross-entropy)
if nargout>1
    grad = -grad * n; % gradient
end

% for decomposition on basis functions, convert weights back to
% original domain
obj.regressor = project_from_basis(obj.regressor);

% if best parameter so far, update the value for initial parameters
if errorscore < fval
    fval = errorscore;
    UU = concatenate_weights(M);
end

% if return full GUM object instead of score
if return_obj
    errorscore = obj;
end

end

%% NEGATIVE CROSS-ENTROPY BETWEEN PRIOR AND POSTERIOR
% multivariate gaussians (for M-step of EM in hyperparameter optimization)
function [Q, grad] = mvn_negxent(covfun, mu, scale, m, Sigma, ct, HP, HPs, B)

HPs.HP(HPs.fit) = HP;

msgid1 = warning('off','MATLAB:nearlySingularMatrix');
msgid2 = warning('off','MATLAB:SingularMatrix');

mdif = m-mu;
noConstraint = isFreeStructure(ct);
if ~noConstraint
    P=ct.P;
    k = size(P,1); % dimensionality of free space
    mdif = mdif*P'; % difference between prior and posterior means (in free basis)
else
    k = length(mu);
end

%% prior covariance matrix and gradient w.r.t hyperparameters
if ~iscell(covfun)
    nSet =1;
    HPs.index = ones(1,length(HPs.HP));
    index_weight = ones(1,size(scale,2));
    covfun = {covfun};
else
    index_weight = scale(end,:);
    scale(end,:) = [];
    nSet = length(covfun);% number of weights sets (more than one if uses regressor concatenation)
end
K = cell(1,nSet);
gradK = cell(1,nSet);
isCovHP = contains(string(HPs.type),"cov");

for s=1:nSet

    iHP = HPs.index ==s & isCovHP; % hyperparameter for this set of weight
    if isempty(B) ||  isequal(B(s).fun, 'none')
        this_B = [];
    else
        this_B = B(s);
    end
    this_scale = scale(:,index_weight==s);

    nW = sum(index_weight==s);
    gradK{s} = zeros(nW, nW, sum(HPs.index==s));
    this_covHP = isCovHP(HPs.index==s);
    [K{s}, gg] = covfun{s}(this_scale, HPs.HP(iHP), this_B); % prior covariance matrix and gradient w.r.t hyperparameters
    if isstruct(gg)
        gg = gg.grad;
    end
    gradK{s}(:,:,this_covHP) = gg;

    if isstruct(gradK{s})
        gradK{s} = gradK{s}.grad;
    end
end

% merge over sets of weights
K = blkdiag(K{:});
gradK = blkdiag3(gradK{:});

%% deal with basis functions
if any(contains(HPs.type,'basis') & HPs.fit) && ~all([B.fixed]) % if basis functions parametrized by fittable HP, work on original space

    nHP = length(HPs.HP);

    % compute basis functions and gradient of basis fun w.r.t HP
    [B,~, gradB] = compute_basis_functions(B, B(1).scale, HPs);
    Bmat = B(1).B;

    % gradient for K for basis hyperparameter
    gradK_tmp = zeros(B(1).nWeight, B(1).nWeight,nHP);
    basisHP = find(contains(HPs.type, "basis"));

    for p=1:length(basisHP) % for all fitted hyperparameter
        BKgradB =Bmat'*K*gradB(:,:,p);
        gradK_tmp(:,:,basisHP(p)) = Bmat'*gradK(:,:,p)*Bmat + BKgradB + BKgradB';
        %gradK_tmp(:,:,p) = B.B'*gradK(:,:,p)*B.B + 2*B.B'*K*gradB(:,:,p);
    end
    gradK = gradK_tmp;

    % project prior covariance on full space
    K = Bmat'*K*Bmat;
    K = symmetric_part(K);

    % we make prior and posterior matrices full rank artificially to allow for the EM to rotate the
    % basis functions - now there's no guarantee that the LLH will increase
    % in each EM iteration, and no guarantee that it converges to a
    % meaningful result
    K = force_definite_positive(K, sqrt(min(diag(K)))*1e-3); % changing to see if we can avoid null values along diagonal
    Sigma = force_definite_positive(Sigma, sqrt(min(diag(Sigma)))*1e-3);
end

% prior and posterior covariances on free basis
[K, gradK] = covariance_free_basis(K, ct,gradK);

MatOptions = struct('POSDEF',true,'SYM',true); % is symmetric positive definite
try
    KinvSigma = linsolve(K,Sigma,MatOptions);
catch
    % if not symmetric positive despite enforcing it, HPs probably have bad
    % values
    Q = Inf;
    grad= nan(1,sum(HPs.fit));
    return;
end

LD = logdet(KinvSigma);

% negative cross-entropy (equation 19)
Kmdif = linsolve(K,mdif',MatOptions);
Q = (trace(KinvSigma) - LD + mdif*Kmdif + k*log(2*pi) )/2;

% gradient (note: gradient check may fail if K is close to singular but
% that's because of the numerical instability in computing Q, the formula
% for the gradient is correct)
grad= zeros(1,sum(HPs.fit));
cnt = 1; % HP counter
for p=find(HPs.fit) % for all fitted hyperparameter
    this_gK = gradK(:,:,p);

    %     if ~noConstraint
    %         gradJ = V/VKV*V'*this_gK*(V/VKV*V'*Kfull - eye(size(Kfull,1)));
    %         this_gK = P*(gradK(:,:,p)*J + Kfull*gradJ)*P';
    %     end
    KgK = linsolve(K,this_gK,MatOptions);
    grad(cnt) = -(  trace( KgK*(KinvSigma-eye(k))) + mdif*KgK*Kmdif )/2;
    cnt = cnt + 1;
end

warning(msgid1.state,'MATLAB:nearlySingularMatrix');
warning(msgid2.state,'MATLAB:SingularMatrix');
end

%% make matrix symmetric (correct numerical imprecisions)
function M = symmetric_part(M)
if  ~issymmetric(M)
    M = (M+M')/2;
end
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
T_fmla = trimspace(T_fmla);

if isempty(target)
    error('formula should start with dependent variable from the list of table variables');
end

T = Tbl.(target);
nObs = length(T);

% check if binary model with 'success / failure ~ ...' syntax
if ~isempty(T_fmla) && T_fmla(1) == '/'
    T_fmla = trimspace(T_fmla(2:end)); % remove / and subsequent spaces
    [target_total, T_fmla] = starts_with_word(T_fmla, VarNames);
    if isempty(target_total)
        error('incorrect variable name for cumulative number of observations in syntax '' success / total ~...''');
    end
    T = [T Tbl.(target_total)]; % add as second column
    T_fmla = trimspace(T_fmla);
end

% check if splitting model
if ~isempty(T_fmla)
    if ~T_fmla(1) == '|'
        error('dependent variable in formula should be followed either by ''~'',''/'' or ''|'' symbol');
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

option_list = {'sum','mean','tau','variance','basis','binning','constraint', 'ref','type', 'period','fit','prior'}; % possible option types
TypeNames = {'linear','categorical','continuous','periodic', 'constant'}; % possible values for 'type' options
FitNames = {'all','none','scale','tau','ell','variance'}; % possible value for 'fit' options
FitNamesLinear = {'all','none','variance'}; % possible value for 'fit' options for linear regressors
summing_opts =  {'joint', 'weighted', 'linear','equal','split','continuous'};% possible values for 'sum' options for multidim nonlinearity
        constraint_opts = ["free";"fixed";"mean1";"sum1";"nullsum";"first0";"first1"; "zero0"];

basis_regexp = {'poly([\d]+)(', 'exp([\d]+)(', 'raisedcosine([\d]+)(','gamma([\d]+)(','none(','auto(','fourier('}; % regular expressions for 'basis' options
lag_option_list = {'Lags','group','basis', 'split','placelagfirst'}; % options for lag operator

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

    is_null_regressor = false;

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
        fmla = trimspace(fmla);
        assert(~inLag, 'cannot define nested lag operators');
        inLag = true;
        LS = struct;
        LS.ini = length(O); % position of start of lag operator
    end

    transfo_label =  {'f(','cat(', 'flin(', 'fper(', 'poly[\d]+(', 'exp[\d]+(', 'raisedcosine[\d]+(', 'gamma[\d]+(',};
    [transfo, fmla, transfo_no_number] = starts_with_word(fmla,transfo_label);
    if ~isempty(transfo) % for syntax f(..) or cat(...)

        opts = struct();
        switch transfo_no_number
            case {'f(', 'poly(','exp(', 'raisedcosine(','gamma('}
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
            error('''%s'' in formula must be followed by variable name',transfo_no_number);
        end
        v = string(v); % string array

        % check for multiple variables, i.e f(x,y)
        while fmla(1)==','
            fmla(1) = [];

            [new_v, fmla] = starts_with_word(fmla, VarNames);
            assert(~isempty(new_v), ', in formula of type f(x,y) must be followed by a valid variable name');
            v(end+1) = string(new_v);

            opts.sum = 'joint';
        end

        % check for splitting variable, i.e. f(x|y)
        if fmla(1)=='|'
            fmla(1) = [];

            [new_v, fmla] = starts_with_word(fmla, VarNames);
            assert(~isempty(new_v),'| in formula f(x|y) must be followed by a valid variable name');
            opts.split = string(new_v);

        end

        %% process regressor options
        while fmla(1)~=')'
            if fmla(1) ~= separator
                error('incorrect character in formula (should be semicolon or closing parenthesis) at ''%s''', fmla);
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
                    [Num, fmla] = starts_with_number(fmla, strcmpi(option,'tau'));
                    if isnan(Num)
                        error('Value for option ''%s'' should be numeric',option);
                    end
                    opts.(option) = Num;
                case {'sum','mean'} % constraint type
                    [Num, fmla] = starts_with_number(fmla);
                    if isnan(Num)
                        if strcmpi(option,'mean')
                            error('Value for option ''sum'' should be numeric');
                        end

                        % could also be of type f(x; sum=weighted)
                        [summing, fmla] = starts_with_word(fmla, summing_opts);
                        if isnan(summing)
                            error('Value for option ''sum'' should be numeric or one of the following:''joint'', ''weighted'', ''linear'',''equal'',''split'',''continuous''');
                        end
                        opts.sum = summing;
                    else
                        if Num==0
                            opts.constraint="nullmean";
                        elseif Num==1 && strcmpi(option, 'sum')
                            opts.constraint="sum1";
                        elseif Num==1 && strcmpi(option, 'mean')
                            opts.constraint="mean1";
                        else
                            error('%s=%f: not coded yet', option, Num);
                        end
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
                    type = 'periodic';
                    i = find(fmla==')' | fmla==separator,1); % find next semicolon or parenthesis
                    if isempty(i)
                        error('incorrect formula, could not parse the value of period')
                    end
                    opts.period = eval(fmla(1:i-1)); % define period

                    fmla(1:i-1) = [];
                case 'constraint'
                    [ct, fmla] = starts_with_word(fmla, constraint_opts);
                    assert(~isempty(ct),'incorrect constraint type');
                    opts.constraint = string(ct);
                case 'ref'
                    i = find(fmla==')' | fmla==separator,1); % find next semicolon or parenthesis
                    if isempty(i)
                        error('incorrect formula, could not parse the value of ref');
                    end
                    opts.ref = fmla(1:i-1);
                    fmla(1:i-1) = [];
                case 'prior'
                    i = find(fmla==')' | fmla==separator,1); % find next semicolon or parenthesis
                    if isempty(i)
                        error('incorrect formula, could not parse the value of prior');
                    end
                    opts.prior = fmla(1:i-1);
                    fmla(1:i-1) = [];
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
        v = trimspace(fmla(2:i-1));
        fmla = trimspace(fmla(i+1:end));

        V{end+1} = struct('variable',v, 'type','constant', 'opts',struct());

    else % variable name (linear/categ)
        [v, fmla] = starts_with_word(fmla, VarNames);

        if ~isempty(v) %% linear variable

            opts = struct;
            if iscategorical(Tbl.(v)) || ischar(Tbl.(v)) || isstring(Tbl.(v))
                type = 'categorical';
            else
                type = 'linear';
            end
            fmla = trimspace(fmla);

            %%%%%%
            if ~isempty(fmla) && fmla(1)=='('
                %% process regressor options 'e.g. x(prior=none)'
                fmla(1) = ';';
                while fmla(1)~=')'
                    if fmla(1) ~= separator
                        error('incorrect character in formula (should be semicolon or closing parenthesis) at ''%s''', fmla);
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
                        case 'variance'
                            [Num, fmla] = starts_with_number(fmla,1);
                            if isnan(Num)
                                error('Value for option ''%s'' should be numeric',option);
                            end
                            opts.variance = Num;

                        case 'fit'
                            [opts.HPfit, fmla] = starts_with_word(fmla, FitNamesLinear);
                            if isempty(opts.HPfit)
                                error('incorrect fit option for variable ''%s''', v);
                            end
                        case 'prior'
                            i = find(fmla==')' | fmla==separator,1); % find next semicolon or parenthesis
                            if isempty(i)
                                error('incorrect formula, could not parse the value of prior');
                            end
                            opts.prior = fmla(1:i-1);
                            fmla(1:i-1) = [];
                    end
                end
                fmla = trimspace(fmla(2:end));

            end
            %%%%%%

            V{end+1} = struct('variable',v, 'type',type, 'opts', opts);

        elseif ~isnan(str2double(fmla(1))) %% numeric constant

            [Num, fmla] = starts_with_number(fmla);
            if Num==0 && (isempty(O) || any(O(end)=='+-'))% special case: 0 means do not include offset
                param.constant = 'off';
                is_null_regressor = true;

            else
                V{end+1} = struct('variable',Num, 'type','constant','opts',struct());
            end
            fmla = trimspace(fmla);

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
    while ~isempty(fmla) && any(fmla(1)==');')


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
                        assert( ~isnan(LS.placelagfirst), 'Value for option ''placelagfirst'' should be boolean');
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
    OperatorChars = '+*-:^;|,';
    if ~any(fmla(1)==OperatorChars)
        error('Was expecting an operator (%s) in formula at point ''%s''', OperatorChars, fmla);
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
        x = cell(1,length(w));
        for idxV = 1:length(w)
            x{idxV} = Tbl.(w(idxV));
            if idxV>1
                label = [label ',' char(w(idxV))];
            end
        end
        x = cat(2,x{:});
    else % numerical value
        x = w*ones(nObs,1);
        label = num2str(w);
    end

    opts_fields = fieldnames(V{v}.opts);
    opts_values = struct2cell(V{v}.opts);
    opts_combined = [opts_fields opts_values]';

    % condition regressor on value of split variable
    if isfield(V{v}.opts, 'split')
        split_var =  V{v}.opts.split;
        opts_combined(:,strcmp(opts_fields,'split')) = []; % remove split (not an option to rgressor constructor method)
    else
        split_var = [];
    end

    V{v} = regressor(x, V{v}.type, 'label',label, opts_combined{:});

    if ~isempty(split_var)
        V{v} = split(V{v}, Tbl.(split_var),[],[],split_var);
    end
end

%build predictor from operations between predictors
[M, idxC] = compose_regressors(V,O, OpenBracketPos, CloseBracketPos, LagStruct);

if withMixture
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

% do not include intercept if any dimension-1 polynomial included (because
% it is the 0 order polynomial
W = [M([M.nDim]==1).Weights]; % weights for dim1 regressors
if ~isempty(W)
    basis = [W.basis];
    if ~isempty(basis) && any(cellfun(@(f) isequal(f,@basis_poly), {basis.fun}))
        param.constant = 'off';
    end
end
end

%% COMPOSE REGRESSORS WITH OPERATORS
function [M, idxC] = compose_regressors(V,O, OpenBracketPos, CloseBracketPos, LagStruct)


%% process parentheses and lag operators first
while ~isempty(OpenBracketPos) || ~isempty(LagStruct) % as long as there are parenthesis or lag operators to process

    if ~isempty(OpenBracketPos)
        % select which parenthesis to process first
        iBracket = 1;
        while length(OpenBracketPos)>iBracket && OpenBracketPos(iBracket+1)<CloseBracketPos(iBracket)
            iBracket = iBracket+1; % if another bracket is opened before this one is closed, then process the inner bracket first
        end
        %  iReg = OpenBracketPos(iBracket)+1;

        % make sure there's no lag operator within this bracket
        if ~isempty(LagStruct) % if there's a lag operator...
            LSini = cellfun(@(x) x.ini,LagStruct);
            iLag = find(LSini>OpenBracketPos(iBracket) & LSini<=CloseBracketPos(iBracket),1); % ... within this bracket
            process_lag = ~isempty(iLag); %% then process it first
        else
            process_lag = false;
        end

    else % if there's no more brackets to process, then we're left with processing lags
        process_lag = true;
        iLag =1; % start with first lag operator
    end

    if process_lag
        %process lag
        LS = LagStruct{iLag};
        idx = LS.ini+1 : LS.end; % indices of regressors into the bracket
        iBracket = iLag;
        %  iReg = iLag; % not sure at all about this line
    else
        % process brackets
        idx = OpenBracketPos(iBracket)+1:CloseBracketPos(iBracket)+1; % indices of regressors into the bracket
    end


    % recursive call: compose regressors within brackets
    iReg = idx(1); % index for new regressor
    [V{iReg}, this_idxC] = compose_regressors(V(idx),O(idx(1:end-1)), [], [],[]);
    assert( all(this_idxC==1), 'incorrect formula: cannot use mixture break within brackets');


    % if possible, concatenate regressors within parenthesis into single
    % regressor (useful e.g. for later multiplying with other term) -
    % unless for x|(x2+x3)
    if (iReg==1 || O(iReg-1)~='|') && cat_regressor(V{iReg}, 1, 1)
        V{iReg} = cat_regressor(V{iReg}, 1);
        this_idxC(length(V{iReg})+1:end) = [];
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
        V{iReg} = laggedregressor(V{iReg},Lags, opts_combined{:});

        LagStruct(iBracket) = [];
    else

        % remove elements already processed
        OpenBracketPos(iBracket) = [];
        CloseBracketPos(iBracket) = [];
    end

    % remove processed regressors from pending list
    V(idx(2:end)) = []; % regressors
    O(idx(1:end-1)) = []; % operators

    % update indices for brackets after this bracket
    nRemove = length(idx)-1;
    after_this_bracket = OpenBracketPos>=idx(end);
    OpenBracketPos(after_this_bracket) = OpenBracketPos(after_this_bracket) - nRemove;
    CloseBracketPos(after_this_bracket) = CloseBracketPos(after_this_bracket) - nRemove;
    for k=1:length(LagStruct)
        if LagStruct{k}.ini>=idx(end)
            LagStruct{k}.ini = LagStruct{k}.ini - nRemove;
            LagStruct{k}.end = LagStruct{k}.end - nRemove;
        end
    end
end

%% build predictor from operations between predictors
% priority: interactions, then splitting, then product, then sum
current_idx = 1;
idxC = [];

while length(V)>1

    if any(O==',') % find comas as in x|x2,x3 and process first
        v = find(O==',',1);

        V{v} = V{v} + V{v+1}; % compose product

        % remove regressor and operation
        V(v+1) = [];
        O(v) = [];

    elseif any(O==':' | O=='^') % then process interaction

        v = find(O==':' || O=='^',1);
        if O(v) == ':'
            V{v} = V{v} : V{v+1}; % compose interaction
        else
            V{v} = V{v} ^ V{v+1}; % compose interaction (with main effects)
        end

        % remove regressor and operation
        V(v+1) = [];
        O(v) = [];
    elseif any(O=='|') % then perform splitting
        v = find(O=='|',1);

        V{v+1}.Weights.dimensions = V{v+1}.Weights.label;
        V{v} = V{v}.split(V{v+1}); % compose product

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
function  [Num, fmla] = starts_with_number(fmla, useVector)
if nargin<2
    useVector = false;
end

% if accepts various values, e.g. '2,5,10'
NumVec = [];
fmla2 = fmla;
while useVector && any(fmla2==',')
    i = find(fmla2==',',1); % first coma
    Num = str2double(fmla2(1:i-1)); % try to convert to number what comes before first coma
    if isnan(Num) % failed -> forget about that come
        useVector = false;
    else
        NumVec(end+1) = Num; % append to list
        fmla2(1:i) = []; % remove all list before
    end
end
if ~isempty(NumVec) % if we've seen digits before comas
    [Num, fmla2] = starts_with_number(fmla2); % process what comes after the last valid coma
    if ~isnan(Num)% if also valid number after last coma
        Num = [NumVec Num];
        fmla = fmla2;
    end
    return;
end

% find longest string that corresponds to numeric value
Num = nan(1,length(fmla));
for i=1:length(fmla)
    if ~any(fmla(1:i)==',') % for some reason still gives non nan when there is a coma
        Num(i) = str2double(fmla(1:i));
    end
end
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
metrics = {'nObservations','nParameters','df','Dataset','LogPrior','LogLikelihood','LogJoint','AIC','AICc','BIC','LogEvidence',...
    'accuracy','scaling','ExplainedVariance','PredictorVariance','PredictorExplainedVariance','validationscore','r2','isEstimated','isFitted','FittingTime'};
end

function c = char_type(X)
if isnumeric(X)
    c = 'f';
else
    c = 's';
end
end

%% no constraint structure
function bool = isFreeStructure(ct)
bool = isequal(ct,"free") || ct.type=="free";
end

%% figure name
function set_figure_name(gcf, obj, str)
if ~isempty(obj.label) && ~iscell(obj.label)
    str = [str ' ' obj.label];
end
if isfield(obj.score,'Dataset') && ~isempty(obj.score.Dataset) && isscalar(obj.score.Dataset)
    str = [str ' (' char(obj.score.Dataset) ')'];
end
set(gcf, 'name',str);

end
