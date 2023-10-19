classdef regressor
    % converts variable to regressors, i.e. defines function f(x) to be
    % included in GUM regression model.
    %
    % V = regressor(X, type). V should be a column vector or a matrix (possibly
    % of dimension > 2). Type indicates the data type of V, building the
    % covariance accordingly. Possible values are:
    % - 'linear' to use one regressor for V
    % - 'categorical' to use one regressor per value of V
    % - 'continuous' to use Gaussian Processes (function has prior defined by GP
    % with squared exponential kernel)
    % - 'periodic' to for periodic variable (function has prior defined by GP
    % with periodic kernel)
    % - 'constant': fixed regressor (no parameter)
    %
    % V = regressor(...,'prior',str) to use a non-default prior type. Non-default value can be:
    % - 'none' to use no-prior (i.e. infinite covariance) instead of L2-regularization
    % (for linear and categorical regressors only).
    % - 'ard' for Automatic Relevance Discrimination (i.e. one hyperparameter per regressor), for linear regressors only.
    %
    % V = regressor(..., 'binning', B) bins values of X with size B (for
    % continuous and periodic variables only) to reduce size of covariance
    % matrix. Use regressor(..., 'binning', 'auto') to ensure fast approximate
    % inference.
    %
    % V = regressor(..., 'HPfit',  F) where F can be 'all' to make all
    % regressors hyperparameteres fittable, 'none' to make none fittable,
    % 'variance' to make only variance HP fittable, 'tau' to make only tau
    % (or scale) HP fittable only (for continuous or periodic regressors
    % only).
    % F can also be boolean vector defining which
    % hyperparameters are fitted:
    % - for 'linear' regressor: one hyperparameter (L2-regularization) per dimension
    % - for 'categorical', one hyperparameter (L2-regularization)
    % - for 'continuous', three parameters of the gaussian kernel (log(scale), log(variance),
    % additive noise)
    % - for 'periodic' , two hyperparameters (log(scale)), log(standard deviation of
    % variance)). See covPeriodic from GPML.
    % V = regressor(...,'period',p) to define
    % period (default: 2pi)
    %
    % V = regressor(...,'hyperparameter', HP) is a vector defining value of hyperparameter
    % (initial value if parameter is fitted, fixed value otherwise). See field
    % 'HPfit' for description of each hyperparameter. Use NaN to use default
    % values.
    %
    % V = regressor(..., 'OneHotEncoding',bool) will use OneHotEncoding regressor if bool is
    % set to true (true by default for continuous and categorial regressors)
    %
    % V = regressor(...,'scale',sc) for categorical regressor to specify which
    % values to code (others have zero weight)
    %
    % V = regressor(...,'tau',tau) to define scale hyperparameter for
    % continuous regressor
    %
    % V = regressor(..., 'basis',B) to define a set of basis functions for
    % continuous/periodic regressor. Possible values for B are:
    % - 'polyN' for polynomials (where N is an integer, i.e. 'poly3' for 3rd order polynomial)
    % - 'expN' for exponentials
    % - 'fourier' or 'fourierN' for cosines and sines (by default with
    % Squared Exponential prior; for 'fourier' the number of basis
    % functions is determined automatically)
    % - 'raisedcosineN'
    % - 'gammaN' for gamma distribution
    %
    % V = regressor('basis','fourier','condthresh', c) to define conditional threshold for defining number of basis functions (default:1e12)
    %
    % V = regressor(...,'sum', S)
    % If X is a matrix, you may define how f(x) for each columns are summed
    % into factor. Possible values for S are:
    % - 'joint' to use multidimensional function F_n = f(X_n1,...X_nk)
    % - 'weighted' or 'linear' to assign a different weight for each column [default], i.e. to define
    % factor as the weighted sum: F_n = sum_k w_k f(X_nk)
    % - 'equal' to assign equal weight for each column, i.e. to define
    % factor as the simple sum: F_n = sum_k f(X_nk)
    % - 'split' to use different functions for each column, i.e. to define
    % factor as the sum: F_n = sum_k f_k(X_nk)
    % Different functions f_k will use the same hyperparameter set.
    % - 'continuous' to use a smooth function of sc_k where vector sc is
    % defined in scale: F_n = sum_k g(sc_k) f(X_nk)
    % - 'separate' if each function should be used to a separate set of output
    % values, i.e. to define factor as F_nk = g_k(X_nk) [!! not coded yet]
    % Different functions f_k will use the same hyperparameter set.
    %
    %
    % V = regressor(...,'constraint', C) to define constraints for the
    % different factors. C is an array of character string where character string at position d
    % defines the constraint for the corresponding dimension. Possible
    % character values are:
    % - 'free': free, unconstrained (one and only one dimension)
    % - 'nullsum': sum of weights is set to 0
    % - 'first0': first weight set to 0
    % - 'mean1': mean over weights in dimension d is set to 1
    % - 'sum1': sum over weights in dimension d is set to 1
    % - 'first1': first element in weight vector is set to 1
    % - 'fixed': fixed weights (not estimated)
    % Default value is 'fixed' (unconstrained) for first dimension, 'mean1' for other
    % dimensions. C must be a character array of length D (or rank x D matrix)
    %
    % V = regressor(...,'label', N) to add label to regressor
    %
    % Main method for regressors: .
    % (TODO ADD)
    %
    %
    %See also gum, covFunctions, timeregressor

    % add details of output field, and input variance and tau

    properties
        Data
        DataOriginal = []
        formula = ""
        nDim
        nObs
        Weights
        Prior
        HP = HPstruct()
        rank = 1
        ordercomponent = false
    end
    properties (Dependent)
        nParameters
        nFreeParameters
        nTotalParameters
    end

    methods


        %% CONSTRUCTOR %%%%%
        function obj = regressor(X,type, varargin)

            if nargin==0
                return;
            end

            % default values
            condthresh = 1e12; % threshold for spectral trick
            OneHotEncoding = [];
            scale = [];
            basis = 'auto';
            %do_spectral =0;
            period = 2*pi;
            single_tau = false;
            HPfit = [];
            HP = [];
            summing = 'weighted';
            constraint = [];
            binning = [];
            label = "";
            variance = [];
            tau = [];
            color = [];
            plot = [];
            prior = [];

            if nargin<2
                type = 'linear';
            end

            assert(any(strcmp(class(type), {'char','function_handle','cell'})),...
                'second argument should be a string array (data type) or a covariance function (function handle or cell array)');
            is_covfun = ~ischar(type);

            assert(mod(length(varargin),2)==0,'arguments should go by pair');

            v=1;
            while v<length(varargin)
                switch lower(varargin{v})
                    case 'prior'
                        prior = varargin{v+1};
                    case 'hpfit'
                        HPfit = varargin{v+1};
                    case 'hyperparameter'
                        HP = varargin{v+1};
                    case 'onehotencoding'
                        OneHotEncoding  = varargin{v+1};
                    case 'basis'
                        basis  = varargin{v+1};
                    case 'condthresh'
                        condthresh = varargin{v+1};
                    case 'scale'
                        scale = varargin{v+1};
                    case 'variance'
                        variance = varargin{v+1};
                        assert(all(variance>0),'variance hyperparameter must be positive');
                    case 'tau'
                        tau = varargin{v+1};
                    case 'period'
                        period = varargin{v+1};
                        %   case 'condthresh'
                        %       condthresh = varargin{v+1};
                    case 'singlescale'
                        single_tau = varargin{v+1};
                    case 'sum'
                        summing = varargin{v+1};
                        %   assert(any(strcmp(summing, {'weighted','linear','continuous','equal','split','separate'})), 'incorrect value for option ''sum''');
                    case 'constraint'
                        constraint = varargin{v+1};

                        % assert(ischar(constraint), 'constraint C should be a character array');
                    case 'binning'
                        binning = varargin{v+1};
                        if ~isempty(binning)
                            assert(any(strcmp(type, {'continuous','periodic'})), 'binning option only for continuous or periodic variable');
                        end
                    case 'label'
                        label = string(varargin{v+1});
                    case 'rank'
                        obj.rank = varargin{v+1};
                    case 'plot'
                        plot = varargin{v+1};
                    case 'color'
                        color = varargin{v+1};
                    otherwise
                        error('incorrect option: %s',varargin{v});
                end
                v = v +2;
            end


            %% which hyparameters are estimated
            % default values
            if is_covfun
                nPar = str2double(type()); % number of parameters
                HPfit_def = ones(1,nPar);
                HPfit_lbl = {};
            else % data type
                switch type
                    case 'constant'
                        HPfit_def = [];
                        HPfit_lbl = {};
                    case {'linear','categorical'}
                        if isempty(prior)
                            prior = 'L2';
                        end
                        switch prior
                            case 'L2'
                                if strcmp(type,'linear')
                                    HPfit_def = ones(1,ndims(X)-1);
                                else
                                    HPfit_def = 1;
                                end
                                HPfit_lbl = {'variance'};
                            case 'none'
                                HPfit_def = [];
                                HPfit_lbl = {};
                            case 'ARD'
                                HPfit_def = ones(1,size(X,ndims(X)));
                                HPfit_lbl = num2strcell('variance_%d',1:size(X,ndims(X)));
                            otherwise
                                error('incorrect prior type: %s',prior);
                        end

                    case 'continuous'
                        HPfit_def = [1 1];
                        HPfit_lbl = {'tau','variance'};
                    case 'periodic'
                        HPfit_def = [1 1];
                        HPfit_lbl = {'tau','variance'};
                    otherwise
                        error('incorrect type');
                end
            end
            if ischar(HPfit)
                switch HPfit
                    case 'all' % fit all
                        HPfit = 1;
                    case 'none' % fit none
                        HPfit = 0;
                    case 'variance'
                        HPfit = {'variance'};
                    case {'ell','tau','scale'}
                        HPfit = {'tau'};
                        assert(any(strcmp(type, {'continuous','periodic'})),'tau hyperparameter is only defined for continuous or periodic regressor');
                    otherwise
                        error('incorrect option for HPfit:%s', HPfit );
                end
            end
            if iscell(HPfit)
                assert(~is_covfun, 'cannot define names of hyperparameters to be fitted for custom covariance function')
                HPfit = ismember(HPfit, HPfit_lbl);

            elseif isempty(HPfit)
                HPfit = HPfit_def;
            elseif length(HPfit)==1
                HPfit = HPfit*ones(size(HPfit_def));
            end

            HPfit = logical(HPfit);

            if isrow(X)
                X = X';
            end

            % scale (i.e. levels)
            if isempty(scale)
                scale = {[]};
            else
                scale = tocell(scale);
            end
            summing = tocell(summing);

            % we perform one-hot-encoding for all regressor types except
            % linear - except for continuous var if scale is provided, which means the tensor
            % is already provided
            if isempty(OneHotEncoding)
                OneHotEncoding = ~strcmpi(type, 'linear') && (isempty(scale{end}) || ~strcmpi(type, 'continuous'));
            end

            % dimensionality of regressor
            if iscolumn(X)
                nD = 1;
            else
                % we'll add one dimension if one-hot encoding is performed
                % (unless we use joint function)
                add_one_dim = OneHotEncoding && ~strcmp(summing{end},'joint');
                nD = ndims(X) -1 +add_one_dim ;
                % elseif strcmpi(type, 'linear')
                %     nD = ndims(X)-1;
                % else
                %     nD = ndims(X);
            end
            obj.nDim = nD;

            % make sure scale and basis are cell array of length nD
            scale = [cell(1,nD-length(scale)) scale];
            basis = tocell(basis);
            prior = tocell(prior);
            basis = [cell(1,nD-length(basis)) basis];
            prior = [cell(1,nD-length(prior)) prior];
            summing = [repmat({''},1,nD-length(summing)) summing];
            single_tau = [false(1,nD-length(single_tau)) single_tau];

            %  obj.scale(length(scale)+1:nD) = {[]}; % extend scale cell array
            siz = size(X); % size of design tensor
            obj.nObs = siz(1); % number of observations

            %create weights structure
            obj.Weights = empty_weight_structure(nD, siz, scale, color);
            obj.Weights(nD).type = type;

            % create prior structure
            obj.Prior = empty_prior_structure(nD);

            % bin data (if requested)
            if ~isempty(binning)
                if ischar(binning) && strcmp(binning,'auto')
                    binning = (max(X(:))- min(X(:)))/100; % ensore more or less 100 points spanning range of input values
                end
                X = binning*round(X/binning); % change value of X to closest bin
            end

            % by default do not use OneHotEncoding, except one-hot-encoding dimension (if
            % categorical/continuous/periodic)
            if isempty(OneHotEncoding)
                OneHotEncoding = false(1,max(nD-1,1));
                if any(strcmp(type,{'categorical','continuous','periodic'}))
                    OneHotEncoding(nD) = true;
                end
            elseif isscalar(OneHotEncoding)
                OneHotEncoding = [false(1,nD-1) OneHotEncoding];
            end

            % whether splitting or separate regressors options are
            % requested
            SplitOrSeparate = strcmpi(summing, 'split') | strcmpi(summing, 'separate');
            SplitOrSeparate(end+1:nD) = false;

            % labels
            if isempty(label)
                label = "x" + (1:nD-1);
            end

            %% build data and variables depending on coding type
            switch lower(type)
                %% constant regressor (no fitted weight)
                case 'constant'
                    obj.Data = X;
                    obj.nDim = 1;
                    obj.Weights(1).constraint = 'fixed';
                    obj.formula = label(1);
                    %obj.Prior(1).type = 'none';

                    %% linear regressor
                case 'linear'

                    obj.Data = X;
                    obj.Data(isnan(obj.Data)) = 0; % convert nans to 0: missing regressor with no influence on model

                    % process hyperparameters
                    HP = tocell(HP,nD);
                    if ~isempty(variance), HP{1} = log(variance)/2; end

                    nWeight = siz(1+nD);

                    if isempty(obj.Weights(nD).scale)
                        if nWeight==1
                            scale{nD} = nan;
                        else
                            scale{nD} = 1:nWeight;
                        end
                        obj.Weights(nD).scale = scale{nD};
                    end

                    % hyperpameters for first dimension
                    switch prior{nD}
                        case 'L2'
                            obj.HP(nD) = HPstruct_L2(nD, HP{nD}, HPfit);
                            obj.Prior(nD).CovFun = @L2_covfun;  % L2-regularization (diagonal covariance prior)
                        case 'none'
                            obj.Prior(nD).CovFun = @infinite_cov; % L2-regularization (diagonal covariance prior)
                        case 'ARD'
                            if nD ==1, ch=""; else ch=nD; end
                            obj.HP(nD) = HPstruct_ard(nWeight, ch, HP{nD}, HPfit);
                            obj.Prior(nD).CovFun = @ard_covfun;
                    end
                    obj.Prior(nD).type = prior{nD};

                    obj.HP(nD).fit = HPfit;

                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, 1:nD, summing, prior, HP, scale, basis, single_tau, condthresh);
                    % !!! seems to overwrite all commands before, maybe we
                    % can just remove them?

                    obj.formula = label(nD);

                    %% categorical regressor
                case 'categorical'

                    if ~OneHotEncoding(nD)
                        warning('categorical variables should be coded by subindex');
                        OneHotEncoding(nD) = 1;
                    end

                    % build design matrix
                    this_scale = unique(X); % unique values
                    if isnumeric(this_scale)
                        this_scale(isnan(this_scale)) = []; % remove nan values
                    end
                    if ~isempty(obj.Weights(nD).scale)
                        if any(~ismember(obj.Weights(nD).scale,this_scale))
                            warning('exclude scale values not present in data');
                            obj.Weights(nD).scale(~ismember(obj.Weights(nD).scale,this_scale)) = [];
                        end
                    else
                        obj.Weights(nD).scale = this_scale'; % convert to row vector
                    end

                    % use one-hot encoding
                    obj.Data = one_hot_encoding(X,obj.Weights(nD).scale, OneHotEncoding(nD),nD);
                    nVal = size(obj.Weights(nD).scale,2); % number of values/levels
                    obj.Weights(nD).nWeight = nVal;

                    if strcmp(prior{nD}, 'L2')
                        % define prior covariance function (L2-regularization)
                        obj.Prior(nD).CovFun = @L2_covfun; % L2-regularization (diagonal covariance prior)

                        % define  hyperparameters
                        if ~iscell(HP)
                            HP = [cell(1,nD) {HP}];
                        end
                        if ~isempty(variance), HP{nD} = log(variance)/2; end
                        HP_L2 = HPwithdefault(HP{nD}, 0);
                    else % no prior (infinite covariance)
                        obj.Prior(nD).CovFun = @infinite_cov;
                    end
                    obj.Prior(nD).type = prior{nD};


                    obj.HP(nD) = HPstruct_L2(nD, HP_L2, HPfit);
                    obj.HP(nD).fit = HPfit;

                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, 1:nD-1, summing, prior, HP, scale, basis, single_tau, condthresh);

                    obj.formula = "cat("+label(nD)+ ")";

                    %% CONTINUOUS OR PERIODIC VARIABLE
                case {'continuous','periodic'}

                    if strcmpi(type, 'periodic')
                        X = mod(X,period);
                    end

                    % build design matrix/tensor
                    if OneHotEncoding(nD) % use one-hot-encoding
                        if strcmp(summing{nD},'joint')
                            % look for all unique combinations of rows
                            this_scale = unique(reshape(X,prod(siz(1:nD)), siz(nD+1)), 'rows')';
                        else
                            this_scale = unique(X)'; % use unique values as scale
                        end
                        this_scale(:,any(isnan(this_scale),1)) = []; % remove nan values
                        nVal = size(this_scale,2);
                        obj.Data = one_hot_encoding(X, this_scale, OneHotEncoding(nD),nD);
                    else % user-provided scale
                        this_scale = scale{nD};
                        nVal = size(X,nD+1);
                        assert(size(this_scale, 2)==nVal, 'the number of columns in scale does not match the number of columns in data array');
                        obj.Data = X;
                    end
                    obj.Weights(nD).nWeight = nVal;

                    % initial value of hyperparameters
                    if ~iscell(HP)
                        HP = [cell(1,nD-1) {HP}];
                    end
                    if ~isempty(tau), HP{nD}(1:length(tau)) = log(tau); end
                    if ~isempty(variance), HP{nD}(2) = log(variance)/2; end

                    % define continuous prior
                    obj = define_continuous_prior(obj,type, nD,this_scale, HP{nD}, basis{nD}, ...
                        binning,summing, period, single_tau, condthresh);
                    obj.HP(nD).fit = HPfit;

                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, 1:nD-1, summing, prior, HP, scale, basis, single_tau, condthresh);

                    obj.formula = "f(" + label(nD)+ ")";


                otherwise
                    error('unknown type:%s', type);
            end

            for d=1:nD
                obj.Weights(d).label = label(d);
            end
            obj.Weights(nD).plot = plot;

            % by default, first dimension (that does not split or separate)
            % is free, other have sum-one constraint
            if ~strcmpi(type, 'constant')
                FreeDim = find(~SplitOrSeparate,1);
                for d=1:nD
                    obj.Weights(d).constraint = "sum1"; % sum-one constraint
                end
                obj.Weights(FreeDim).constraint = "free";
                if strcmpi(type, 'categorical') % categorical variable: we fix the first weight to reference value
                    if nD==1
                        obj.Weights.constraint = "first0";
                    elseif nD~=FreeDim
                        obj.Weights.constraint = "first1";
                    end
                end
            end

            % whether components should be reordered according to variance, default: reorder if all components have same constraints
            if obj.rank>1
                obj.ordercomponent = true;
                for d=1:nD
                    % cc_rank1 = constraint(1,d);
                    %  if ~all( cellfun(@(cc) isequal(cc,cc_rank1), constraint(:,d)) )
                    if ~all(  constraint(:,d) == constraint(1,d))
                        obj.ordercomponent = false;
                    end
                end
                %  obj.ordercomponent = all(all(constraint==constraint(1,:)));
            end

            obj = obj.split_or_separate(summing);

            if ~isempty(constraint)
                assert(length(constraint)==nD, 'length of constraint C should match the number of dimensions in the regressor');
                for d=1:nD
                    obj.Weights(d).constraint = constraint(d);
                end
            end
        end

        %% OTHER METHODS %%%%

        %% SET RANK
        function obj = set.rank(obj, rank)
            if ~isnumeric(rank) || ~isscalar(rank) || rank<=0
                error('rank should be a positive integer');
            end
            obj.rank = rank;
        end


        %% GET NUMBER OF PARAMETERS
        function nWeight = get.nParameters(obj)
            if length(obj)>1
                error('this method is only defined for single object');
            end
            nWeight = [obj.Weights.nWeight]; % number of weights for each component
            %   nWeight = cellfun(@sum, {obj.Weights.nWeight}); % number of weights for each component
            nWeight = repelem(nWeight,obj.rank); % replicate by rank
        end

        %% GET NUMBER OF FREE PARAMETERS
        function nFreeWeights = get.nFreeParameters(obj)

            % number of total weights
            W = [obj.Weights];
            nWeight =[W.nWeight];
            %   nWeight = cellfun(@sum, {W.nWeight});

            % reduce free weights for constraints
            nConstraint = zeros(size(W));
            for c=1:numel(W)
                cc = constraint_structure(W(c));
                nConstraint(c) = cc.nConstraint;
            end
            nFreeWeights = nWeight - nConstraint;

            % when using basis, use number of basis functions instead
            withBasis = ~cellfun(@isempty, {W.basis});
            for c=find(withBasis)
                if W(c).basis(1).projected
                    nFreeWeights(c) = W(c).nWeight;
                else
                    nFreeWeights(c) = sum([W(c).basis.nWeight]);
                end
            end

            % number of free parameters per set weight
            nFreeWeights = repmat(nFreeWeights,obj.rank,1);
        end

        %% GET NUMBER OF FREE DIMENSIONS
        function nFreeDim = nFreeDimensions(obj)
            nFreeDim = zeros(size(obj));
            for i=1:numel(obj)
                constraint = constraint_type([obj(i).Weights]);
                isFixed = constraint~="fixed";
                nFreeDim(i) = max(sum(isFixed,2));
                % fd(i) = max(sum(constraint~='n',2));
            end
        end

        %% IS MULTIPLE RANK FREE DIMENSIONS
        function bool = isFreeMultipleRank(obj)
            bool = false(size(obj));
            for i=1:numel(obj)

                constraint = constraint_type([obj(i).Weights]);
                isFree = constraint=="free";
                bool(i) = obj(i).rank>1 && all(isFree,'all');
            end
        end

        %% TOTAL NUMBER OF PARAMETERS
        function np = get.nTotalParameters(obj)
            np = zeros(size(obj));
            for i=1:numel(obj)
                np(i) = sum([obj(i).Weights.nWeight])*obj(i).rank;
            end
        end

        %% GET LABELS OF WEIGHT SET
        function label = weight_labels(obj)
            % label = weight_labels(R) outputs cell array of all labels for
            % set of weights
            W = [obj.Weights];
            label = [W.label];
        end

        %% FIND SET OF WEIGHT WITH GIVEN LABEL
        function I = find_weights(obj, label)
            % I = find_weights(R, label) finds the set of weight with given
            % label in regressors R. I is a 2-by-nLabel matrix where the first
            % row signals the index of the regressor in the regressor array
            % and the second row signals the dimension for the corresponding label.
            %
            % I = find_weights(R, C) if C is a cell array of labels
            % provides a matrix of indices with two rows, one for the index
            % of the regressor and the second for the corresponding
            % dimension.
            % NaN indicates when no weight is found.
            %
            % find_weights(R) provides matrix for all weights

            nR = numel(obj); % number of regressor objects
            nD = [obj.nDim]; % dimensionality for each regressor object

            % vector of regressor inde for each set of weight
            Ridx = repelem(1:nR, nD);

            % build vector for dimensionality for each set of weight
            Dims = cell(1,nR);
            for i=1:numel(obj)
                Dims{i} = 1:nD(i);
            end
            Dims = [Dims{:}];
            RD = [Ridx;Dims]; % first row: regressor index; second row: dimension

            if nargin==1 % select all weights
                I = RD;
                return;
            end

            label = string(label);

            Wlbl = weight_labels(obj); % label for all set of weights

            % build index matrix corresponding to each label
            nL = length(label); % number of labels
            I = nan(2,nL);
            for i=1:nL
                j = find(label(i)== Wlbl);
                if ~isempty(j)
                    if length(j)>1
                        warning('more than one set of weights with label''%s''', label(i));
                    end
                    I(:,i) = RD(:,j(1));
                else
                    all_weight_labels = join(compose("""%s""\n", obj.weight_labels));
                    error('no set of weights with label ''%s''. Possible labels are:\n%s', label(i), all_weight_labels);
                end
            end
        end

        %% CONCATENATE ALL WEIGHTS
        function U = concatenate_weights(obj, dims, fild)
            % U = concatenate_weights(M) concatenate all weights posterior
            % mean in a single vector
            %
            % U = concatenate_weights(M, dims) to extract from a specific
            % regressor dimension for each regressor object (specified by
            % vector dims)
            % U = concatenate_weights(M,0) is the same as U =
            % concatenate_weights(obj)
            %
            % U = concatenate_weights(M,dims,metric) to extract a specific
            % metric related to the weights. Possible values: 'PosteriorMean'
            % [default], 'PosteriorStd', 'T','p', 'scale'

            if nargin<3
                fild = 'PosteriorMean';
            else
                assert(ismember(fild, {'PosteriorMean', 'PosteriorStd', 'T','p', 'scale','PriorMean'}),...
                    'incorrect metric');
            end
            if nargin==1 || isequal(dims,0) % all weights
                if strcmp(fild,'PriorMean')
                    U = [obj.Prior];
                else
                    U = [obj.Weights];
                end
                U = {U.(fild)};
                U = cellfun(@(x) x(:)', U,'unif',0);
                U = [U{:}];

            else
                assert(length(dims)==length(obj));

                % total number of regressors/weights
                nRegTot = 0;
                nW = zeros(1,length(obj));
                for m=1:length(obj)
                    nW(m)  = obj(m).Weights(dims(m)).nWeight;
                    nRegTot = nRegTot + nW(m);
                end

                %%select weights on specific dimensions
                U = zeros(1,nRegTot); % concatenante set of weights for this dimension for each component and each module

                ii = 0; % index for regressors in design matrix

                for m=1:length(obj)

                    d = dims(m); %which dimension we select
                    % project on all dimensions except the dimension to optimize

                    % for r=1:obj(m).rank
                    idx = ii + (1:nW(m)*obj(m).rank); % index of regressors in output vector

                    if strcmp(fild,'PriorMean')
                        thisU = obj(m).Prior(d).PriorMean;
                    else
                        thisU = obj(m).Weights(d).(fild);
                    end
                    U(idx) = thisU(:);

                    ii = idx(end); %
                    %  end
                end
            end
        end


        %% SET WEIGHTS AND HYPERPARAMETERS FROM ANOTHER MODEL / SET OF REGRESSORS
        function [obj,I] = set_weights_and_hyperparameters_from_model(obj, obj2, lbl, bool)
            % R = set_weights_and_hyperparameters_from_model(R, R2),
            % or R = set_weights_and_hyperparameters_from_model(R, M2)
            % sets the values of weights and hyperparameters in regressor R to corresponding values
            % in regressor R2 or GUM model M2 by matching labels for set of
            % weights
            %
            % obj = set_weights_and_hyperparameters_from_model(obj, obj2, lbl) set the values for
            % set(s) of weights with label lbl (lbl is a string or cell array
            % of string)

            if nargin<4
                bool = [true true];
            end

            if isa(obj2,'gum')
                obj2 = obj2.regressor;
            end

            if nargin<3 || isempty(lbl)
                % default: find all set of weights that are present on both
                % regressor objects
                Wlbl = weight_labels(obj);
                Wlbl2 = weight_labels(obj2);
                lbl = intersect(Wlbl, Wlbl2);
            else
                lbl = string(lbl);
            end

            I = nan(2,length(lbl));
            for i=1:length(lbl)
                % index for this label in both objects
                ind1 = obj.find_weights(lbl(i));
                ind2 = obj2.find_weights(lbl(i));

                assert(~any(isnan([ind1;ind2])), 'No set of weight label has the corresponding label in at least one of the regressor objects');

                if bool(1) % set weights
                    % corresponding set of weights
                    W = obj(ind1(1)).Weights(ind1(2));
                    W2 = obj2(ind2(1)).Weights(ind2(2));

                    assert(all(W.nWeight==W2.nWeight), 'the number of weights do not match for at least one set of weights with same label');

                    % assign value
                    obj(ind1(1)).Weights(ind1(2)).PosteriorMean = W2.PosteriorMean;
                end

                if bool(2) % set hyperparameters
                    % corresponding set of HPs
                    hp = obj(ind1(1)).HP(ind1(2));
                    hp2 = obj2(ind2(1)).HP(ind2(2));

                    assert(length(hp.HP)==length(hp2.HP), 'the number of hyperparameters do not match for at least one set of weights with same label');

                    % assign value
                    obj(ind1(1)).HP(ind1(2)).HP = hp2.HP;
                end
            end
        end

        %% SET HYPERPARAMETERS FROM ANOTHER MODEL / SET OF REGRESSORS
        function [obj,I] = set_hyperparameters_from_model(obj, obj2, lbl)
            % R = set_hyperparameters_from_model(R, R2) or R = set_hyperparameters_from_model(R, M2)
            % sets the values of hyperparameters in regressor R to corresponding values
            % in regressor R2 or GUM model M2 by matching labels for set of
            % weights
            %
            % obj = set_hyperparameters_from_model(obj, obj2, lbl) set the values for
            % set(s) of weights with label lbl (lbl is a string or cell array
            % of string)

            if nargin<3
                lbl = [];
            end

            bool = [false true]; % copy just hyperparameters, not weights
            [obj,I] = set_weights_and_hyperparameters_from_model(obj, obj2, lbl, bool);
        end

        %% SET WEIGHTS FROM ANOTHER MODEL / SET OF REGRESSORS
        function [obj,I] = set_weights_from_model(obj, obj2, lbl)
            % R = set_weights_from_model(R, R2) or R = set_weights_from_model(R, M2)
            % sets the values of weights in regressor R to corresponding values
            % in regressor R2 or GUM model M2 by matching labels for set of
            % weights
            %
            % obj = set_weights_from_model(obj, obj2, lbl) set the values for
            % set(s) of weights with label lbl (lbl is a string or cell array
            % of string)

            if nargin<3
                lbl = [];
            end

            bool = [true false]; % copy just weights, not hyperparameters
            [obj,I] = set_weights_and_hyperparameters_from_model(obj, obj2, lbl, bool);
        end

        %% SET WEIGHTS FOR SET OF MODULES
        function obj = set_weights(obj,U, dims, fild)
            % M = set_weights(M,M2) set values of regressor M provided by
            %  regressor M2
            %  M = set_weights(M,U) set values of regressor M provided by
            %  vector U
            % M = set_weights(M,U, dims) to apply only to specific weights
            %
            % M = set_weights(M,U, dims, field) or M = set_weights(M,U, [],
            % field) to apply to specific field: 'PosteriorMean' (default=,
            % 'PosteriorStd', 'U_allstarting','U_CV', 'U_bootstrap','ci_boostrap'

            if isa(U, 'regressor')
                % if regressor object, extract vector of weights
                % !! check for size of M, creating method 'same_size'
                if nargin>2
                    error('not coded yet');
                end
                U = concatenate_weights(U);

            end
            if isrow(U)
                U = U';
            end

            if nargin<4
                fild = 'PosteriorMean';
            end

            ii = 0;
            for m=1:length(obj)

                % if dimension is provided, apply only to
                % specific dim
                if nargin>=3 && ~isempty(dims)
                    dim_list = dims(m);
                else % otherwise to all dimensions
                    dim_list = 1:obj(m).nDim;
                end
                for d = dim_list
                    nW = obj(m).Weights(d).nWeight; % number of regressors
                    this_constraint = constraint_type(obj(m).Weights(d));

                    switch fild
                        case 'PosteriorMean'
                            for r=1:obj(m).rank % assign new set of weight to each component
                                if this_constraint~="fixed" % unless fixed weights

                                    idx = ii + (nW*(r-1)+1 : nW*r); % index of regressors for this component
                                    obj(m).Weights(d).PosteriorMean(r,:) = U(idx); % weight for corresponding adim and component
                                    if this_constraint=="first1" % enforce this
                                        obj(m).Weights(d).PosteriorMean(r,1)=1;
                                        %  warning('check if I should change idx to idx+1');
                                    elseif this_constraint=="first0" % enforce this
                                        obj(m).Weights(d).PosteriorMean(r,1)=0;
                                    end
                                end
                            end
                        case 'PosteriorStd'
                            for r=1:obj(m).rank % assign new set of weight to each component
                                %if this_constraint~="fixed" % unless fixed weights

                                idx = ii + (nW*(r-1)+1 : nW*r); % index of regressors for this component
                                obj(m).Weights(d).PosteriorStd(r,:) = U(idx); % weight for corresponding adim and component
                                if any(this_constraint==["first0","first1"]) % enforce this
                                    obj(m).Weights(d).PosteriorStd(r,1)=0;
                                end
                                %end
                            end
                        case {'U_allstarting','U_CV', 'U_bootstrap','ci_boostrap'}
                            idx = ii+(1:nW*obj(m).rank);
                            this_U = reshape(U(idx,:), [nW, obj(m).rank, size(U,2)]); % weights x rank x initial point
                            this_U = permute(this_U,[3 1 2]); % initial point x weights x rank
                            obj(m).Weights(d).(fild) = this_U;
                        otherwise
                            error('incorrect field');
                    end

                    ii = ii + obj(m).rank*nW;
                end
            end
            assert(ii==size(U,1), 'did not arrive to the end of the vector');
        end

        %% SET FREE WEIGHTS (or anything of same size - used only for gradient i think)
        function FW = set_free_weights(obj,U, FW, dims)
            % FW = set_free_weights(M,U, FW, dims)
            % used to compute gradient

            if isempty(FW)
                FW = cell(1, length(obj));
                for m=1:length(obj)
                    FW{m} = cell(obj(m).rank, obj(m).nDim);
                end
            end
            ii = 0;
            for m=1:length(obj)

                % if dimension is provided, apply only to
                % specific dim
                if nargin>=3 && ~isempty(dims)
                    dim_list = dims(m);
                else % otherwise to all dimensions
                    dim_list = 1:obj(m).nDim;
                end
                nR = obj(m).nFreeParameters;
                for d = dim_list

                    constraint = constraint_type(obj(m).Weights(d));
                    for r=1:obj(m).rank % assign new set of weight to each component
                        if constraint(r)~="fixed" % unless fixed weights
                            idx = ii + (1:nR(r,d)); % index of regressors for this component
                            FW{m}{r,d} = U(idx); % weight for corresponding adim and component
                        end
                        ii = ii + nR(r,d);

                    end

                end
            end
        end

        %% FREEZE WEIGHTS
        function obj = freeze_weights(obj, varargin)
            % R = R.freeze_weights;
            % "freezes" weights of model, i.e. change them to "fixed"
            %
            % R = R.freeze_weights(regressor_label);
            % freezes weights for regressor with corresponding label

            % if needed, initialize weights to default values
            obj = obj.initialize_weights('normal');

            % regressor index and dimensionality of all sets of weights
            idx = find_weights(obj,varargin{:});


            for j=1:size(idx,2)
                i = idx(1,j);
                d = idx(2,j);
                obj(i).Weights(d).constraint = "fixed";
                obj(i).Weights(d).PosteriorStd = nan(size(obj(i).Weights(d).PosteriorMean));
                obj(i).Weights(d).basis = []; % remove basis functions
                obj(i).Weights(d).nFreeWeight = 0;

                obj(i).Prior(d) = empty_prior_structure(1); % remove prior
                obj(i).HP(d) =  HPstruct(); % empty HP structure
                %  obj(i).HP(d).fit(:) = 0;
            end
        end

        %% ORTHOGONALIZE WEIGHTS (FOR PWA)
        function obj = orthogonalize_weights(obj, d)
            if nargin<1
                %  constraint = [obj.Weights.constraint];
                constraint = constraint_type([obj.Weights]);

                FreeDims = find(all(constraint=="free",1));
                [~,i_d] = min(obj.nWeight(FreeDims));
                d = FreeDims(i_d);
            end

            other_dims = setdiff(1:obj.nDim,d);

            UU = obj.Weights.PosteriorMean;
            normU = zeros(obj.rank,1);
            normU(1) = sum(UU(1,:).^2);
            for r=2:obj.rank
                prj = UU(1:r-1,:)*UU(r,:)' ./normU(1:r-1); % project onto previous vectors
                this_UU = UU(r,:) - prj'*UU(1:r-1,:); % orthogonal vector
                obj.U{r,d} = this_UU;
                UU(r,:) = this_UU;
                normU(r) = sum(this_UU.^2); % norm2 of new vector

                error('change other weights!');
                for dd = other_dims
                    obj.Weights(dd).PosteriorMean(r,:) = xxx;
                end
            end

        end

        %% ASSIGN HYPERPARAMETER VALUES TO REGRESSORS
        function obj = set_hyperparameters(obj, HP, HPidx)
            % R = set_hyperparameters(R, HP) sets hyperparameters in regressor R to
            % values in vector HP
            %
            % R = set_hyperparameters(R, HP, idx) provides cell array of
            % hyperparameter positions
            if nargin<3
                [HPidx, nHP] = get_hyperparameters_indices(obj);
                assert(length(HP)==sum([nHP{:}]),'incorrect length for HP');
            end

            for m=1:length(obj) % for each regressor

                for d=1:obj(m).nDim % for each dimension
                    for r=1:size(obj(m).HP,1) %M(m).rank
                        cc = HPidx{m}{r,d};  %index for corresponding module

                        % hyperparameter values for this component
                        HP_fittable = logical(obj(m).HP(r,d).fit);
                        obj(m).HP(r,d).HP(HP_fittable) = HP(cc); % fittable values
                    end
                end

            end
        end

        %% GET INDICES FOR REGRESSOR HYPERPARAMETERS
        function [HPidx, nHP] = get_hyperparameters_indices(obj)
            % [HPidx, nHP] = get_hyperparameters_indices(obj) get indices for index
            % of fittable hyperparameters in regressor set.
            nM = length(obj);

            nHP  = cell(1,nM); % number of hyperparameters in each module
            HPidx = cell(1,nM);     % index of hyperparameters for each component
            cnt = 0; % counter
            for m=1:nM
                this_HP = obj(m).HP;
                HPidx{m} = cell(size(this_HP,1),obj(m).nDim);

                %retrieve number of fittable hyperparameters for each function
                this_HPfit = reshape({this_HP.fit}, size(this_HP));
                nHP{m} = cellfun(@sum, this_HPfit);

                % retrieve indices of hyperparameters for this component
                for d=1:obj(m).nDim
                    for r=1:size(nHP{m},1)
                        HPidx{m}{r,d} = cnt + (1:nHP{m}(r,d));
                        cnt = cnt + nHP{m}(r,d); % update counter
                    end
                end
            end
        end

        %% SET RANK
        function obj = set_rank(obj, rank, bool)
            % R = R.set_rank(rank);
            % R = R.set_rank(rank, independent_hps) where independent_hps is a boolean set to true
            % if rank have independent hyperparameters (e.g. for Automatic
            % Relevance Determinatino)

            obj.rank = rank;

            if nargin && any(bool) % use different set of hyperparameters
                obj.Prior = repmat(obj.Prior, rank,1);
                if isscalar(bool)
                    bool = repmat(bool, 1, obj.nDim);
                end
                for d=find(bool) %for all specified dimensions
                    obj.HP(:,d) = repmat(obj.HP(1,d), rank,1);
                    obj.Prior(:,d) = repmat(obj.Prior(1,d), rank, 1);
                end
            end
        end

        %% REMOVE CONSTRAINED PART FROM PREDICTOR
        function  [rho, Uconst] = remove_constrained_from_predictor(obj, dims, rho, Phi, UU)
            %  [rho, Uconst] = remove_constrained_from_predictor(obj, dims, rho, Phi, UU)
            % used in IRLS algorithm

            nM = length(obj); % number of regressor bojects
            W = empty_weight_structure(nM); % concatenate specified weights structure

            %nR = zeros(1,nM);
            for m=1:nM % for each module
                W(m) = obj(m).Weights(dims(m)); % select weight structures for specified dimensions
            end
            nW = [W.nWeight]; % corresponding number of weights
            % nW = cellfun(@sum, {W.nWeight}); % corresponding number of weights

            nRegTotal = sum(nW); % total number of regressors

            %% first process fixed set of weights
            ii=0;
            %recompute_rho = false;
            FixedWeight = false(1,nRegTotal);
            for m=1:nM
                % constraint = constraint_type(W(m));
                %  if any(constraint == "fixed") % if weight is fixed, remove entire projection
                %      for r = find(constraint == "fixed") % for each set fo fixed weight
                for r=1:obj(m).rank
                    idx  = ii + (nW(m)*(r-1)+1:nW(m)*r); % index of regressors
                    FixedWeight(idx) = constrained_weight(W(m)); % fixed weights (either whole set or just subset of weights)
                end
                %          FixedWeight(idx) = true;
                %          recompute_rho = true;

                %      end
                %   end
                ii = ii + obj(m).rank*nW(m);
            end

            if any(FixedWeight) % compute the projection without fixed weights (if any)
                rho = rho - Phi(:,FixedWeight)*UU(FixedWeight)';
            end


            %% then weights with linear constraint
            Uc = cell(1,nM);

            ii=0;
            for m=1:nM % for each module
                U_const = zeros(1,obj(m).rank); % constant weight for each component

                for r=1:obj(m).rank

                    idx = ii + (1:nW(m)); % index of regressors

                    % removing portion of rho due to projections of weights perpendicular to constraints
                    ct = constraint_structure(W(m));
                    nMultiConstraint = ct.nConstraint - sum(constrained_weight(W(m))); % constraints that apply on sets of weights (not individual ones, which have been dealt before hand)
                    if nMultiConstraint>0
                        if ct.nConstraint==1 && all(ct.V==ct.V(1)) % constraint applies equally to all weights (e.g. "sum1", "mean1")
                            if ct.u~=0
                                U_const(r) = ct.u / sum(ct.V); %

                                % removing portion of rho due to projections
                                rho = rho - U_const(r)*sum(Phi(:,idx),2);
                            end
                        else
                            error('not coded yet');
                        end

                    end
                    ii =idx(end);

                end
                Uc{m} = U_const;
            end
            Uc2 = [Uc{:}]; % concatenate over modules

            % fixed part of the weights
            Uconst = repelem(Uc2,repelem(nW',[obj.rank]))';

            % we treated fixed weights apart, let's add them here
            Uconst(FixedWeight) = Uconst(FixedWeight) + UU(FixedWeight)';
        end

        %% DEFINE PRIOR FOR ARRAY DIMENSIONS
        function  obj = define_priors_across_dims(obj, dims, summing, prior, HP, scale, basis, single_tau, condthresh)
            % make sure these options are in cell format and have minimum
            % size

            if isempty(dims)
                return;
            end

            summing = tocell(summing, max(dims));
            HP = tocell(HP, max(dims));
            prior = tocell(prior, max(dims));
            single_tau(end+1:max(dims)) = true;

            for d= dims % for all specified dimensions
                if isempty(summing{d}) || strcmp(summing{d},'weighted') % by default: linear set of weights for each dimension
                    summing{d} = 'linear';
                end
                obj.Weights(d).type = summing{d};
                if ~isempty(scale{d})
                    obj.Weights(d).scale = scale{d};
                    obj.Weights(d).nWeight = size(scale{d},2);
                else
                    obj.Weights(d).scale = 1:obj.Weights(d).nWeight;
                end

                % what type of prior are we using for this dimension
                switch summing{d}
                    case {'weighted','linear'}

                        % hyperpameters for first dimension
                        if isempty(prior{d})
                            prior{d} = 'L2';
                        end
                        switch prior{d}
                            case 'L2'
                                obj.HP(d) = HPstruct_L2(d);
                                obj.HP(d).HP = HPwithdefault(HP{d}, 0); % log-variance hyperparameter
                                obj.Prior(d).CovFun = @L2_covfun; % L2-regularization (diagonal covariance prior)
                            case 'none'
                                obj.Prior(d).CovFun = @infinite_cov;
                            case 'ARD'
                                obj.HP(d) = HPstruct_ard(obj.Weights(d).nWeight, d, HP{d});
                                obj.Prior(d).CovFun = @ard_covfun;
                        end

                        obj.Prior(d).type = prior{d};

                    case 'equal'
                        obj.Weights(d).PosteriorMean = ones(1,obj.Weights(d).nWeight);
                        obj.HP(d) = HPstruct;
                        obj.Prior(d).type = 'none';

                    case {'continuous','periodic'}
                        if isempty(obj.Weights(d).scale)
                            scl = 1:obj.Weights(d).nWeight;
                        else
                            scl = obj.Weights(d).scale;
                        end

                        % define continuous prior
                        obj = define_continuous_prior(obj,summing{d}, d,scl,HP{d}, basis{d}, [],[],2*pi, single_tau(d), condthresh);
                end
            end
        end

        %% DEFINE PRIOR OVER CONTINUOUS FUNCTION
        function obj = define_continuous_prior(obj, type, d, scale, HP, basis, binning, summing, period, single_tau, condthresh)

            if nargin<4
                HP = [];
            end

            obj.HP(d) = HPstruct; % initialize HP structure
            obj.Weights(d).scale = scale; % values
            obj.Weights(d).type = type; % 'continuous';

            if nargin<6 || strcmp(basis,'auto')
                if size(scale,2)>100 && ~any(strcmp(summing{d},{'split','separate'})) && size(scale,1)==1
                    % Fourier basis is the default if more than 100 levels
                    basis = 'fourier';
                else % otherwise no basis functions
                    basis = [];
                end
            elseif strcmp(basis,'none')
                basis = [];
            end
            obj.Weights(d).basis = basis;

            % default basis structure
            B = struct('nWeight',nan,'fun',[], 'fixed',true, 'params',[]);

            %% define prior covariance function (and basis function if required)
            if isempty(basis)

                % prior covariance function
                if strcmpi(type,'periodic')
                    obj.Prior(d).CovFun = @(s,x) cov_periodic(s, x, period); % session covariance prior
                    obj.Prior(d).type = 'periodic';
                else
                    obj.Prior(d).CovFun = @covSquaredExp; % session covariance prior
                    obj.Prior(d).type = 'SquaredExponential';
                end
                obj.Weights(d).nFreeWeight = obj.Weights(d).nWeight;

            elseif length(basis)>=7 && strcmpi(basis(1:7), 'fourier')

                if length(basis)>7 % 'fourierN'
                    nBasisFun = str2double(basis(8:end));
                    assert(~isnan(nBasisFun), 'incorrect basis label');
                    condthresh = nan;
                else % 'fourier': number of basis functions is calculated automatically
                    nBasisFun = nan;
                end

                if any(strcmp(summing,{'split','separate'}))
                    error('spectral trick not coded for sum=''%s''', summing);
                end
                %if strcmpi(type,'periodic')
                %    tau = period/4; % initial time scale: period/4
                %else
                %    tau = scale(end)-scale(1); % initial time scale: span of values
                %end

                obj.Prior(d).CovFun = @covSquaredExp_Fourier;
                obj.Weights(d).nFreeWeight = nBasisFun;

                B.fun = @fourier_basis;
                B.fixed = true; % although the actual number of regressors do change
                padding = ~strcmpi(type, 'periodic'); % virtual zero-padding if non-periodic transform

                B.params = struct('nFreq',nBasisFun,'condthresh',condthresh, 'padding', padding);
                if strcmpi(type,'periodic')
                    % !! for periodic shouldn't use SquaredExponential
                    % prior
                    B.params.Tcirc = period;
                end
                obj.Prior(d).type = 'SquaredExponential';


            elseif length(basis)>4 && strcmp(basis(1:4),'poly')

                %% POLYNOMIAL BASIS FUNCTIONS
                nBasisFun = str2double(basis(5:end));

                obj.Prior(d).CovFun = @L2basis_covfun;
                obj.Prior(d).type = 'L2_polynomial';

                B.nWeight = nBasisFun;
                B.fun = @basis_poly;
                B.fixed = true;
                B.params = struct('order',nBasisFun);

                obj.Weights(d).nFreeWeight = nBasisFun;

            elseif length(basis)>=3 && strcmp(basis(1:3),'exp')

                %% EXPONENTIAL BASIS FUNCTIONS
                if length(basis)==3
                    nBasisFun =1;
                else
                    nBasisFun = str2double(basis(4:end));
                end

                obj.Prior(d).CovFun = @L2basis_covfun;
                obj.Prior(d).type = 'L2_exponential';

                B.nWeight = nBasisFun;
                B.fun = @basis_exp;
                B.fixed = false;
                B.params = struct('order',nBasisFun);

                obj.Weights(d).nFreeWeight = nBasisFun;

            elseif length(basis)>=12 && strcmp(basis(1:12),'raisedcosine')
                %% RAISED COSINE BASIS FUNCTIONS (Pillow et al., Nature 2008)
                if length(basis)==12
                    nBasisFun =1;
                else
                    nBasisFun = str2double(basis(13:end));
                end

                obj.Prior(d).CovFun = @L2basis_covfun;
                obj.Prior(d).type = 'L2_raisedcosine';

                B.nWeight = nBasisFun;
                B.fun = @basis_raisedcos;
                B.fixed = false;
                B.params = struct('nFunctions',nBasisFun);

                obj.Weights(d).nFreeWeight = nBasisFun;


            elseif length(basis)>=5 && strcmp(basis(1:5),'gamma')

                %% GAMMA FUNCTIONS (i.e. http://dx.doi.org/10.17605/OSF.IO/4FQJ9)
                if length(basis)==5
                    nBasisFun =1;
                else
                    nBasisFun = str2double(basis(6:end));
                end

                obj.Prior(d).CovFun = @L2basis_covfun;
                obj.Prior(d).type = 'L2_gamma';

                B.nWeight = nBasisFun;
                B.fun = @basis_gamma;
                B.fixed = false;
                B.params = struct('nFunctions',nBasisFun);

                obj.Weights(d).nFreeWeight = nBasisFun;
            else
                error('incorrect basis type: %s', basis);

            end
            if ~isempty(basis)
                B.projected = false;
                obj.Weights(d).basis = B;
            end

            % remove constraints from number of free weights
            if isempty(obj.Weights(d).constraint)
                obj.Weights(d).constraint = "free";
            end
            cc = constraint_structure(obj.Weights(d));
            obj.Weights(d).nFreeWeight = obj.Weights(d).nFreeWeight - cc.nConstraint;

            %% define hyperparameters
            HH = obj.HP(d);
            switch obj.Prior(d).type
                case 'SquaredExponential'
                    HH = HPstruct_SquaredExp(HH, scale, HP, type, period, single_tau, basis, binning);
                case 'L2_polynomial'
                    HH = HPstruct_L2(nBasisFun+1, HP);
                case 'L2_exponential'
                    HH = HPstruct_L2exp(HH, scale, HP, nBasisFun);
                case 'L2_raisedcosine'
                    HH = HPstruct_L2raisedcos(HH, scale,nBasisFun);
                case 'L2_gamma'
                    HH = HPstruct_L2gamma(HH, scale,nBasisFun);
            end

            obj.HP(d) =  HH ;
        end


        %% PROJECTION MATRIX from free set of parameters to complete set of
        % parameters
        function PP = ProjectionMatrix(obj)

            D = obj.nDim;
            rr = obj.rank;
            PP = cell(rr,D);
            for d=1:D % for each dimension
                for r=1:rr % for each component
                    this_constraint = constraint_structure(obj.Weights(d));
                    PP{r,d} = compute_orthonormal_basis_project(this_constraint.V);
                end
            end

        end

        %% COMPUTE DESIGN MATRIX
        function [Phi,nWeight, dims] = design_matrix(obj,subset, dims, init_weight)
            % Phi = design_matrix(R);
            % computes the design matrix associated with regressors R
            %
            % Phi = design_matrix(R, subset);
            % uses a subset of observations
            %
            % Phi = design_matrix(R, subset, dims);
            % defines which dimension to project onto for each regressor
            % (for multidimensional regressors). dims must be a vector or cell array of
            % the same size as R. Default value is 1. Cell array allows to
            % project over multiple dimensions (use 0 to project over all
            % dimensions).
            %
            % Phi = design_matrix(obj,subset, dims, init_weight) sets
            % whether weights should be initialized (default:true).
            %
            % [Phi,nReg] = design_matrix(...) provides the number of
            % columns corresponding to each regressor

            nM = length(obj); % number of modules

            if nargin>1 && ~isempty(subset) % only plot for subset of trials
                obj = extract_observations(obj,subset);
            end

            if nargin<3 || isempty(dims) % by default, project on dimension 1
                dims = ones(1,nM);
                dims = num2cell(dims);
            elseif isequal(dims,0)
                % project over all dimensions
                dims = cell(1,nM);
                for m=1:nM
                    dims{m} = 1:obj(m).nDim;
                end

            elseif ~iscell(dims)
                dims = num2cell(dims);

            end

            if nargin<4 % by default, initialize weights
                init_weight = true;
            end

            nF = sum(cellfun(@length, dims));

            %  computes the number of columns corresponding to each
            %  regressor
            nWeight = zeros(1,nF);
            ii = 1;
            for m=1:nM
                for d=dims{m}
                    nWeight(ii) = obj(m).rank * obj(m).Weights(d).nWeight; % add number of regressors for this module
                    ii = ii+1;
                end
            end

            %% initialize weights
            if init_weight
                % initialize weights to default value
                obj = obj.initialize_weights();
            end

            ii = 0; % index for regressors in design matrix

            % use sparse coding if any data array is sparse
            SpCode = any(cellfun(@issparse, {obj.Data}));
            if SpCode
                Phi = sparse(obj(1).nObs,sum(nWeight));
            else
                Phi = zeros(obj(1).nObs,sum(nWeight));
            end
            for m=1:nM % for each module

                % project on all dimensions except the dimension to optimize
                for d=dims{m}
                    for r=1:obj(m).rank
                        idx = ii + (1:obj(m).Weights(d).nWeight); % index of regressors
                        Phi(:,idx) = ProjectDimension(obj(m),r,d); % tensor product, and squeeze into observation x covariate matrix
                        ii = idx(end); %
                    end
                end
            end


        end


        %% CHECK PRIOR COVARIANCE and provide default covariance if needed
        function obj = check_prior_covariance(obj)
            % if more than one object, do it iteratively
            if length(obj)>1
                for i=1:numel(obj)
                    obj(i) = check_prior_covariance(obj(i));
                end
                return;
            end

            nD = obj.nDim; % dimension in this module
            rk = obj.rank;

            for r=1:rk % for each rank
                for d=1:nD % for each dimension
                    W = obj.Weights(d);
                    nW = W.nWeight;

                    % if more than one rank and only defined for first, use
                    % same as first
                    if r>1 && ((size(obj.Prior,1)<r) || isempty(obj.Prior(r,d).PriorCovariance))
                        obj.Prior(r,d).PriorCovariance = obj.Prior(1,d).PriorCovariance;
                    end

                    if isempty(obj.Prior(r,d).PriorCovariance) % by default:
                        ct = constraint_type(W);
                        if d<=nD && ismember(ct,["first0","first1"]) % no covariance for first weight (set to zero/one), unit for the others
                            obj.Prior(r,d).PriorCovariance = diag([0 ones(1,nW-1)]);
                        elseif ct == "fixed"
                            obj.Prior(r,d).PriorCovariance = zeros(nW);
                        else % otherwise diagonal unit covariance
                            obj.Prior(r,d).PriorCovariance = speye(nW);
                        end
                    end
                    if ~isequal(size(obj.Prior(r,d).PriorCovariance),[nW nW])
                        error('Prior covariance for dimension %d should be a square matrix of size %d', d, nW);
                    end
                end
            end
        end


        %% PROJECT TO SPACE OF BASIS FUNCTIONS
        function obj = project_to_basis(obj)
            % R = R.project_to_basis;
            % projects data onto basis functions.

            for m=1:length(obj) % for each module

                for d=1:obj(m).nDim % each dimension
                    W = obj(m).Weights(d);
                    B = W.basis;

                    if ~isempty(B) && any(~[B.projected]) % if this regressor uses a set of non-projected basis functions

                        % hyperparameter values for this component
                        this_HP = obj(m).HP(d); %fixed values
                        this_scale = W.scale;
                        nW = W.nWeight;

                        % compute projection matrix and levels in projected space
                        [B,new_scale] = compute_basis_functions(B, this_scale, this_HP);

                        Bmat = B(1).B;
                        assert( size(Bmat,2)==nW, 'Incorrect number of columns (%d) for projection on basis in component ''%s'' (expected %d)',...
                            size(Bmat,2), W.label, nW);
                        new_nWeight = size(Bmat,1);

                        % store values for full space in B
                        B(1).scale = W.scale;
                        B(1).nWeight = W.nWeight;
                        W.nWeight = new_nWeight;
                        W.scale = new_scale;

                        %% if weights are constrained to have mean or average one, change projection matrix so that sum of weights constrained to 1
                        cc = constraint_structure(W);
                        if cc(1).nConstraint>0
                            for r=2:obj(m).rank
                                if ~isequal(cc(1), cc(r))
                                    error('Basis functions cannot use different constraints for different orders');
                                end
                            end

                            if cc(1).nConstraint==1 && all(cc(1).U == cc(1).U(1)) % mean1, sum1, nullmean

                                B2 = sum(Bmat,2); % ( 1*(B*U) = const -> (1*B)*U = const

                                if cc(1).u~=0
                                    B2 = B2 / cc(1).u;
                                    cc(1).u = 1;
                                end

                                invB = diag(1./B2);
                                Bmat = invB * Bmat; % change projection matrix so that constraint is sum of weight equal to one

                                cc(1).V = ones(new_nWeight,1);

                                if cc(1).u~=0
                                    cc(1).u = 1;
                                end
                            else
                                %V*(B*U) = u -> (V*B)*U = u
                                cc(1).V = cc(1).V * Bmat;
                            end
                            B(1).B = Bmat;

                            % !! I'm commenting because I don't
                            % understand what it's supposed to be
                            % doing... go back to it later on
                            %                                 for r2=1:obj(m).rank
                            %                                     if cc(r)~='b'
                            %                                         obj(m).Weights(d).constraint(r2) = 's';
                            %                                     end
                            %                                     obj(m).Prior(r2,d).CovFun = @(sc,hp) covfun_transfo(sc,hp,  diag(B.B) , obj(m).Prior(r2,d).CovFun);
                            %                                 end
                            %break; % do not do it for other orders
                            %end

                            B(1).constraint = W.constraint;
                            obj(m).Weights(d).constraint = cc(1);
                        end


                        %% if initial value of weights are provided, compute in new basis
                        U = W.PosteriorMean;
                        if ~isempty(U)
                            B(1).PosteriorMean = U;
                            W.PosteriorMean = zeros(obj(m).rank,size(Bmat,1));
                            for r=1:obj(m).rank
                                % use pseudo-inverse to project back
                                W.PosteriorMean(r,:) = pinv(Bmat')* U';
                            end
                        end
                        W.PosteriorStd = [];

                        % store copy of original data
                        if isempty(obj(m).DataOriginal)
                            obj(m).DataOriginal = obj(m).Data;
                        end

                        % apply change of basis to data (tensor product)
                        Bcell = cell(1,obj(m).nDim+1);
                        Bcell{d+1} = Bmat;
                        obj(m).Data = tensorprod(obj(m).Data, Bcell);

                        [B.projected] = deal(true); % set projected boolean to true
                        W.basis = B;

                        obj(m).Weights(d) = W;
                    end
                end
            end
        end

        %% PROJECT FROM SPACE OF BASIS FUNCTIONS BACK TO ORIGINAL SPACE
        function obj = project_from_basis(obj)
            % R = project_from_basis(R) projects weights in R back to original space.

            for m=1:length(obj) % for each module

                for d=1:obj(m).nDim
                    B = obj(m).Weights(d).basis;
                    if ~isempty(B) && any([B.projected])

                        tmp_nWeight = B(1).nWeight;
                        tmp_scale = B(1).scale;
                        B(1).nWeight = obj(m).Weights(d).nWeight;
                        obj(m).Weights(d).nWeight = tmp_nWeight;
                        obj(m).Weights(d).scale = tmp_scale;

                        W = obj(m).Weights(d); % weight structure

                        % save weights (mean, covariance, T-stat, p-value) in basis space in basis structure
                        B(1).PosteriorMean =  W.PosteriorMean;
                        B(1).PosteriorStd =  W.PosteriorStd;
                        B(1).PosteriorCov = W.PosteriorCov;
                        B(1).T = W.T;
                        B(1).p = W.p;

                        % compute posterior back in original domain
                        W.PosteriorMean = W.PosteriorMean * B(1).B;

                        % compute posterior covariance, standard error
                        % of weights in original domain
                        rk = obj(m).rank;
                        if ~isempty(B(1).PosteriorCov)
                            W.PosteriorCov = zeros(W.nWeight, W.nWeight,rk);
                            W.PosteriorStd = zeros(rk, W.nWeight);
                            for r=1:rk
                                PCov  = B(1).B' * B(1).PosteriorCov(:,:,r) * B(1).B;
                                W.PosteriorCov(:,:,r) = PCov;
                                W.PosteriorStd(r,:) = sqrt(diag(PCov))'; % standard deviation of posterior covariance in original domain
                            end

                            % compute T-statistic and p-value
                            W.T = W.PosteriorMean ./ W.PosteriorStd; % wald T value
                            W.p = 2*normcdf(-abs(W.T)); % two-tailed T-test w.r.t 0
                        else
                            W.PosteriorCov = [];
                            W.PosteriorStd = [];
                            W.T = [];
                            W.p = [];
                        end

                        % replace basis structure
                        [B.projected] = deal(false);
                        if isfield(B, 'constraint')
                            W.constraint = B(1).constraint;
                        end
                        W.basis = B;

                        obj(m).Weights(d) = W;

                        % recover original data
                        obj(m).Data = obj(m).DataOriginal;
                    end
                end
            end
        end


        %% PLOT BASIS FUNCTIONS
        function h = plot_basis_functions(obj, dims)
            % plot_basis_functions(R) plots basis functions in regressor
            %
            %plot_basis_functions(R, dims) to specify which regressor to
            %select
            %
            %h = plot_basis_functions(...) provides graphical handles

            if nargin<2
                % if not provided compute for all regressors with basis
                % functions
                dims = cell(1,numel(obj));
                for m=1:numel(obj)
                    % check set of weights with basis functions
                    Bcell = {obj(m).Weights.basis};
                    dims{m} = find(~cellfun(@isempty, Bcell));
                end
            end

            nSub = sum(cellfun(@length, dims)); % total number of plots
            iSub = 1; % subplot counter
            h = cell(1,nSub);

            for m=1:numel(obj)
                for d=dims{m}
                    W = obj(m).Weights(d);
                    HPs = obj(m).HP(d);
                    [B,scale_basis] = compute_basis_functions(W.basis, W.scale, HPs);
                    % [Bmat,scale_basis] = compute_basis_functions(obj(m),d);

                    scale_basis(2:end,:) = []; % if concatenated regressors                  
                    if isnumeric(scale_basis)
                                                is_real_basis = ~isnan(scale_basis);
                        scale_basis = num2strcell(scale_basis);
                    else
                        is_real_basis = true(1,length(scale_basis));
                    end
                     Bmat = B(1).B(is_real_basis,:);


                    subplot2(nSub,iSub);
                    h{iSub} =  plot(obj(m).Weights(d).scale(1,:), Bmat);
                    
                    box off;
                    legend(scale_basis(1,is_real_basis));
                    axis tight;
                    title(obj(m).Weights(d).label);
                    iSub = iSub+1;
                end

            end
        end

        %% COMPUTE PRIOR COVARIANCE (and gradient)
        function [obj,GHP] = compute_prior_covariance(obj, recompute)
            % R = compute_prior_covariance(R);
            % computes prior covariance for weights
            %
            % [R,GHP] = compute_prior_covariance(R);
            % provides gradient over hyperparameters
            %
            % compute_prior_covariance(R, bool)
            % to specify if covariance should be computed again if
            % the matrix is already provided (default: true)

            if nargin<2
                recompute = true;
            elseif  nargout>1
                error('need to recompute covariances when gradient is needed');
            end

            nMod = length(obj);
            grad_sf  = cell(1,nMod);
            with_grad = (nargout>1); % compute gradient

            %% project to basis functions
            obj = project_to_basis(obj);

            %% evaluate prior covariance
            for m=1:length(obj)

                nD = obj(m).nDim; % number of dimensions for this module
                rk = obj(m).rank; % rank
                gsf = cell(rk,nD);
                for d=1:nD
                    if size(obj(m).Prior,1)==1 || isempty(obj(m).Prior(2,d).CovFun)
                        % if more than rank 1 but same covariance prior for
                        % all
                        rrr = 1;
                    else
                        rrr = rk;
                    end
                    for r=1:rrr
                        P = obj(m).Prior(r,d); % prior structure
                        W = obj(m).Weights(d);
                        nW = W.nWeight;
                        HH = obj(m).HP(d);
                        if ~recompute && ~isempty(P.PriorCovariance)
                            % do nothing we already have the covariance
                            % prior
                            Sigma = P.PriorCovariance;
                        else
                            if isa(P.CovFun, 'function_handle') || iscell(P.CovFun) % function handle(s)

                                if with_grad % gradient
                                    [Sigma, gsf{r,d}] = evaluate_prior_covariance_function(P, HH, W);
                                else
                                    Sigma = evaluate_prior_covariance_function(P, HH, W);

                                end
                            elseif isempty(P.CovFun) % default (fix covariance)
                                Sigma = [];
                                gsf{r,d} = zeros(nW,nW,0);
                            else % fixed custom covariance
                                Sigma = P.CovFun;
                                gsf{r,d} = zeros(nW,nW,0);
                            end
                        end
                        if ((size(Sigma,1)~=nW) || (size(Sigma,2)~=nW)) && ~isempty(Sigma)
                            error('covariance prior, dimension %d and rank %d should be square of size %d',d,r,nW);
                        end
                        obj(m).Prior(1,d).PriorCovariance = Sigma;
                    end

                    %end

                    if rk>1 && (rrr==1) % rank>1 and same covariance function and HPs for each rank
                        % replicate covariance matrices
                        for r=2:rk
                            obj(m).Prior(r,d).PriorCovariance = obj(m).Prior(1,d).PriorCovariance;
                        end

                        % extend gradient
                        if with_grad
                            this_nHP = sum(contains(HH.type,"cov")); % number of covariance HPs
                            gsf_new = zeros(nW*rk,nW*rk,this_nHP);
                            for l=1:this_nHP
                                gg = repmat({gsf{1,d}(:,:,l)},rk,1);
                                gsf_new(:,:,l) = blkdiag( gg{:});
                            end
                            gsf{1,d} = gsf_new;
                        end
                    end

                    % add zeros for non-covariance HPs
                    g_tmp = gsf{d};
                    gsf{d} = zeros(size(g_tmp,1),size(g_tmp,2), length(HH.HP));
                    gsf{d}(:,:,contains(HH.type, "cov")) = g_tmp;
                end

                grad_sf{m} = gsf(:)';
            end

            % gradient over module
            if with_grad
                grad_sf = [grad_sf{:}]; % concatenate over modules
                GHP = blkdiag3(grad_sf{:}); % overall gradient of covariance matrices w.r.t hyperparameters
            end

        end

        %% UPDATE PRIOR COVARIANCE
        function [obj, GHP] = update_prior_covariance(obj)
            for m=1:length(obj)
                % clear current value of prior covariance
                for i=1:numel(obj(m).Prior)
                    obj(m).Prior(i).PriorCovariance = {};
                end

                % compute again
                [obj(m), GHP]= compute_prior_covariance(obj);
            end

        end

        %% COMPUTE PRIOR MEAN
        function obj = compute_prior_mean(obj, redo)
            if nargin<2
                redo = true;
            end

            % if more than one object, compute iteratively
            if length(obj)>1
                for i=1:numel(obj)
                    obj(i) = compute_prior_mean(obj(i), redo);
                end
                return;
            end
            nD = obj.nDim; % dimension in this module
            rk = obj.rank;

            for d=1:nD % for each dimension
                for r=1:rk
                    W = obj.Weights(d);
                    nW = W.nWeight;
                    if  ~isempty(obj.Prior(r,d).PriorMean) && ~redo
                        mu = obj.Prior(r,d).PriorMean;
                        assert( size(mu,2) ==nW, 'incorrect length for prior mean');
                        %if isvector(Mu) && rk>1
                        %    Mu = repmat(Mu,rk,1);
                        %end
                    else
                        S = constraint_structure(W);
                        if S.nConstraint>0 && S.nConstraint<nW % if there is a constraint (but not a full fixed weight set)
                            obj.Prior(r,d).PriorMean = S.u*pinv(full(S.V));
                        else
                            obj.Prior(r,d).PriorMean = zeros(1,nW);
                        end
                    end
                end
            end
        end

        %% GLOBAL PRIOR COVARIANCE
        function sigma = global_prior_covariance(obj)
            % put all posterior covariances in one cell array
            P = {obj.Prior};
            P = cellfun(@(x) x(:), P, 'unif',0);
            P = cat(1,P{:});
            sigma = {P.PriorCovariance};

            % global prior cov is the block-diagonal matrix
            sigma = blkdiag(sigma{:});
        end

        %% PRIOR COVARIANCE CELL
        function C = prior_covariance_cell(obj, do_group)
            C = cell(size(obj));
            for i=1:numel(obj)
                C{i} = cell(size(obj(i).Prior));
                for d=1:numel(obj(i).Prior)
                    C{i}{d} = obj(i).Prior(d).PriorCovariance;
                end
            end

            % group prior covariance from all modules
            if nargin>1 && do_group
                C = cellfun(@(x) x(:)', C, 'unif',0);
                C = [C{:}];
            end
        end

        %% PRIOR MEAN CELL
        function C = prior_mean_cell(obj)
            C = cell(size(obj));
            for i=1:numel(obj)
                C{i} = cell(size(obj(i).Prior));
                for d=1:numel(obj(i).Prior)
                    C{i}{d} = obj(i).Prior(d).PriorMean;
                end

            end

        end

        %% INITIALIZE WEIGHTS
        function obj = initialize_weights(obj, obs)
            if nargin<2
                obs = 'binomial';
            end
            obj = obj.check_prior_covariance; % make sure prior covariance is defined
            obj = obj.compute_prior_mean;

            % if more than one object, compute iteratively
            if length(obj)>1
                for i=1:numel(obj)
                    obj(i) = initialize_weights(obj(i), obs);
                end
                return;
            end

            for r=1:obj.rank
                for d=1: obj.nDim
                    W = obj.Weights(d); %weight structure
                    UU = W.PosteriorMean;
                    nW = W.nWeight;
                    constraint = constraint_structure(W);
                    if isempty(UU) || size(UU,1)<r
                        if first_update_dimension(obj)==d && constraint.type~="fixed" && ~strcmp(obs, 'normal')
                            % if first dimension to update, leave as nan to
                            % initialize by predictor and not by weights (for
                            % stability of IRLS)
                            UU = nan(1,nW);

                            % except constrained weights, set to their
                            % values
                            [ct_weight, ct_u] = constrained_weight(W);
                            UU(ct_weight) = ct_u;

                        elseif constraint.type=="fixed"
                            % fixed weights (if not provided by user, ones)
                            % warning('Fixed set of weights for component %d and rank %r not provided, will use zeros',d,r);
                            UU = ones(1,nW);
                        elseif constraint.type=="free"
                            if d>1 && any(constraint_type(obj.Weights(1:d-1)) =="free")
                                % if there's a lower free dimesion, sample from prior (multivariate
                                % gaussian)
                                P = obj.Prior(r,d);
                                UU = mvnrnd(P.PriorMean,P.PriorCovariance);
                            else % only the first free basis is set to 0 (otherwise just stays at 0)
                                UU = zeros(1,nW);
                            end
                        else %  under constraint
                            % by
                            UU =  constraint.u*pinv(full(constraint.V));

                            if d>1 && all(UU==0) && any(cellfun(@(x) all(x(:)==0),  {obj.Weights(1:d-1).PosteriorMean}))
                                % make sure we avoid the case of
                                % weights along two dimensions set to all
                                % zeros
                                UU = mvnrnd(P.PriorMean,P.PriorCovariance);
                            end
                        end
                        obj.Weights(d).PosteriorMean(r,:) = UU;
                    elseif size(UU,2) ~= W.nWeight
                        error('number of weights for component %d (%d) does not match number of corresponding covariates (%d)',d, size(UU,2),ss(d));
                    elseif isvector(UU) && iscolumn(UU)
                        obj.Weights(d).PosteriorMean = UU';
                    end
                end
            end
        end

        %% SAMPLE WEIGHTS FROM PRIOR
        function obj = sample_weights_from_prior(obj)
            for i=1:numel(obj)
                PP = ProjectionMatrix(obj(i));

                for r=1:obj(i).rank
                    for d=1:obj(i).nDim
                        P = obj(i).Prior(r,d);
                        % cc = obj(i).Weights(d).constraint(r);
                        W = obj(i).Weights(d);
                        C = constraint_structure(W);
                        if C.type == "fixed"
                            % fixed weights: set to 1
                            obj(i).Weights(d).PosteriorMean(r,:) = ones(1,obj(i).Weights(d).nWeight);
                        else

                            if d==first_update_dimension(obj(i))
                                % first dimension to update should be zero (more generally: prior mean)
                                UU = P.PriorMean;
                            else % other dimensions are sampled for mvn distribution with linear constraint
                                pp = PP{r,d};
                                Sigma = pp*P.PriorCovariance*pp';
                                if ~issymmetric(Sigma) % sometimes not symmetric for numerical reasons
                                    Sigma = (Sigma+Sigma')/2;
                                end
                                UU =  P.PriorMean + mvnrnd(zeros(1,size(pp,1)), Sigma)*pp;
                            end

                            % offset to make sure that it fullfils the
                            % constraint
                            if C.nConstraint==1 && C.type == "mean1"
                                UU = UU/mean(UU); % all weight set to one
                            elseif C.nConstraint==1 && C.type == "sum1"
                                UU = UU/sum(UU); % all weights equal summing to one
                            elseif C.nConstraint>0

                                %general formula
                                UU = UU + (C.u - UU*C.V)*pinv(full(C.V));
                            end

                            obj(i).Weights(d).PosteriorMean(r,:) = UU;

                        end
                    end
                end
            end

        end

        %% SAMPLE WEIGHTS FROM POSTERIOR
        function [obj,U] = sample_weights_from_posterior(obj)
            % R = R.sample_weights_from_posterior() sample weights from
            % posterior. The sampled weights are placed in field 'PosteriorMean'.
            %
            %[R, U] = sample_weights_from_posterior(R);
            % U is the vector of weights
            for i=1:numel(obj)
                % PP = ProjectionMatrix(obj(i));

                for r=1:obj(i).rank
                    for d=1:obj(i).nDim
                        %  P = obj(i).Prior(r,d);
                        W = obj(i).Weights(d);
                        if isempty(W.PosteriorCov) && W.nWeight>0
                            error('Posterior has not been computed. Infer model first');
                        end
                        if constraint_type(W)~="fixed" % except for fixed weights
                            UU = mvnrnd(W.PosteriorMean, W.PosteriorCov);
                            obj(i).Weights(d).PosteriorMean(r,:) = UU;
                        end
                    end
                end
            end

            if nargout>1
                U = M.concatenate_weights;
            end
        end

        %% COMPUTE POSTERIOR FOR TEST DATA
        function [mu, S,K] = posterior_test_data(obj, scale, d)
            % [mu, S, K] = posterior_test_data(M, scale)
            % returns the mean mu, standard deviation S and full covariance K of the posterior distribution
            % for values scale of transformation of regressor M.
            %
            %  posterior_test_data(M, scale, d) to specify over which dimension of transformation (default:1).
            %
            % Use scale = {x1values, x2values...} to use meshgrid scales
            if nargin<3
                d=1;
            end

            assert(length(obj)==1);
            assert(isscalar(d) && d<=obj.nDim, 'd must be a scalar integer no larger than the dimensionality of M');

            obj = obj.project_from_basis;

            W = obj.Weights(d);
            P = obj.Prior(d);

            [mu,S,K] = computes_posterior_test_data(W,P,obj.HP(d),scale);
        end

        %% COMPUTE LOG-PRIOR (unnormalized)
        function LP = LogPrior(obj,dims)
            % LogPrior(M) computes log-prior of regressor M
            %
            % LogPrior(M,D) computes log-prior of regressor M
            % for weights along dimension(s) D
            if nargin<2
                dims =1:obj.nDim;
            end

            LP = zeros(obj.rank,length(dims)); % for each rank
            for d=1:length(dims)
                d2 = dims(d);
                for r=1:obj.rank % assign new set of weight to each component
                    if constraint_type(obj.Weights(d2))~="fixed" % unless fixed weights
                        P = obj.Prior(r,d2); % corresponding prior
                        dif =  obj.Weights(d2).PosteriorMean(r,:) - P.PriorMean; % distance between MAP and prior mean

                        % ignore weights with infinite variance (their prior will always be null)
                        inf_var = isinf(diag(P.PriorCovariance));
                        if any(~inf_var)
                            dif = dif(~inf_var);
                            LP(r,d) = - dif * (P.PriorCovariance(~inf_var,~inf_var) \ dif') /2; % log-prior for this weight
                        end
                    end

                end
            end

        end

        %% first dimension to be updated in IRLS
        function d = first_update_dimension(obj)
            % start update with first free dimension
            d = find(any(isFreeWeightSet(obj),1),1);

            % if no one then start with first dim
            if isempty(d)
                d = 1;
            end
        end

        %% COMPUTE PREDICTOR (RHO)
        function  rho = Predictor(obj, rr)
            % compute predictor from regressor
            % rho = Predictor(R)
            % rho = Predictor(R,r) to compute for specific rank
            if isempty(obj.Data)
                error('cannot compute predictor, data has been cleared');
            end
            rho = zeros(obj.nObs,1);
            % nW = cellfun(@sum, {obj.Weights.nWeight});
            nW = [obj.Weights.nWeight];

            if any(nW==0) % empty regressor
                return;
            end
            if nargin<2 % default: use all ranks
                rr = 1:obj.rank;
            else
                assert(all(ismember(rr, 1:obj.rank)));
            end
            for r=rr
                rho = rho + ProjectDimension(obj,r,zeros(1,0)); % add activation due to this component
            end

        end

        %% PROJECT REGRESSOR DIMENSION
        function P = ProjectDimension(obj, r, d, do_squeeze, Uobs, to_double)

            if nargin<6 % by default, always convert output to double if sparse array
                to_double = true;
            end

            X = obj.Data;
            VV = cell(1,obj.nDim);
            for dd = setdiff(1:obj.nDim,d) % dimensions to be collapsed (over which tensor product must be computed)
                VV{dd} = obj.Weights(dd).PosteriorMean(r,:);
            end

            if nargin<4 % by default, squeeze resulting matrix
                do_squeeze = 1;
            end
            if nargin<5 % by default, do not collapse over observations
                Uobs = [];
            end

            VV = [{Uobs} VV];

            P = tensorprod(X,VV);

            if do_squeeze
                P = squeeze(P);
                if obj.nObs==1 && size(P,1)~=1
                    P =  shiftdim(P,-1);
                end

                % if singleton tensor matrix along non-projected dimension
                % (not too sure where this is useful)
                dd=1;
                while dd<=length(d) && obj.Weights(d(dd)).nWeight==1
                    P =  shiftdim(P,-1);
                    dd= dd+1;
                end

            end

            if to_double && isa(P, 'sparsearray')
                P = matrix(P);
            end
        end


        %% COMPUTE GRADIENT OF LLH W.R.T. BASIS FUNCTION HYPERPARAMETERS
        function P = compute_gradient_LLH_wrt_basis_fun_hps(obj, err, gradLLH, dY_drho)

            if gradLLH % directly gradient of LLH w.r.t HP
                P =[];
            else
                P = {};
                drho_dHP = zeros(obj(1).nObs,0); % derivative of rho w.r.t HP
            end

            for i=1:numel(obj)
                for d=1:obj(i).nDim
                    W = obj(i).Weights(d);
                    B = W.basis;
                    HPs = obj(i).HP(d);
                    nHP = length(HPs.HP);
                    if any(contains(HPs.type,'basis') & HPs.fit) && ~all([B.fixed]) %% for any fittable HP parametrizes basis functions

                        basisHP = find(contains(HPs.type, "basis"));

                        % compute basis functions and gradient of basis fun w.r.t HP
                        [~,~, gradB] = compute_basis_functions(B, B(1).scale, HPs);

                        %                         % compute gradient of basis fun w.r.t HP
                        %                         if isrow(B.scale)
                        %                             [B.B,~,~,gradB] = B.fun(B.scale, HPs.HP, B.params);
                        %                         else
                        %                             % more rows in scale means we fit different
                        %                             % functions for each level of splitting
                        %                             % variable
                        %                             [id_list,~,split_id] = unique(B.scale(2:end,:)','rows'); % get id for each observation
                        %                             B.B = zeros(0,size(B.scale,2)); % the matrix will be block-diagonal
                        %
                        %                             gradB = zeros(0,size(B.scale,2),nHP);
                        %                             for g=1:length(id_list)
                        %                                 subset = split_id==g; % subset of weights for this level of splitting variable
                        %                                 [this_B,~,~, this_gradB] = B.fun(B.scale(1,subset), HPs.HP, B.params);
                        %                                 n_new = size(this_B,1);
                        %                                 B.B(end+1 : end+n_new, subset) = this_B; %
                        %                                 gradB(end+1 : end+n_new, subset,:) = this_gradB;
                        %                             end
                        %
                        %                         end

                        % original data
                        X = obj(i).DataOriginal;
                        VV = cell(1,obj(i).nDim+1);
                        VV{1} = err';
                        for dd = setdiff(1:obj(i).nDim,d) % dimensions to be collapsed (over which tensor product must be computed)
                            VV{dd+1} = obj(i).Weights(dd).PosteriorMean;
                        end

                        if gradLLH % multiply with weights (gradient of LLH)
                            gradB = tensorprod(gradB, {W.PosteriorMean,[],[]});
                            gradB = permute(gradB,[3 2 1]);

                            VV{d+1} = gradB;
                            this_P = tensorprod(X,VV);
                            if isa(this_P, 'sparsearray')
                                this_P = matrix(this_P);
                            end
                            PP = zeros(1,nHP); % add zeros for non-basis function HP
                            PP(basisHP) = this_P;
                            P = [P PP];
                        else

                            this_P = zeros(W.nWeight,nHP);
                            for h=1:length(basisHP)
                                VV{d+1} = gradB(:,:,h);
                                this_P(:,basisHP(h)) = tensorprod(X,VV);
                            end
                            P{end+1} = this_P;

                            % to compute other part of the term
                            WW = VV;
                            WW{1} = []; % do not project over observations
                            this_drho_dHP = zeros(obj(i).nObs, nHP);
                            for h=1:length(basisHP)
                                WW{d+1} = W.PosteriorMean*gradB(:,:,h);
                                this_drho_dHP(:,basisHP(h)) = tensorprod(X,WW);
                            end
                            drho_dHP = [drho_dHP this_drho_dHP];
                        end

                    elseif gradLLH
                        % no basis function for this set of weight, direct grad
                        P = [P zeros(1,nHP)];
                    else
                        this_P = zeros(W.nWeight,nHP);
                        P{end+1} = this_P;

                        drho_dHP = [drho_dHP zeros(obj(1).nObs,nHP)];
                    end
                end
            end

            if ~gradLLH
                % part of the matrix linked with derivate of basis function
                P = blkdiag(P{:});

                % now add other part linked to derivative of prediction error
                dY_dHP = dY_drho .*drho_dHP; % derivative of predicted value w.r.t HP
                DM = design_matrix(obj,[],0);
                P = P - DM'*dY_dHP;
            end


        end


        %% CLEAR DATA (E.G. TO FREE UP MEMORY)
        function obj = clear_data(obj)
            for i=1:numel(obj)
                obj(i).Data = [];
                obj(i).DataOriginal = [];
            end
        end


        %% EXTRACT REGRESSOR FOR A SUBSET OF OBSERVATIONS
        function obj = extract_observations(obj,subset) %,dim)
            % M = M.extract_observations(subset); extracts observations for regressor
            % object

            for i=1:numel(obj)
                S = size(obj(i).Data);
                obj(i).Data = obj(i).Data(subset,:);
                n_Obs = size(obj(i).Data,1); % update number of observations
                obj(i).Data = reshape(obj(i).Data, [n_Obs, S(2:end)]); % back to nd array
                if ~isempty(obj(i).DataOriginal)
                    S = size(obj(i).DataOriginal);
                    obj(i).DataOriginal = obj(i).DataOriginal(subset,:);
                    obj(i).DataOriginal = reshape(obj(i).DataOriginal, [n_Obs, S(2:end)]); % back to nd array
                end
                obj(i).nObs = n_Obs;
            end
        end

        %% PRODUCT OF REGRESSORS
        function obj = times(obj1,obj2, varargin)
            % R = R1 * R2 or R = times(R1,R2);
            %   multiplies regressors for GUM
            %
            % R = times( R1, R2, 'SharedDim') if R1 and R2
            % share all dimensions. Weights, priors and hyperparameters for
            % new object are inherited from M1.
            %
            % R = times( R1, R2, 'SharedDim', D)
            %to specify dimensions shared by M1 and M2 (D is a vector of integers)
            %
            % See also gum, regressor

            SharedDim = [];
            if not(isempty(varargin))
                if isequal(varargin{1}, 'SharedDim')
                    if length(varargin)==1
                        assert(obj1.nDim==obj2.nDim, 'number of dimensions of both regressors should be equal');
                        SharedDim = 1:obj1.nDim;
                    elseif length(varargin)==2
                        SharedDim = varargin{2};
                    end
                    SharedDimOk = isnumeric(SharedDim) && isvector(SharedDim) && ...
                        all(SharedDim>0) && all(SharedDim<obj1.nDim) && all(SharedDim<obj2.nDim);
                    assert(SharedDimOk, 'SharedDim must be a vector of dimensions');
                    %  nWeight1 = cellfun(@sum, {obj1.Weights(SharedDim).nWeight});
                    %  nWeight2 = cellfun(@sum, {obj2.Weights(SharedDim).nWeight});
                    nWeight1 = [obj1.Weights(SharedDim).nWeight];
                    nWeight2 = [obj2.Weights(SharedDim).nWeight];
                    assert(all(nWeight1== nWeight2),...
                        'the number of weights in both regressors are different along shared dimensions');
                    assert(length(varargin)<3, 'too many inputs');
                else % R1 * R2 * R3...
                    obj = times(times(obj1,obj2), varargin{:}); % recursive call, compute as (M1*M2)*M3...
                    return;
                end
            end

            % multiplication of regressor object with vector is easy
            if isnumeric(obj1)
                %if ndims(obj1)>=ndims(obj2.val) % sum over new dimensions
                %    obj1 = regressor2(obj1,'constant');
                %else

                obj2.Data = obj1 .* obj2.Data; % just multiply data
                obj = obj2;
                return;
            elseif isnumeric(obj2)
                % obj2 = regressor2(obj2,'constant');
                obj1.Data = obj2 .* obj1.Data;
                obj = obj1;
                return;
            end

            if obj1.nObs ~= obj2.nObs
                error('regressors should have the same number of observations');
            end
            obj = regressor();
            obj.nObs = obj1.nObs;
            obj.nDim = obj1.nDim + obj2.nDim - length(SharedDim); % new dimensionality is simply sum of both

            %% build design tensor
            if isnumeric(obj2.Data) && issparse(obj2.Data)
                obj2.Data = sparsearray(obj2.Data); % conver to sparse array class to allow for larger than 2 arrays
            end

            NotSharedDim = setdiff(1:obj2.nDim, SharedDim); % we keep all weights, priors, HP from M2 except for shared dimensions

            % permute second object to use different dimensions for each
            % object (except first dim = observations & other shared dimensions)
            P = 1:obj1.nDim+1; % original dimensions
            OldDims = NotSharedDim+1;
            NewDims = obj1.nDim+1+(1:length(NotSharedDim));
            P(OldDims) = NewDims; % send non-shared dimensions at the end
            P(NewDims) = OldDims; % and use singletons
            %  P = [1 obj2.nDim+1+(1:obj1.nDim) 2:obj2.nDim+1];
            obj2.Data = permute(obj2.Data, P);

            % pairwise multiplication
            if isa(obj2.Data, 'double') && ~isa(obj1.Data, 'double')
                % make sure we don't run into problems if multiplying
                % integers with doubles
                obj1.Data = double(obj1.Data);
            end
            obj.Data = obj1.Data .* obj2.Data;

            %% other properties (if present in at least one regressor)
            obj.Weights = [obj1.Weights obj2.Weights(NotSharedDim)];
            obj.Prior = [obj1.Prior obj2.Prior(NotSharedDim)];
            obj.HP = [obj1.HP obj2.HP(NotSharedDim)];

            % formula (add parentheses if needed)
            if any(obj1.formula=='+')
                obj1.formula = ['(' obj1.formula ' )'];
            end
            if any(obj2.formula=='+')
                obj2.formula = ['(' obj2.formula ' )'];
            end
            obj.formula = [obj1.formula ' * ' obj2.formula];

            % for categorical regressors: turn "first0" into "free" or "first1", i.e.
            % the reference weight is 1
            isFirst0 =  constraint_type(obj.Weights)=="first0";
            for w = find(isFirst0)
                if w==find(isFirst0,1) % first becomes free
                    obj.Weights(w).constraint = "free";
                elseif isstruct(obj.Weights(w).constraint) % others become "first1"
                    obj.Weights(w).constraint.u(:) = 1;
                    obj.Weights(w).constraint.type = "first1";
                else
                    obj.Weights(w).constraint = "first1";
                end
            end

            FreeRegressorSet = isFreeWeightSet(obj);
            if sum(FreeRegressorSet)>1
                % turning free regressors in second object to one-mean regressor
                for d=find(FreeRegressorSet(obj1.nDim+1:end))
                    obj.Weights(obj1.nDim+d).constraint = "mean1";
                end
            end
        end

        %% WHETHER WEIGHTS ARE FREE (NOT CONSTRAINED)
        function bool = isFreeWeightSet(obj)
            assert(numel(obj)==1, 'only for scalar regressor object');
            bool = constraint_type(obj.Weights)=="free";
        end

        %% WHETHER WEIGHTS ARE FIXED (FULLY CONSTRAINED)
        function bool = isFixedWeightSet(obj)
            assert(numel(obj)==1, 'only for scalar regressor object');
            bool = constraint_type(obj.Weights)=="fixed";
        end


        % R1 * R2 is the same as R1 .* R2
        function obj = mtimes(varargin)
            obj = times(varargin{:});
        end

        %% SUM OF REGRESSORS
        function obj= plus(obj1,varargin)
            % R = R1 + R2 sums regressors (equivalent to concatenation)
            % R = R1 + R2 + ...
            %
            % If more than one regressor is continuous of dimension 1, the constraints of all but the first one are changed to zero-sum to avoid identifiability issues with the likelihood.

            obj2 = varargin{1};
            if length(varargin)>1 % R1 + R2 + R3 ...
                obj = plus(obj1+obj2,varargin{:}); % recursive call
                return;

            end

            if isa(obj1,'regressor')
                nO1 = [obj1.nObs];
                if ~all(nO1==nO1(1))
                    error('number of observations is not consistent within first set of regressors');
                end
                nO1 = nO1(1);
            end
            if isa(obj2,'regressor')
                nO2 = [obj2.nObs];
                if ~all(nO2==nO2(1))
                    error('number of observations is not consistent within second set of regressors');
                end
                nO2 = nO2(1);
            end

            if isnumeric(obj1)
                if isscalar(obj1)
                    obj1 = obj1*ones(nO2,1);
                end
                obj1 = regressor(obj1,'linear');
            elseif isnumeric(obj2)
                if isscalar(obj2)
                    obj2 = obj2*ones(nO1,1);
                end
                obj2 = regressor(obj2,'linear');
            else

                if nO1 ~= nO2
                    error('sum of regressors: regressors should have the same number of observations');
                end
            end

            % If both regressors are dimension 1 and continuous type and no constraint
            if any([obj1.nDim] ==1) && any([obj2.nDim]==1)
                i1 = [obj1.nDim] ==1;
                i2 = find([obj2.nDim] ==1);
                W1 = [obj1(i1).Weights];
                W2 = [obj2(i2).Weights];
                free_continuous1 = strcmp({W1.type}, 'continuous') & constraint_type(W1)=="free";
                free_continuous2 = strcmp({W2.type}, 'continuous') & constraint_type(W2)=="free";

                if    any(free_continuous1) && any(free_continuous2)
                    %  change second constraint to zero-sum to avoid identifiability issues with the likelihood.
                    chg = i2(free_continuous2);
                    for i=chg
                        obj2(i).Weights.constraint = "nullsum";
                    end
                end
            end

            % concatenate
            obj = [obj1 obj2];
        end

        %% INTERACTIONS (FOR CATEGORICAL REGRESSORS)--> x1:x2
        function obj = colon(obj1, obj2)

            if ~isa(obj2,'regressor')
                obj2 = regressor(obj2, 'categorical');
            end
            assert(isscalar(obj1) && isscalar(obj2), 'interaction should be between scalar regressor objects');
            assert(obj2.nObs==obj1.nObs, 'number of observations do not match');
            assert(obj1.nDim ==1 && obj2.nDim ==1 && strcmp(obj1.Weights.type,'categorical') && strcmp(obj2.Weights.type,'categorical'),...
                'regressors must be one-dimensional categorical');

            obj = obj1;

            nLevel1 = obj1.Weights.nWeight; % number of levels for each regressor
            nLevel2 = obj2.Weights.nWeight;
            nLevel = nLevel1 * nLevel2; % number of interaction inf_terms

            % build design matrix
            X = zeros(obj1.nObs, nLevel);
            if isa(obj1.Data, 'sparsearray')
                obj1.Data = matrix(obj1.Data);
            end
            if isa(obj2.Data, 'sparsearray')
                obj2.Data = matrix(obj2.Data);
            end
            for i=1:nLevel2
                idx = (i-1)*nLevel1 + (1:nLevel1);
                X(:,idx) = obj1.Data .* obj2.Data(:,i); % multiply covariates
            end
            obj.Data = X;

            % scale
            scale = interaction_levels(obj1.Weights.scale, obj2.Weights.scale);

            obj.Weights  = empty_weight_structure(1, [obj1.nObs nLevel], scale, []);
            obj.Weights.label = obj1.Weights.label + ":" + obj2.Weights.label;

            %% constraint (zeros for all default)
            if isFreeWeightSet(obj1)
                obj1.Weights.constraint = "first0";
            end
            if isFreeWeightSet(obj2)
                obj2.Weights.constraint = "first0";
            end
            S1 = constraint_structure(obj1.Weights); % extract structure for each constraint
            S2 = constraint_structure(obj2.Weights);

            if S1.type==S2.type
                S.type = S1.type;
            else
                S.type = "mixed";
            end

            % replicate projection vectors
            V1 = kron(eye(nLevel2),S1.V);
            u1 = repmat(S1.u,1,nLevel2);

            % we don't replicate all combinations for constraint 2 to avoid
            % duplicating with previous matrix
            repmatrix2 = full(compute_orthonormal_basis_project(S1.V))';
            V2 = kron(S2.V, repmatrix2);
            u2 = repmat(S2.u,1,size(repmatrix2,2));
            S.V = [V1 V2];
            S.u =  [u1 u2];

            S.nConstraint = size(S.V,2);

            obj.Weights.constraint = S;

        end

        %% MAIN EFFECTS AND INTERACTION (FOR CATEGORICAL REGRESSORS) --> x1^x2 (in R-formula: x1*x2)
        function obj = mpower(obj1, obj2)

            if ~isa(obj2,'regressor')
                obj2 = regressor(obj2, 'categorical');
            end

            % main effects plus interaction terms
            obj = obj1 + obj2 + (obj1:obj2);

        end

        %% SPLIT/CONDITION ON REGRESSORS
        function obj = split(obj,X, scale, label)
            % R = split(R, X)
            % splits regressor R for each value for categorical variable X.
            % X should be a column vector with as many observations as in
            % R.
            %
            % split(R, X, scale) to include only the values of X in
            % vector scale
            %
            % split(R, X, [], label) or split(R, X, scale, label) to provide labels for levels of
            % X
            if nargin<3
                scale = [];
            end

            % check that X has appropriate dimensions
            if size(X,1)~=obj.nObs
                error('number of rows in X does not match number of observations in regressor');
            end
            for d=1:obj.nDim
                if ~size(X,d+1)==1 && ~size(X,d+1)==obj.Weights(d).nWeight
                    error('incorrect size of X along dimension %d',d+1);
                end
            end

            % unique values of X
            unq = unique(X); % unique values
            if isnumeric(unq)
                unq(isnan(unq)) = []; % remove nan values
            end
            if ~isempty(scale)
                if any(~ismember(scale,unq))
                    warning('exclude scale values not present in data');
                    scale(~ismember(scale,unq)) = [];
                end
            else
                scale = unq';
            end

            % add label if required
            if nargin>=4
                if length(label) ~= length(scale)
                    error('length of label does not match number of values');
                end
            else
                label = scale;
            end

            nVal = length(scale);
            if nVal==1 % if just one value,let's keep it identical
                return;
            end

            X = replace(X, scale); % replace each value in the vector by its index

            %% replicate design matrix along other dimensions
            for dd = 1:obj.nDim

                if isa(obj.Data,'sparsearray') && subcoding(obj.Data,dd+1) %obj.sparse(dd)

                    shift = (X-1)*obj.Weights(dd).nWeight;  % shift in each dimension (make sure we use a different set of indices for each value along dimension d
                    obj.Data.sub{dd+1} = obj.Data.sub{dd+1} +  uint8(shift);
                    obj.Data.siz(dd+1) = obj.Data.siz(dd+1)*nVal;
                else
                    VV = obj.Data;
                    new_size = size(VV);
                    new_size(dd+1) = new_size(dd+1)*nVal; % each regressor is duplicated for each value along dimension d
                    obj.Data = zeros(new_size);

                    idx = cell(1,ndims(VV));
                    for ee = 1:ndims(VV)
                        idx{ee} = 1:size(VV,ee);
                    end

                    for r=1:nVal
                        idx{1} = find(X==r); % focus on r-th value
                        idx2 = idx;
                        idx2{dd+1} = (r-1)*size(VV,dd+1) + idx{dd+1};
                        obj.Data(idx2{:}) = VV(idx{:}); % copy content
                    end
                end

                %% update scale
                W = obj.Weights(dd);
                W.scale = interaction_levels(W.scale, label);
                if all(isnan(W.scale(1,:)))
                    W.scale(1,:) = [];
                end

                %% update constraint
                if ~any(constraint_type(W)==["free","fixed"])
                    S = constraint_structure(W);
                    S.V = kron(eye(nVal),S.V); % replicate for each set of weights
                    S.u = repmat(S.u, 1,nVal);
                    S.nConstraint = S.nConstraint * nVal;
                    W.constraint = S;
                end

                W.nWeight = W.nWeight * nVal; % one set of weights for each level of splitting variable
                obj.Weights(dd) = W;

                %% build covariance function as block diagonal
                if ~isempty(obj.Prior(dd).CovFun)
                    obj.Prior(dd).replicate = nVal * obj.Prior(dd).replicate;
                end
            end
        end

        %% SPLIT DIMENSION
        function obj = split_dimension(obj,D)
            % R = R.split_dimension(D) splits dimension D

            summing = cell(1,obj.nDim);
            summing(D) = {'split'};
            obj = split_or_separate(obj, summing);
        end

        %% SPLIT OR SEPARATE
        function   obj = split_or_separate(obj, summing)

            % whether splitting or separate regressors options are
            % requested
            SplitOrSeparate = strcmpi(summing, 'split') | strcmpi(summing, 'separate');
            SplitOrSeparate(end+1:obj.nDim) = false;

            %% if splitting along one dimension (or separate into different observations)
            SplitDims = fliplr(find(SplitOrSeparate));

            W = obj.Weights;

            if length(SplitDims)==1 && SplitDims==obj.nDim && obj.nDim==2 && strcmpi(summing{obj.nDim}, 'split') && ~(isa(obj.Data,'sparsearray') && subcoding(obj.Data,3))
                %% exactly the same as below but should be way faster for this special case

                % reshape regressors as matrices
                %    nW = cellfun(@sum, {W.nWeight});% number of weight along each dimension
                nW = [W.nWeight];% number of weight along each dimension
                nWeightCombination = prod(nW); % number of weight combinations
                obj.Data = reshape(obj.Data,[obj.nObs nWeightCombination]);

                if isa(obj.Data, 'sparsearray') && ismatrix(obj.Data)
                    obj.Data = matrix(obj.Data); % convert to basic sparse matrix if it is 2d now
                end

                % how many times we need to replicate covariance function
                obj.Prior(1).replicate = W(2).nWeight * obj.Prior(1).replicate;

                obj.Weights(1).nWeight = W(1).nWeight*W(2).nWeight; % one set of weights in dim 1 for each level of dim 2
                obj.Weights(1).scale = interaction_levels(W(1).scale, W(2).scale);
            else

                for d=SplitDims
                    nRep = W(d).nWeight;

                    % replicate design matrix along other dimensions
                    for dd = setdiff(1:obj.nDim,d)

                        if isa(obj.Data,'sparsearray') && subcoding(obj.Data,dd+1) && strcmpi(summing{d}, 'split')
                            % if splitting dimension is encoding with
                            % OneHotEncoding, faster way
                            shift_value = obj.Data.siz(dd+1) * (0:nRep-1); % shift in each dimension (make sure we use a different set of indices for each value along dimension d
                            shift_size = ones(1,1+obj.nDim);
                            shift_size(d) = nRep;
                            shift = reshape(shift_value,shift_size);


                            obj.Data.sub{dd+1} = obj.Data.sub{dd+1} +  shift;
                        else % general case
                            X = obj.Data;

                            % size of new array
                            new_size = size(X);
                            new_size(dd+1) = new_size(dd+1)*nRep; % each regressor is duplicated for each value along dimension d
                            if strcmpi(summing{d}, 'separate')
                                new_size(1) = new_size(1)*nRep; % if seperate observation for each value along dimension d
                            end

                            % preallocate memory for new data array
                            if ~issparse(X)
                                obj.Data = zeros(new_size);
                            elseif length(new_size)<2
                                obj.Data = spalloc(new_size(1),new_size(2), nnz(X));
                            else
                                obj.Data = sparsearray('empty',new_size, nnz(X));
                            end

                            % indices of array positions when filling up
                            % array
                            idx = cell(1,ndims(X));
                            for ee = 1:ndims(X)
                                idx{ee} = 1:size(X,ee);
                            end

                            for r=1:nRep % loop through all levels of splitting dimension
                                idx{d+1} = r; % focus on r-th value along dimension d
                                idx2 = idx;
                                idx2{d+1} = 1;
                                idx2{dd+1} = (r-1)*size(X,dd+1) + idx{dd+1};
                                if strcmpi(summing{d}, 'separate')
                                    idx2{1} = (r-1)*size(X,1) + idx{1};
                                end
                                obj.Data(idx2{:}) = X(idx{:}); % copy content
                            end
                        end

                        obj.Weights(dd).scale = interaction_levels(W(dd).scale, W(d).scale);

                        % build covariance function as block diagonal
                        obj.Prior(dd).replicate = nRep * obj.Prior(dd).replicate;
                        obj.Weights(dd).nWeight = W(dd).nWeight*nRep; % one set of weights in dim dd for each level of splitting dimension
                    end

                    % remove last dimension
                    idx = repmat({':'},1,ndims(obj.Data));
                    idx{d+1} = 2:size(obj.Data,d+1); % keep this dimension as singleton
                    Str = struct('type','()','subs',{idx});
                    obj.Data = subsasgn(obj.Data, Str, []);

                    if isa(obj.Data, 'sparsearray') && ismatrix(obj.Data)
                        obj.Data = matrix(obj.Data); % convert to standard sparse matrix if it is two-D now
                    end

                end

                if ~isempty(SplitDims)
                    % reshape data array
                    NonSplitDims = setdiff(1:obj.nDim, SplitDims); % non-splitting dimensions
                    obj.Data = reshape(obj.Data, [size(obj.Data,1)  obj.Weights(NonSplitDims).nWeight]);
                end
            end

            % if separate: update number of observations (multiply by
            % number of levels along each splitting dimension)
            SeparateDims = strcmpi(summing, 'separate');
            obj.nObs = obj.nObs*prod([obj.Weights(SeparateDims).nWeight]);

            % remove corresponding Weights and Prior and HP for splitting
            % dimensions, and update dimensionality
            obj.Weights(SplitDims) = [];
            obj.Prior(SplitDims) = [];
            obj.HP(SplitDims) = [];
            obj.nDim = obj.nDim - length(SplitDims);
        end


        %% BUILD REGRESSORS WITH LAGGED VALUES
        function obj = laggedregressor(obj, Lags, varargin)
            % R = R.laggedregressor(Lags) builds a regressor object with lagged value of input
            % regressor. Lags is a vector for how to look for past
            % trials. E.g. if Lags = [1 2 3], it will create a regressor for each
            % feature at previous observation (t-1), at t-2 and at t-3 separately.
            % Use cell array for Lags to group regressor with same weight, e.g Lags =
            % {1, 2, 3,4:5, 6:10};
            % R = R.laggedregressor() is equivalent to R = R.laggedregressor(1), i.e.
            % creates a regressor
            %
            % R = R.laggedregressor(Lags, argument1, value1, ...) with possible
            % arguments:
            % 'group': value G is a group indicator vector indicating the group corresponding to each observation to build
            % regressors separately for each group.
            % - 'basis': using basis functions for lag weights (e.g. 'exp3'). Default:
            % no basis.
            % - 'PlaceLagFirst': boolean specifying whether lags weights are placed as
            % the first dimension (default) or last
            % - 'SplitValue': boolean array specifying whether a different kernel is
            % computed for other dimensions (explain better!). Default: true.
            %
            % Some example of how to use lagged regressor in formula,
            % y  ~ x + lag(y+z; Lags=1:3)
            % lag(y+z; Lags=1,2; group=subject)
            % lag(...; basis = exp2)
            % lag(...; split=false)
            % lag(...; placelagfirst=false)


            if nargin<2
                Lags = 1;
            end

            if numel(obj)>1
                % recursive call
                for i=1:numel(obj)
                    obj(i) = laggedregressor(obj(i), Lags, varargin{:});
                end
                return;
            end

            % default value options
            basis = ''; % '', 'expN'
            Group = [];
            PlaceLagFirst = true;
            SplitValue = true;

            nO = obj.nObs;

            % process options
            for v=1:2:length(varargin)
                switch lower(varargin{v})
                    case 'basis'
                        basis = varargin{v+1};
                    case 'group'
                        Group = varargin{v+1};
                    case 'placelagfirst'
                        PlaceLagFirst = varargin{v+1};
                    case 'splitvalue'
                        SplitValue = varargin{v+1};
                    otherwise
                        error('incorrect option:%s', varargin{v});
                end
            end

            if isempty(Group) % default: only one group
                Group = ones(nO,1);
            else
                assert(isvector(Group),'Group should be a vector of group indices');
                assert(all(diff(Group)>=0), 'indices in Group should be monotically non-decreasing');
            end
            Gunq = unique(Group); % group unique values
            nGroup = length(Gunq); % number of group

            single_regressor = obj.nDim==1 && obj.Weights(1).nWeight ==1;
            if  single_regressor % just one single regressor, no way to split it
                SplitValue = [];
            elseif isscalar(SplitValue)
                SplitValue = repmat(SplitValue, 1, obj.nDim);
            end

            D = obj.Data;
            %if issparse(D) && ~isvector(D) % if two-d we'll convert to 3 d so use sparsearray format
            %    D = sparsearray(D);
            %end

            S = size(D);

            nLag = length(Lags); % number of lags


            % create data with lagged regressor(for now, last dimension is lag)
            if issparse(D)
                if single_regressor
                    Dlag = spalloc(S(1),nLag,nnz(S)*nLag);
                else
                    Dlag =  sparsearray('empty',[S nLag],nnz(S)*nLag);
                end
            elseif single_regressor
                Dlag = zeros([S(1) nLag]);
            else
                Dlag = zeros([S nLag]);
            end

            if ~iscell(Lags)
                LagLevels = Lags;
                Lags = num2cell(Lags);
            else
                LagLevels = LevelString(Lags);
            end

            % cell array of data indices
            Cin = cell(1,ndims(D));
            Cout = cell(1,ndims(Dlag));
            for d=2:ndims(D)
                Cin{d} =1:S(d);
                Cout{d} =1:S(d);
            end


            for l=1:nLag % for each lag group
                Cout{end} = l; % position in last dimension of output array

                for idx = 1:length(Lags{l}) % for each lag within that group
                    lg = Lags{l}(idx); % this lag

                    for g=1:nGroup % for each observation group

                        Cin{1} = find(Group==Gunq(g)); % observations for this group
                        Cout{1} = Cin{1};
                        if lg>0
                            Cin{1}(end-lg+1:end) = []; % remove last observations according to lag
                        elseif lg<0 % negative lag
                            Cin{1}(1:-lg) = []; % remove first observations according to lag
                        end

                        this_D = D(Cin{:}); % input data for this group

                        % pad with zeros
                        if isa(D, 'sparsearray')
                            Dzeros = sparsearray('empty',[abs(lg) S(2:end)]);
                        elseif issparse(D)
                            Dzeros = spalloc(abs(lg),S(2));
                        else
                            Dzeros = zeros([abs(lg) S(2:end)]);
                        end

                        if lg>0 % positive lag: place zeros first
                            this_D = cat(1,Dzeros, this_D);
                        elseif lg<0 % negative lag: place zeros last
                            this_D = cat(1,this_D,Dzeros);
                        end

                        % place in output data
                        if idx==1
                            Dlag(Cout{:}) = this_D;
                        else % if more than one lag in lag group, add regressor values
                            Dlag(Cout{:}) = this_D + Dlag(Cout{:});
                        end

                    end
                end
            end
            obj.Data = Dlag;

            % add weights, prior and HPs
            nDim_new = ndims(Dlag)-1;
            obj.nDim = nDim_new;
            scale = cell(1,nDim_new);
            scale{nDim_new} = LagLevels;
            summing  = cell(1,nDim_new);
            if ~isempty(basis)
                summing{nDim_new} = 'continuous';
            end
            basis = [cell(1,nDim_new-1) {basis}];

            obj = define_priors_across_dims(obj, nDim_new, summing, {}, [], scale, basis, true, []);
            obj.Weights(nDim_new).label = "lag";
            if any(SplitValue) || isempty(SplitValue)
                obj.Weights(nDim_new).constraint = "free";
            else % if no splitting, i.e multidimensional regressor, need to set some constraint
                obj.Weights(nDim_new).constraint = "mean1";
            end

            % place dimension for lag-weights first
            if PlaceLagFirst
                ord = [nDim_new 1:nDim_new-1];
                obj = obj.permute(ord);
            end

            if any(SplitValue)
                SplitDims = find(SplitValue) + PlaceLagFirst;
                obj = obj.split_dimension(SplitDims);
            end
            obj.formula = "lag("+ obj.formula +")";
        end


        %% CONCATENATE REGRESSORS
        function obj = cat_regressor(obj, D, check_dims_only)
            % M = cat_regressor(M);
            % vector of regressors is turned into a single regressor
            % M = cat_regressor(M, D); concatenates along dimension D (default: D=1)
            %
            % bool =  cat_regressor(obj, D, 1) returns a boolean telling if
            % the concatenation operation is possible (if dimensions of regressors are compatible)

            n = length(obj); % number of regressors
            if n==1 % if only one regressor, don't need to do anything
                if nargin>=3 && check_dims_only
                    obj = true;
                end
                return;
            end

            if nargin<2
                D = 1;
            end
            if nargin<3
                check_dims_only = false;
            end

            n_Obs = obj(1).nObs;
            assert(all([obj.nObs]==n_Obs), 'number of observations do not match');

            nD = max([obj.nDim]);
            if nD==1 && check_dims_only
                obj = true;
                return;
            end

            %% deal with formula ( x1 + x2...)
            fmla = string({obj.formula});
            fmla(2,:) = "+";
            fmla = cellstr(fmla(1:end-1));
            obj(1).formula = [fmla{:}];

            %% process special case when all are single linear regressors (e.g. x1+x2+...)
            W = [obj.Weights];
            all_single_linear = nD==1 && all([W.nWeight]==1) && all(strcmp({W.type},'linear')) ...
                && all(cellfun(@(x) isequal(x,obj(1).Prior), {obj.Prior})) ...
                && all(constraint_type(W)==constraint_type(W(1)));
            if all_single_linear
                obj(1).Data = cat(2,obj.Data); % concatenate regressors
                obj(1).Weights.scale = [W.label];
                obj(1).Weights.nWeight = numel(obj);
                obj(1).Weights.nFreeWeight = sum([W.nFreeWeight]);
                obj = obj(1);
                return;
            end

            for i=1:n
                if obj(i).nDim<nD
                    obj(i) = add_dummy_dimension(obj(i), nD);
                end
            end

            % make sure that dimensionality matches along all other
            % dimensions
            nWeight = ones(n,nD);
            for i=1:n
                nWeight(i,1:obj(i).nDim) = [obj(i).Weights.nWeight];
            end
            otherdims = setdiff(1:nD,D);
            check_dims = all(nWeight(:,otherdims)==nWeight(1,otherdims),'all'); % check that dimensions allow for concatenation
            if check_dims_only % return only boolean telling whether regressors can be concatenated
                obj = check_dims;
                return;
            end

            assert(check_dims, 'number of elements in regressor does not match along non-concatenating dimensions');
            % nWeight(1,D) = sum(nWeight(:,D)); % number of elements along dimension D are summed
            nWeight_cat = nWeight(:,D);  % number of elements along dimension D are concatenated
            nWeight = nWeight(1,:); % select from first regressor

            % update properties of first element
            obj(1).nDim = nD;
            if nD>2 && ~isa(obj(1).val, 'sparsearray')
                obj(1).Data = sparsearray(obj(1).Data);
            end
            obj(1).Data = cat(D+1, obj.Data); % concatenate all regressors

            %% deal with prior
            PP = obj(1).Prior(D);
            allPrior = {obj.Prior};

            % merge prior mean
            F = cellfun(@(x) x(D).PriorMean ,allPrior, 'unif',0);
            if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                PP.PriorMean = [];
            else % otherwise concatenate
                PP.PriorMean  = cat(2,F{:});
            end

            nRep = cellfun(@(x) x(D).replicate,allPrior);
            if any(nRep~=1)
                error('not coded for replicated priors')
            end

            % deal with covariance prior function
            PP.CovFun = cellfun(@(x) x(D).CovFun ,allPrior, 'unif',0);
            PP.type = cellfun(@(x) x(D).type ,allPrior, 'unif',0); %'mix';

            % deal with prior covariance
            F = cellfun(@(x) x(D).PriorCovariance ,allPrior, 'unif',0);
            if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                PP.PriorCovariance = {};
            else % otherwise build block diagonal matrix
                PP.PriorCovariance = blkdiag(F{:});
            end
            obj(1).Prior(D) = PP;

            %% merge fields of weight structure
            W = obj(1).Weights(D);
            W.U_allstarting = [];

            fild = { 'PosteriorMean','PosteriorStd','T','p', 'scale'};
            allWeights = {obj.Weights};
            allWeights = cellfun(@(x) x(D), allWeights);
            W.label = [allWeights.label];
            W.nFreeWeight = sum([allWeights.nFreeWeight]);
            for f = 1:length(fild)
                fn = fild{f};
                %  F = cellfun(@(x) x(D).(fn) , allWeights,'unif',0);
                F = {allWeights.(fn)};
                if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                    W.(fn) = [];
                else % otherwise concatenate
                    W.(fn) = cat(2,F{:});
                end
            end

            % deal with scale
            F = {allWeights.scale};
            scale_dim = max(cellfun(@(x) size(x,1),F));
            for i=1:n
                if isempty(F{i})
                    F{i} = 1:obj(i).nWeight(D);
                end
                F{i}(end+1:scale_dim,:) = nan; % fill extra dimensions with nan if needed
                F{i}(scale_dim+1,:) = i; % add extra dimension with group index
            end

            W.scale = cat(2,F{:});
            if all(isnan(W.scale(1,:))) % e.g. if sum of scalar linear regressors
                W.scale(1,:)= [];
            end

            % deal with posterior covariance
            F = {allWeights.PosteriorCov};
            if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                W.PosteriorCov = {};
            else % otherwise build block diagonal matrix
                W.PosteriorCov = blkdiag(F{:});
            end

            % deal with constraint
            if ~all(constraint_type(allWeights)=="free") && ~all(constraint_type(allWeights)=="fixed")
                S_ct = constraint_structure(allWeights);

                % convert first0 to first1 (assuming we're going this is a
                % categorical regressor and we're going to multiply it)
                for w = find( [S_ct.type]=="first0")
                    S_ct(w).u(:) = 1;
                    S_ct(w).constraint.type = "first1";
                end

                ctype = "mixed";
                V = blkdiag(S_ct.V);
                u = [S_ct.u];
                nConstraint = sum([S_ct.nConstraint]);
                W.constraint = struct('type',ctype, 'V',V,'u',u, 'nConstraint',nConstraint);
            end

            % deal with basis
            Bcell = {allWeights.basis};
            noBasis = cellfun(@isempty,Bcell); % regressor without basis function
            if all(noBasis)
                W.basis = [];
            else
                B(~noBasis) = [Bcell{:}]; % structure array
                for i=find(noBasis)
                    B(i).fun = 'none';
                    B(i).nWeight = nWeight_cat(i);
                end
            end
            W.basis = B;
            W.nWeight = sum(nWeight_cat);

            for d=1:nD
                obj(1).Weights(d).nWeight = nWeight(d);
            end
            obj(1).Weights(D) = W;

            %% deal with hyperparameters
            F = {obj.HP};
            F = cellfun(@(x) x(D), F); % select D-th element for each regressor
            obj(1).HP(D).index = [];
            obj(1).HP(D) = cat_hyperparameters(F);

            % keep only first object
            obj = obj(1);
        end

        %% PERMUTE
        function obj = permute(obj, P)
            % R = R.permute(ORDER) rearranges the dimensions of regressor R so that they are in
            %the order specified by the vector ORDER.

            assert(isvector(P) && length(P)>=obj.nDim, 'ORDER must be a vector of length equal to dimensionality of R (or larger)');

            if length(P)>obj.nDim % add dummy dimensions if needed
                obj = obj.add_dummy_dimension(length(P));
            end

            P = round(P);
            assert(all(P>0) && all(P<=obj.nDim), 'ORDER contains an invalid permutation index');
            assert(length(unique(P))==obj.nDim, 'ORDER contains an invalid permutation index');

            obj.Data = permute(obj.Data, [1 P+1]);

            % re-order
            obj.Weights = obj.Weights(P);
            obj.HP = obj.HP(P);
            obj.Prior = obj.Prior(P);

        end

        %% ADD DUMMY DIMENSION
        function obj = add_dummy_dimension(obj, D)
            % R=R.add_dummy_dimension(D);

            if D>obj.nDim+1 % if need to add more than one dummy dimension, recursive call
                obj = add_dummy_dimension(obj, D-1);
            end

            if D<obj.nDim+1 % if not added as last dimension
                % insert dimension one in data array
                P = [1:D obj.nDim+2 D+1:obj.nDim+1]; % permutation order
                obj.Data = sparsearray(obj.Data);
                obj.Data = permute(obj.Data, P);
            end

            P = [1:D-1 obj.nDim+1 D:obj.nDim]; % permutation order

            % add dummy weights, HP
            obj.Weights(obj.nDim+1).nWeight = 1;
            obj.Weights(obj.nDim+1).constraint = "fixed";
            obj.HP(end+1) = HPstruct;
            obj.Prior(end+1) = empty_prior_structure(1);

            % re-order
            obj.Weights = obj.Weights(P);
            obj.HP = obj.HP(P);
            obj.Prior = obj.Prior(P);

            obj.nDim = obj.nDim+1;

        end


        %% PROJECT OBSERVATIONS (FOR TIME REGRESSORS, PROJECT EVENT SPACE TO TIME SPACE)
        function obj = project_observations(obj, P)
            % R = project_observations(R, P)
            %project observations (for time regressors, project event space
            %to time space)

            % use tensor product to project first dimension (observation)
            % to new space
            C = cell(1,obj.nDim+2);
            C{1} = P;
            obj.Data = tensorprod(obj.Data, C);
            obj.nObs = size(P,1); % update number of observations

        end

        %% PLOT FITTED WEIGHTS
        function h = plot_weights(obj, U2, varargin)
            % plot_weights(R) plot weights from regressor R
            % R.plot_weights()
            % R.plot_weights("regressor1")
            % R.plot_weights(["regressor1","regressor2",...])
            %
            % R.plot_weights(..., h); to use subplots defined by handles
            %
            % nSub = M.plot_weights(..., 'nsubplot') to have number of
            % subplots


            if nargin<2 || isequal(U2, 'nsubplot') || isempty(U2)
                if nargin>=2 && isequal(U2, 'nsubplot')
                    varargin = {U2};
                end
                U2 = [];
            end

            % select corresponding regressors to be plotted
            if isempty(U2)
                idx = find_weights(obj);
            elseif isnumeric(U2) % already providing matrix
                assert(size(U2,1)==2, 'U2 should be a matrix of 2 rows (first row: regressor index; second row: dimensionality');
                idx = U2;
            else
                idx = find_weights(obj, U2);
            end

            only_nsubplot = ~isempty(varargin) && isequal(varargin{1}, 'nsubplot');
            cols = defcolor;
            colmaps = {'jet','hsv','winter','automn'};

            NoFixedWeights = zeros(1,size(idx,2));
            for j=1:size(idx,2) % for each set of regressor
                m = idx(1,j);
                isFW = isFixedWeightSet(obj(m));
                d = idx(2,j);
                NoFixedWeights(j) = ~isFW(d) * nRegressorSet(obj(m).Prior(d)); % regressors with constant (count also concatenated regs)
            end
            idx(:,~NoFixedWeights) = []; % remove fixed weights
            nSubplots = sum(NoFixedWeights); % number of subplots

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

            for j=1:size(idx,2) % loop between subpots
                m = idx(1,j);
                d = idx(2,j);

                W = obj(m).Weights(d); % set of weights
                P = obj(m).Prior(d);
                this_HP = obj(m).HP(d);

                % add label
                if isempty(W.label)
                    W.label = "U"+m+"_"+d; % default label
                end

                % split concatenated structures
                [W, this_HP, P] = split_concatenated_weights_HP_prior(W, this_HP,P);

                % plot weights
                for s=1:length(W)
                    axes(h.Axes(i)); % set as active subplot

                    [h_nu, c, cm] = plot_single_weights(W(s), P(s), this_HP(s), obj(m).rank, c, cm, cols, colmaps);

                    h.Objects = [h.Objects {h_nu}];
                    i = i+1; % update subplot counter
                end
            end
        end


        %% EXPORT WEIGHTS TO TABLE
        function T = export_weights_to_table(obj)
            % T = M.export_weights_to_table(obj);
            % creates a table with metrics for all weights (scale, PosteriorMean, PosteriorStd, T-statistics, etc.)
            W = [obj.Weights]; % concatenate weights over regressors
            P = [obj.Prior];
            HPs = [obj.HP];

            [W,~,P] = split_concatenated_weights_HP_prior(W,HPs,P);

            reg_idx = repelem(1:length(W), [W.nWeight])'; % regressor index
            label = repelem([W.label],[W.nWeight])'; % weight labels

            T = table(reg_idx, label, 'VariableNames',{'regressor','label'});

            filds = {'scale','PosteriorMean','PosteriorStd','T','p'};
            for f=1:length(filds)
                ff =  concatenate_weights(obj,0,filds{f}); % concatenate values over all regressors
                if length(ff) == height(T)
                    T.(filds{f}) = ff';
                elseif f==1 % scale
                    all_scale_string = [];
                    for i=1:length(W)
                        sc = W(i).scale;
                        if isempty(sc)
                            sc = nan(1,W(i).nWeight);
                        end
                        if size(sc,1)>1
                            scale_string = "(";
                            for d=1:size(sc,1)
                                scale_string = scale_string + string(sc(d,:));
                                if d==size(sc,1)
                                    scale_string = scale_string + ")";
                                else
                                    scale_string = scale_string + ",";
                                end
                            end
                        elseif isnan(sc)
                            scale_string = "";
                        else
                            scale_string = string(sc);
                        end
                        all_scale_string = [all_scale_string; scale_string'];

                    end
                    T.scale = all_scale_string;
                end
            end

            % prior mean
            PM = {P.PriorMean};
            for i=1:length(W) % if using basis function, project prior mean on full space
                if ~iscell(W(i).basis) && ~isempty(W(i).basis)
                    B = W(i).basis.B;
                    PM{i} = PM{i}*B;
                end
            end
            T.PriorMean = [PM{:}]';
        end

        %% EXPORT WEIGHTS TO CSV FILE
        function export_weights_to_csv(obj, filename)
            % M.export_weights_to_csv(filename) exports weights data as csv file.
            %
            T = obj.export_weights_to_table;
            writetable(T, filename);
        end

        %% EXPORT HYPERPARAMETERS TO TABLE
        function T = export_hyperparameters_to_table(obj)
            % T = M.export_hyperparameters_to_table(obj);
            % creates a table with metrics for all hyperparameters

            H = [obj.HP];
            % nD = [obj.nDim];
            W = [obj.Weights];
            transform = [W.label];
            nHP = cellfun(@length, {H.HP}); % number of HPs for each transformatio
            transform = repelem(transform, nHP)';

            nW = cellfun(@length, {obj.Weights}); % number of weight set per regressor
            nW_cum = [0 cumsum(nW)];
            nHP_cum = [0 cumsum(nHP)];
            nHP_reg = nHP_cum(nW_cum(2:end)+1) - nHP_cum(nW_cum(1:end-1)+1); % total number of HPs per regressor
            RegressorId = repelem(1:length(obj), nHP_reg)';

            value = [H.HP]';
            label = [H.label]';
            fittable = [H.fit]';
            UpperBound = [H.UB]';
            LowerBound = [H.LB]';
            if isfield(H, 'std')
                standardDeviation = [H.std]';
                T = table(RegressorId, transform, label, value, fittable, LowerBound, UpperBound, standardDeviation);
            else
                T = table(RegressorId, transform, label, value, fittable, LowerBound, UpperBound);
            end

        end

        %% EXPORT WEIGHTS TO CSV FILE
        function export_hyperparameters_to_csv(obj, filename)
            % M.export_hyperparameters_to_csv(filename) exports hyperparameter data as csv file.
            %
            T = obj.export_hyperparameters_to_table;
            writetable(T, filename);
        end

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% END OF METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PLOT SINGLE WEIGHTS
function [h_nu, c, cm] = plot_single_weights(W, P, HP, rk, c, cm, cols, colmaps)
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
    sc  = scale{1};
    ColorPlot = 0;
    if ColorPlot
        % color plot
        h_nu = colorplot(sc(:,1),sc(:,2), real(U));

    else % plot all values in a grid

        % get posterior from values in a grid
        xValue = linspace(min(sc(:,1)),max(sc(:,1)), 50);
        yValue = linspace(min(sc(:,2)),max(sc(:,2)), 50);
        % [U,se, ~,scale] = computes_posterior_test_data(W,P,HP,{xValue,yValue});

        U = computes_posterior_test_data(W,P,HP,{xValue,yValue});

        U = reshape(U, length(xValue), length(yValue));

        %     se = reshape(se, length(xValue), length(yValue));

        % convert to RGB with whiter where more uncertainty
        %    cmap = colormap;
        %    Unorm = (U-min(U,[],'all'))/(max(U,[],'all')-min(U,[],'all'));

        hold on;

        % plot image
        h_nu(1) = imagesc(xValue, yValue, U');

        % add black dots representing data points in trained dataset
        h_nu(2) =plot(sc(:,1),sc(:,2),'k.', 'MarkerSize',2);
    end

elseif imageplot
    % image plot
    h_nu = imagescnan(scale{2},scale{1}, U);
else
    if isscalar(scale{1}) && isnan(scale{1})
        scale{1} = 1;
    end

    % cufve/bar plot with errors
    [~,~,h_nu] = wu([],U,se,scale,plot_opt{:});
end

% add horizontal line for reference value
if ~twodmap && ~imageplot && W.nWeight >= 8
    hold on;
    switch constraint_type(W)
        case "mean1"
            y_hline = 1;
        case "sum1"
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
S1 = S(:,1);
S2 = S(:,2);
x_unq = unique(S(:,1));
if isstring(x_unq) && all(~isnan(double(x_unq)) | strcmpi(x_unq,'nan'))
    % in case string of numeric values, convert to numeric values to have
    % numeric sorting
    S1 = double(S1);
    x_unq = unique(S1);
end
y_unq = unique(S2);
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
        bool = equal_numeric_or_cell(S1,x_unq(iX)) & equal_numeric_or_cell(S2,y_unq(iY));
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

% compare cell or numeric arrays
function bool = equal_numeric_or_cell(X,Y)
if isnumeric(X)
    bool = X==Y;
else
    bool = strcmp(X,Y{1});
end

end

%% CREATES A DESIGN MATRIX (OR ARRAY) WITH ONE-HOT ENCODING - ONE COLUMN PER VALUE OF X
function val = one_hot_encoding(X, scale,OneHotEncoding, nD)

nVal = size(scale,2);

[X, siz] = replace(X,scale); % replace each value in the vector by its index

f_ind = zeros(siz);
nFun = 1;

sub = X + nVal*f_ind;
nValTot = nVal*nFun; % total number of values (dimension of array along dimension nD+1)

if OneHotEncoding % create a sparse array
    val = sparsearray(ones(siz)); % array of one
    val = add_onehotencoding(val,nD+1, sub, nValTot); % set indices on dimension nD+1 to values of data

else % non-sparse
    % size of diesgn matrix
    if iscolumn(X), siz = length(X);
    else
        siz = size(X);
    end
    siz(end+1) = nVal*nFun;

    % index for matrix
    idx = cell(1,length(siz));
    for i=1:length(siz)-1
        idx{i} = 1:siz(i);
    end

    val = zeros(siz,'uint8'); % use integer coding
    for u=1:nVal % one column for each value
        for w=1:nFun
            idx{end} = u + nVal*(w-1);
            val(idx{:}) = (X==u) & (f_ind==w-1); % true if corresponding observation value and corresponding column
        end
    end
end
end

%% compute posterior over test data
function [mu,S,K,scale] = computes_posterior_test_data(W,P,HP,scale)

assert(  any(strcmp(W.type, {'continuous','periodic'})), 'test data only for continuous or periodic regressors');


if iscell(scale) % define meshgrid

    % compose along n-dimensional grid
    GridCoords = cell(1,length(scale));
    [GridCoords{:}] = ndgrid(scale{:});

    % stores as matrix
    GridCoords = cellfun(@(x) x(:)', GridCoords, 'UniformOutput',0);
    scale = cat(1,GridCoords{:});

end

assert(size(scale,1)==size(W.scale,1), 'scale must has the same number of rows as the fitting scale data');
nW = size(scale,2); % number of prediction data points

B = W.basis;

if isempty(B) % no basis functions: use equation 15
    % evaluate prior covariance between train and test set
    isCovHP = contains(HP.type, "cov");
    KK = P.CovFun({W.scale,scale},HP.HP(isCovHP), []);

    MatOptions = struct('POSDEF',true,'SYM',true); % is symmetric positive definite
    mu = linsolve(P.PriorCovariance,W.PosteriorMean',MatOptions)' * KK;

    if nargout>1
        Kin = P.CovFun(scale,HP.HP(isCovHP), []);
        rk = size(W.invHinvK,3); % rank
        K = zeros(nW, nW, rk);
        S = zeros(rk, nW);
        for r=1:rk
            K(:,:,r) = Kin - KK'*W.invHinvK(:,:,r)*KK;
            S(r,:) = sqrt(diag(K))';
        end
    end

else
    %% if using basis functions

    % compute projection matrix
    isBasisHP = contains(HP.type, "basis");
    BB = B.fun(scale, HP.HP(isBasisHP), B.params);

    % project mean and covariance in original domain
    mu = B.PosteriorMean * BB;

    if nargout>1
        % compute posterior covariance and standard error
        K = zeros(nW, nW,obj.rank);
        S = zeros(obj.rank, nW);
        for r=1:obj.rank
            PCov  = BB' * B.PosteriorCov(:,:,r) * BB;
            K(:,:,r) = PCov;
            S(r,:) = sqrt(diag(PCov))'; % standard deviation of posterior covariance in original domain
        end
    end

end

end

%% evaluate prior covariance function and gradient over covariance HPs
function [Sigma, grad] = evaluate_prior_covariance_function(P, HP, W)

% hyperparameter values for this component
nRep = P.replicate;
if  ~isempty(nRep) && nRep>1 % if we replicate the covariance matrix, we need to remove splitting dimensions are repetitions
    % we don't know which of the last
    % rows(s) in scale code for the
    % splitting variable, so let's find out
    nRep_check = 1;
    iRow = size(W.scale,1)+1; % let's start from no row at all

    while nRep_check<nRep && iRow>1
        iRow=iRow-1; % move one row up
        [id_list,~,split_id] = unique(W.scale(iRow:end,:)','rows'); % get id using last rows

        nRep_check = size(id_list,1); % number of unique sets
    end
    assert(nRep_check==nRep, 'weird error, could not find out splitting rows in scale');

    W.scale(iRow:end,:) =[]; % remove these rows for computing prior covariance
    W.scale = W.scale(:,split_id==1); % to avoid repetitions of value

end

nSet = nRegressorSet(P); % number of weights sets (more than one if uses regressor concatenation)
if nSet ==1
    HP.index = ones(1,length(HP.HP));
    index_weight = ones(1,W.nWeight);
    P.CovFun = {P.CovFun};
else
    index_weight = W.scale(end,:);
    W.scale(end,:) = [];
    if ~iscell(P.CovFun)

        P.CovFun = repmat({P.CovFun},1,nSet);
    end
end
Sigma = cell(1,nSet);
grad = cell(1,nSet);

for s=1:nSet
    W.nWeight = sum( index_weight==s);
    %  index_weight = iWeight(s)+1:iWeight(s+1);
    this_scale = W.scale(:, index_weight==s); % scale for this set of weights
    if nSet>1
        this_scale(end,:) = []; % remove dimension encoding set index
    end
    iHP = HP.index ==s & contains(HP.type, "cov"); % covariance hyperparameter for this set of weight

    CovFun = P.CovFun{s}; % corresponding prior covariance function

    if isempty(W.basis) ||  isequal(W.basis(s).fun, 'none')
        B = [];
    else
        B = W.basis(s);
    end

    if nargout>1 % need gradient
        [Sigma{s}, gg]= CovFun(this_scale,HP.HP(iHP),B);

        % replicate covariance matrix if needed
        [Sigma{s}, gg]= replicate_covariance(nRep, Sigma{s},gg);
        if isstruct(gg)
            gg = gg.grad;
        end

        % nHP = length(HP.HP); % number of hyperparameters
        nHP = sum(iHP);% number of hyperparameters
        if size(gg,3) ~= nHP
            error('For component %d and rank %d, size of covariance matrix gradient along dimension 3 (%d) does not match corresponding number of hyperparameters (%d)',...
                d,r, size(gg,3),nHP);
        end

        %% compute gradient now
        grad{s} = zeros(size(gg)); % gradient of covariance w.r.t hyperparameters
        for l=1:nHP
            freeW = ~constrained_weight(W); % exclude fixed weights
            freeSigma = Sigma{s}(freeW,freeW);
            MatOptions = struct('POSDEF',true,'SYM',true); % is symmetric positive definite
            SigmaInvGrad = linsolve(freeSigma,  gg(freeW,freeW,l),MatOptions);
            grad{s}(freeW,freeW,l) = - linsolve(freeSigma, SigmaInvGrad,MatOptions )';
        end

        % select gradient only for fittable HPs
        HP_fittable = logical(HP.fit(iHP));
        grad{s} = grad{s}(:,:,HP_fittable);
    else
        Sigma{s}= CovFun(this_scale, HP.HP(iHP), B);

        % replicate covariance matrix if needed
        Sigma{s} = replicate_covariance(nRep, Sigma{s});
    end
end

% merge over sets of weights
Sigma = blkdiag(Sigma{:});
grad = blkdiag3(grad{:});
end

%% REPLICATE COVARIANCE MATRIX (AND GRADIENT) WHEN SPLITTING REGRESSOR
function [cov, grad] = replicate_covariance(nRep,cov,gg)

if isempty(nRep)|| nRep==1
    if nargout>1
        grad = gg;
    end
    return;
end


% full covariance matrix is block
% diagonal
cov = repmat({cov}, 1, nRep);
cov = blkdiag(cov{:});

% compute gradient
if nargout>1
    if isstruct(gg)
        gg = gg.grad;
    end
    siz = size(gg); % size of gradient matrix
    if length(siz)==2
        siz(3) = 1;
    end
    grad = zeros(siz(1)*nRep, siz(2)*nRep,siz(3));

    for hp=1:siz(3)
        gc = repmat({gg(:,:,hp)}, 1, nRep);
        grad(:,:,hp) = blkdiag(gc{:});
    end
end
end

%% CREATE EMPTY WEIGHT STRUCTURE
function W = empty_weight_structure(nD, DataSize, scale, color)
W = struct('type','','label', '', 'nWeight',0, 'nFreeWeight',0,...
    'PosteriorMean',[], 'PosteriorStd',[],'V',[],...
    'PosteriorCov', [], 'T',[],'p',[],...
    'scale',[],'constraint',"free",'plot',[],'basis',[],'invHinvK',[],'U_allstarting',[]);
% other possible fields: U_CV

% make it size nD
W = repmat(W, 1, nD);

if nargin>1
    DataSize(end+1:nD+1) = 0;
    for d=1:nD
        W(d).nWeight = DataSize(d+1); % number of weights/regressors in each dimension
        W(d).nFreeWeight = DataSize(d+1); % number of weights/regressors in each dimension

        % define plotting color
        if ~isempty(color)
            W(d).plot = {'color',color};
        end

    end

    % scale /levels for each dimension
    scale = tocell(scale);
    for d=1:length(scale)
        W(d).scale = scale{d};
    end
end
end

%% empty prior structure
function P = empty_prior_structure(nD)
P = struct('type', '', 'CovFun',[], 'PriorCovariance',[],...
    'PriorMean',[], 'replicate',1, 'spectral',[], 'label','none');

% make it size nD
P = repmat(P, 1, nD);
end

%% concatenate hyperparameter structure
function S = cat_hyperparameters(Sall)
S=struct;

nHP = cellfun(@length, {Sall.HP}); % number of hyperparameters for each set of weights
S.HP = cat(2, Sall.HP);
S.label = cat(2, Sall.label);
S.fit = cat(2, Sall.fit);
S.LB = cat(2, Sall.LB);
S.UB = cat(2, Sall.UB);
S.index = repelem(1:length(Sall), nHP); % index matching each hyperparameter to set of weights
S.type = cat(2,Sall.type);
end

%% split concatenated HP and prior structure
function [W, HP, P] = split_concatenated_weights_HP_prior(W, HP,P)
if ~isscalar(W)
    % recursive call
    Wc = cell(size(W));
    HPc = cell(size(W));
    Pc = cell(size(W));
    for i=1:numel(W)
        [Wc{i}, HPc{i}, Pc{i}] = split_concatenated_weights_HP_prior(W(i), HP(i),P(i));
    end
    W = [Wc{:}];
    HP = [HPc{:}];
    P = [Pc{:}];
    return;
end

if nRegressorSet(P)==1
    return;
end

% split weights first
[W,index_weight] = split_concatenated_weights(W);

nSet = length(W);
Pc = P; % copy values
HPc = HP;

P = repmat(P,1,nSet);
HP = repmat(HP,1,nSet);


for i=1:nSet
    % prior structure
    P(i).CovFun = Pc.CovFun{i};
    if isempty([W.basis])
        P(i).PriorMean = Pc.PriorMean(:,index_weight==i);
        P(i).PriorCovariance = Pc.PriorCovariance(index_weight==i,index_weight==i);
    else % if uses basis function, should go back and compute it in projected space
        P(i).PriorMean = nan(size(Pc.PriorMean,1),sum(index_weight));
        P(i).PriorCovariance = nan(sum(index_weight),sum(index_weight));

    end

    % HP structure
    index_HP = HPc.index==i;
    HP(i).HP = HPc.HP(index_HP);
    HP(i).LB = HPc.LB(index_HP);
    HP(i).UB = HPc.UB(index_HP);
    HP(i).label = HPc.label(index_HP);
    HP(i).fit = HPc.fit(index_HP);
    HP(i).index = [];
end
%HP = rmfield(HP, 'index');
end

%% split concatenated weight structure (for plotting)
function [W,set_index] = split_concatenated_weights(W)
% warning: does not check that weights indeed are concatenated

Wc = W; % copy

set_index = Wc.scale(end,:); % last dim of scale indicate set index
nSet = max(set_index);

W = repmat(W, 1, nSet);

for s=1:nSet
    W(s).label = Wc.label(s);
    W(s).nWeight = sum(set_index==s);
    index_weight = set_index==s;  % indices of weights for this set
    W(s).PosteriorMean = Wc.PosteriorMean(:,index_weight);
    W(s).PosteriorStd = Wc.PosteriorStd(:,index_weight);
    W(s).PosteriorCov = Wc.PosteriorCov(index_weight,index_weight,:);
    W(s).T = Wc.T(:,index_weight);
    W(s).p = Wc.p(:,index_weight);
    W(s).scale = Wc.scale(1:end-1,index_weight); % remove last dim (set index)
    dummy_scale_dim = all(isnan(W(s).scale),2); % dimensions for scale that are not used for this set
    W(s).scale(dummy_scale_dim,:) = [];
    if isempty(Wc.basis) || isequal(Wc.basis(s).fun,'none')
        W(s).basis = [];
        W(s).U_allstarting = Wc.U_allstarting(:,index_weight);
    else
        W(s).basis = Wc.basis(s);
        W(s).U_allstarting = [];
    end
end
end

%% converts to cell if not already a cell
function x= tocell(x,d)

if ~iscell(x)
    x = {x};
end

if nargin>1 && length(x)<d
    x = [x cell(1,d-length(x))];
end
end

%% replace value by index
function [I,siz] = replace(X, scale) % replace X values by indices

siz = size(X);
if size(scale,1)>1 % if using combination of values
    D = siz(end); % dimensionality
    siz(end) = [];
    if length(siz)==1
        siz(2) = 1;
    end
    nX = prod(siz); % number of data points

    % we'll use columns for different values and match them one by one
    I = zeros(siz);
    for u=1:size(scale,2) % find matching data points
        bool = true(nX,1); % start with all data points
        for d=1:D % loop through dimensions
            % keep only if matches this dimension also
            if iscell(scale)
                bool = bool & strcmp(X(:,d),scale{d,u});
            else
                bool = bool & (X(:,d)==scale(d,u));
            end
        end
        % assign index
        I(bool) = u;
    end

else % one-dimensional scale
    I = zeros(siz);
    for u=1:size(scale,2) % replace each value in the vector by its index
        if iscell(scale) % cell array of strings
            I( strcmp(X,scale{u})) = u;
        else
            I( X==scale(u)) = u;
        end
    end
end
end

%% interaction levels (all pairs of levels)
function scale = interaction_levels(scale1, scale2)
% compose all interactions levels (all pairs of combinations from single
% level)

% if single level for any variable, return the other one
if check_single_level(scale1)
    scale = scale2;
    return;
elseif check_single_level(scale2)
    scale = scale1;
    return;

end

if iscell(scale1) && ~iscell(scale2)
    scale2 = num2cell(scale2);
elseif ~iscell(scale1) && iscell(scale2)
    scale1 = num2cell(scale1);
end
nLevel1 = size(scale1,2);
nLevel2 = size(scale2,2);
scale = [repmat(scale1,1,nLevel2); repelem(scale2, 1, nLevel1)];
end

%% check if single level
function bool = check_single_level(scale)
if iscell(scale)
    bool = strcmp(scale, repmat(scale(:,1),1,size(scale,2))); % compare to first value
else
    bool = scale ==scale(:,1);
end
bool = all(bool(:));
end

%% create Level string for lags
function Lvl = LevelString(X)
nLevel = length(X);
Lvl = string(nLevel);

% make levels strings
for l=1:nLevel
    Lvl(l) = string(X{l}(1));


    idx = 2;
    interval = false;
    while idx <= length(X{l})
        if X{l}(idx) ~= X{l}(idx-1)+1 % not a step of one, separate with commas
            if interval % if closing an interval
                Lvl(l) =  Lvl(l) + "-" + X{l}(idx-1);
            end
            Lvl(l) =  Lvl(l) + "," + X{l}(idx);

        elseif ~interval % we're starting interval (e.g. all steps of ones)
            interval = true;
        end
        idx = idx+1;
    end
    if interval % if closing an interval
        Lvl(l) =  Lvl(l) + "-" + X{l}(idx-1);
    end

end
end

%% constraint structure
function S = constraint_structure(W)
for i=1:numel(W)
    C = W(i).constraint;
    nWeight = W(i).nWeight;
    if isstruct(C)
        assert(all(isfield(C,{'V','u'})), 'incorrect constraint structure');
        if ~isfield(C, 'type')
            C.type = "custom";
        end
        if ~isfield(C, 'nConstraint')
            C.nConstraint = length(C.v);
        end
        S(i) = C;
    else

        assert(ischar(C) || isstring(C), 'incorrect constraint field');

        opt_list = ["free";"fixed";"mean1";"sum1";"nullsum";"first0";"first1"];
        old_optlist = 'fnmsb1';
        if length(char(C))==1
            %old format
            C = opt_list(char(C)==old_optlist);
        end


        switch string(C)
            case "free"
                V = spalloc(nWeight,0,0);
                u = [];
            case "fixed"
                V = speye(nWeight);
                u = [];
            case "mean1"
                V = ones(nWeight,1);
                u = nWeight;
            case "sum1"
                V = ones(nWeight,1);
                u = 1;
            case "first0"
                V = [1; zeros(nWeight-1,1)];
                u = 0;
            case "first1"
                V = [1; zeros(nWeight-1,1)];
                u = 1;
            case "nullsum"
                V = ones(nWeight,1);
                u = 0;
            otherwise
                error('incorrect constraint type: %s',C);
        end
        nConstraint = size(V,2);
        S(i) = struct('type',string(C),'V',V, 'u',u,'nConstraint',nConstraint);
    end
end
end

%% constraint type
function str = constraint_type(W)
str = strings(size(W));
for i=1:numel(W)
    S = constraint_structure(W(i));
    str(i) = string(S.type);
end
end

%% constrained weight
function [bool,u] = constrained_weight(W)
% boolean array: for each weight, whether it is constrained to a fixed
% value
S = constraint_structure(W);
one_constrained_weight = (sum(S.V~=0,1)==1); % for each constraint,whether it affects only one weight (i.e.fully constrains it)
bool = any(S.V(:,one_constrained_weight),2);

if nargout>1
    u = S.V(bool, one_constrained_weight) * S.u(one_constrained_weight)';
end
end

%% number of regressor set (for concatenated regressors)
function n = nRegressorSet(P)
n = ones(size(P)); % by default one
isCellCovFun = cellfun(@iscell, {P.CovFun});
for i=find(isCellCovFun) % if basis is cell array, then these are concatenated regressors
    n(i) = length(P(i).CovFun);
end
end