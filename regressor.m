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
    % different factors. C is a character string where character at position d
    % defines the constraint for the corresponding dimension. Possible
    % character values are:
    % - 'f': free, unconstrained (one and only one dimension)
    % - 'b': sum of weights is set to 0
    % - 'm': mean over weights in dimension d is set to 1
    % - 's': sum over weights in dimension d is set to 1
    % - '1': first element in weight vector is set to 1
    % - 'n': fixed weights (not estimated)
    % Default value is 'f' (unconstrained) for first dimension, 'm' for other
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
        formula = ''
        nDim
        nObs
        Weights
        Prior
        HP = HPstruct
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
            constraint = '';
            binning = [];
            label = '';
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
                        assert(ischar(constraint), 'constraint C should be a character array');
                    case 'binning'
                        binning = varargin{v+1};
                        if ~isempty(binning)
                            assert(any(strcmp(type, {'continuous','periodic'})), 'binning option only for continuous or periodic variable');
                        end
                    case 'label'
                        label = varargin{v+1};
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
                                error('incorrect prior type');
                        end

                        %  case 'categorical'
                        %        HPfit_def = 1;
                        %        HPfit_lbl = {'variance'};

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
                label = 'x1';
            end
            if ~iscell(label)
                label_tmp = label;
                label = cell(1,nD);
                for d=1:nD-1
                    label{d} = ['x' num2str(d)];
                end
                label{nD} = label_tmp;
            end

            %% build data and variables depending on coding type
            switch lower(type)
                %% constant regressor (no fitted weight)
                case 'constant'
                    obj.Data = X;
                    obj.nDim = 1;
                    obj.Weights(1).constraint = 'n';
                    % obj.U = {}; %{ones(1,obj.nWeight)};
                    obj.formula = label{1};
                    obj.Prior(1).type = 'none';

                    %% linear regressor
                case 'linear'

                    obj.Data = X;
                    obj.Data(isnan(obj.Data)) = 0; % convert nans to 0: missing regressor with no influence on model
                    % if any(strcmp(summing,{'split','separate'}))
                    %     nWeight = nWeight * prod(siz(3:end));
                    %     obj.Data = reshape(X, siz(1), nWeights); % vector set replicated along dimension 3 and beyond
                    %     obj.nDim = 1;
                    % else
                    %obj.nDim = nD;
                    %  nD = obj.nDim;

                    % end

                    % process hyperparameters
                    HP = tocell(HP,nD);
                    if ~isempty(variance), HP{1} = log(variance)/2; end

                    nWeight = siz(1+nD);
                    
                    if isempty(obj.Weights(nD).scale)
                        scale{nD} = 1:nWeight;
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
                            obj.HP(nD) = HPstruct_ard(nWeight, nD, HP{nD}, HPfit);
                            obj.Prior(nD).CovFun = @ard_covfun;
                    end
                    obj.Prior(nD).type = prior{nD};

                    obj.HP(nD).fit = HPfit;

                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, 1:nD, summing, prior, HP, scale, basis, single_tau, condthresh);
                    % !!! seems to overwrite all commands before, maybe we
                    % can just remove them?

                    obj.formula = label{nD};

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

                    obj.formula = ['cat(' label{nD} ')'];

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

                    obj.formula = ['f(' label{nD} ')'];


                otherwise
                    error('unknown type:%s', type);
            end

            for d=1:nD
                obj.Weights(d).label = label{d};
            end
            obj.Weights(nD).plot = plot;

            % by default, first dimension (that does not split or separate)
            % is free, other have sum-one constraint
            if ~strcmpi(type, 'constant')
                FreeDim = find(~SplitOrSeparate,1);
                for d=1:nD
                    obj.Weights(d).constraint = 's'; % sum-one constraint
                end
                obj.Weights(FreeDim).constraint = 'f';
            end

            % whether components should be reordered according to variance, default: reorder if all components have same constraints
            if obj.rank>1
                obj.ordercomponent = all(all(constraint==constraint(1,:)));
            end

            obj = obj.split_or_separate(summing);

            %             %% if splitting along one dimension (or separate into different observations)
            %             SplitDims = fliplr(find(SplitOrSeparate));
            %
            %             if length(SplitDims)==1 && SplitDims==obj.nDim && obj.nDim==2 && strcmpi(summing{obj.nDim}, 'split') && ~(isa(obj.Data,'sparsearray') && subcoding(obj.Data,3))
            %                 %% exactly the same as below but should be way faster for this special case
            %
            %                 % reshape regressors as matrices
            %                 nWeightCombination = prod([obj.Weights.nWeight]);
            %                 obj.Data = reshape(obj.Data,[obj.nObs nWeightCombination]);
            %
            %                 if isa(obj.Data, 'sparsearray') && ismatrix(obj.Data)
            %                     obj.Data = matrix(obj.Data); % convert to basic sparse matrix if it is 2d now
            %                 end
            %
            %                 % how many times we need to replicate covariance function
            %                 %nRep = obj.Weights(2).nWeight;
            %                 obj.Prior(1).replicate = obj.Weights(2).nWeight * obj.Prior(1).replicate;
            %
            %                 % build covariance function as block diagonal
            %                 %obj.Prior(1).CovFun = @(P)  replicate_covariance(obj.Prior(1).CovFun, P, nRep);
            %
            %                 obj.Weights(1).nWeight = obj.Weights(1).nWeight*obj.Weights(2).nWeight; % one set of weights in dim 1 for each level of dim 2
            %             else
            %
            %                 for d=SplitDims
            %                     nRep = obj.Weights(d).nWeight;
            %
            %                     % replicate design matrix along other dimensions
            %                     for dd = setdiff(1:obj.nDim,d)
            %
            %                         if isa(obj.Data,'sparsearray') && subcoding(obj.Data,dd+1) && strcmpi(summing{d}, 'split')
            %                             % if splitting dimension is encoding with
            %                             % OneHotEncoding, faster way
            %                             shift_value = obj.Data.siz(dd+1) * (0:nRep-1); % shift in each dimension (make sure we use a different set of indices for each value along dimension d
            %                             shift_size = ones(1,1+obj.nDim);
            %                             shift_size(d) = nRep;
            %                             shift = reshape(shift_value,shift_size);
            %
            %
            %                             obj.Data.sub{dd+1} = obj.Data.sub{dd+1} +  shift;
            %                         else % general case
            %                             X = obj.Data;
            %
            %                             % size of new array
            %                             new_size = size(X);
            %                             new_size(dd+1) = new_size(dd+1)*nRep; % each regressor is duplicated for each value along dimension d
            %                             if strcmpi(summing{d}, 'separate')
            %                                 new_size(1) = new_size(1)*nRep; % if seperate observation for each value along dimension d
            %                             end
            %
            %                             % preallocate memory for new data array
            %                             if ~issparse(X)
            %                                 obj.Data = zeros(new_size);
            %                             elseif length(new_size)<2
            %                                 obj.Data = spalloc(new_size(1),new_size(2), nnz(X));
            %                             else
            %                                 obj.Data = sparsearray('empty',new_size, nnz(X));
            %                             end
            %
            %                             % indices of array positions when filling up
            %                             % array
            %                             idx = cell(1,ndims(X));
            %                             for ee = 1:ndims(X)
            %                                 idx{ee} = 1:size(X,ee);
            %                             end
            %
            %                             for r=1:nRep % loop through all levels of splitting dimension
            %                                 idx{d+1} = r; % focus on r-th value along dimension d
            %                                 idx2 = idx;
            %                                 idx2{d+1} = 1;
            %                                 idx2{dd+1} = (r-1)*size(X,dd+1) + idx{dd+1};
            %                                 if strcmpi(summing{d}, 'separate')
            %                                     idx2{1} = (r-1)*size(X,1) + idx{1};
            %                                 end
            %                                 obj.Data(idx2{:}) = X(idx{:}); % copy content
            %                             end
            %                         end
            %
            %                         obj.Weights(dd).scale = interaction_levels(obj.Weights(dd).scale, obj.Weights(d).scale);
            %
            %                         % build covariance function as block diagonal
            %                         obj.Prior(dd).replicate = nRep * obj.Prior(dd).replicate;
            %                         %  obj.Prior(dd).CovFun = @(P)  replicate_covariance(obj.Prior(dd).CovFun, P, nRep);
            %
            %                         obj.Weights(dd).nWeight = obj.Weights(dd).nWeight*nRep; % one set of weights in dim dd for each level of splitting dimension
            %                     end
            %
            %                     % remove last dimension
            %                     idx = repmat({':'},1,ndims(obj.Data));
            %                     idx{d+1} = 2:size(obj.Data,d+1); % keep this dimension as singleton
            %                     Str = struct('type','()','subs',{idx});
            %                     obj.Data = subsasgn(obj.Data, Str, []);
            %
            %                     if isa(obj.Data, 'sparsearray') && ismatrix(obj.Data)
            %                         obj.Data = matrix(obj.Data); % convert to standard sparse matrix if it is two-D now
            %                     end
            %
            %                 end
            %
            %                 if ~isempty(SplitDims)
            %                     % reshape data array
            %                     NonSplitDims = setdiff(1:obj.nDim, SplitDims); % non-splitting dimensions
            %                     obj.Data = reshape(obj.Data, [size(obj.Data,1)  obj.Weights(NonSplitDims).nWeight]);
            %                 end
            %             end
            %
            %             % if separate: update number of observations (multiply by
            %             % number of levels along each splitting dimension)
            %             SeparateDims = strcmpi(summing, 'separate');
            %             obj.nObs = obj.nObs*prod([obj.Weights(SeparateDims).nWeight]);
            %
            %             % remove corresponding Weights and Prior and HP for splitting
            %             % dimensions, and update dimensionality
            %             obj.Weights(SplitDims) = [];
            %             obj.Prior(SplitDims) = [];
            %             obj.HP(SplitDims) = [];
            %             obj.nDim = obj.nDim - length(SplitDims);

            if ~isempty(constraint)
                assert(length(constraint)==nD, 'length of constraint C should match the number of dimensions in the regressor');
                assert(all(ismember(constraint, 'fbsm1n')), 'possible characters in constraint C are: ''fbsm1n''');
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
        function np = get.nParameters(obj)
            if length(obj)>1
                error('this method is only defined for single object');
            end
            ss = [obj.Weights.nWeight]; % number of weights for each component
            np = repelem(ss,obj.rank); % replicate by rank
        end

        %% GET NUMBER OF FREE PARAMETERS
        function nf = get.nFreeParameters(obj) % !! change to nFreeWeights?
            cc = [obj.Weights.constraint];
            ss = [obj.Weights.nWeight]; % number of total weights
            ss = ss - (cc~='f') - (ss-1).*(cc=='n' | ~cc); % reduce free weights for constraints

            % when using basis, use number of basis functions instead
            Bcell = {obj.Weights.basis};
            B = [Bcell{:}];
            withBasis = ~cellfun(@isempty, Bcell);
            if ~isempty(B)
                            projectedBasis = [B.projected];
                %ss(cellfun(@(x) ~isempty(x) && ~x.projected,Bcell)) = [B.nWeight];
                ss(withBasis) = [B.nWeight].*(1-projectedBasis) + [obj.Weights(withBasis).nWeight].*projectedBasis;
            end

            % number of free parameters per set weight (remove one if there is a constraint)
            nf = repmat(ss,obj.rank,1);
            %               nf = repmat(ss,obj.rank,1) - (cc~='f') - (ss-1).*(cc=='n' | ~cc);

        end

        %% GET NUMBER OF FREE DIMENSIONS
        function fd = nFreeDimensions(obj)
            fd = zeros(size(obj));
            for i=1:numel(obj)
                cc = [obj(i).Weights.constraint];
                fd(i) = max(sum(cc~='n',2));
            end
        end

        %% IS MULTIPLE RANK FREE DIMENSIONS
        function bool = isFreeMultipleRank(obj)
            bool = false(size(obj));
            for i=1:numel(obj)
                bool(i) = obj(i).rank>1 && all([obj(i).Weights.constraint]=='f','all');
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
            label = {W.label};
        end

        %% FIND SET OF WEIGHT WITH GIVEN LABEL
        function I = find_weights(obj, label)
            % I = find_weights(R, label) finds the set of weight with given
            % label in regressors R. I is a 2-by-1 vector where the first
            % element signals the index of the regressor in the regressor array and the second element
            % signals the dimension for the corresponding label.
            %
            % I = find_weights(R, C) if C is a cell array of labels
            % provides a matrix of indices with two rows, one for the index
            % of the regressor and the second for the corresponding
            % dimension.
            % NaN indicates when no weight is found.
            %

            label = tocell(label);

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

            Wlbl = weight_labels(obj); % label for all set of weights

            % build index matrix corresponding to each label
            nL = length(label); % number of labels
            I = nan(2,nL);
            for i=1:nL
                j = find(strcmp(label{i}, Wlbl));
                if ~isempty(j)
                    if length(j)>1
                        warning('more than one set of weights with label''%s''', label{i});
                    end
                    I(:,i) = RD(:,j(1));
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
                assert(ismember(fild, {'PosteriorMean', 'PosteriorStd', 'T','p', 'scale'}),...
                    'incorrect metric');
            end
            if nargin==1 || isequal(dims,0) % all weights
                U = [obj.Weights];
                U = {U.(fild)};
                U = cellfun(@(x) x(:)', U,'unif',0);
                U = [U{:}];

            else
                assert(length(dims)==length(obj));

                % total number of regressors/weights
                nRegTot = 0;
                nR = zeros(1,length(obj));
                for m=1:length(obj)
                    nR(m)  = obj(m).Weights(dims(m)).nWeight;
                    nRegTot = nRegTot + nR(m);
                end

                %%select weights on specific dimensions
                U = zeros(1,nRegTot); % concatenante set of weights for this dimension for each component and each module

                ii = 0; % index for regressors in design matrix

                for m=1:length(obj)

                    d = dims(m); %which dimension we select
                    % project on all dimensions except the dimension to optimize

                    % for r=1:obj(m).rank
                    idx = ii + (1:nR(m)*obj(m).rank); % index of regressors in output vector
                    thisU = obj(m).Weights(d).(fild);
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
                lbl = tocell(lbl);
            end

            I = nan(2,length(lbl));
            for i=1:length(lbl)
                % index for this label in both objects
                ind1 = obj.find_weights(lbl{i});
                ind2 = obj2.find_weights(lbl{i});

                assert(~any(isnan([ind1;ind2])), 'No set of weight label has the corresponding label in at least one of the regressor objects');

                if bool(1) % set weights
                    % corresponding set of weights
                    W = obj(ind1(1)).Weights(ind1(2));
                    W2 = obj2(ind2(1)).Weights(ind2(2));

                    assert(W.nWeight==W2.nWeight, 'the number of weights do not match for at least one set of weights with same label');

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
                    nR = obj(m).Weights(d).nWeight; % number of regressors
                    switch fild
                        case 'PosteriorMean'
                            for r=1:obj(m).rank % assign new set of weight to each component
                                if obj(m).Weights(d).constraint(r)~='n' % unless fixed weights

                                    idx = ii + (nR*(r-1)+1 : nR*r); % index of regressors for this component
                                    obj(m).Weights(d).PosteriorMean(r,:) = U(idx); % weight for corresponding adim and component
                                    if obj(m).Weights(d).constraint(r)=='1' % enforce this
                                        obj(m).Weights(d).PosteriorMean(r,1)=1;
                                        warning('chech if I should change idx to idx+1');
                                    end
                                end
                            end
                        case {'U_allstarting','U_CV', 'U_bootstrap','ci_boostrap'}
                            idx = ii+(1:nR*obj(m).rank);
                            this_U = reshape(U(idx,:), [nR, obj(m).rank, size(U,2)]); % weights x rank x initial point
                            this_U = permute(this_U,[3 1 2]); % initial point x weights x rank
                            obj(m).Weights(d).(fild) = this_U;
                        otherwise
                            error('incorrect field');
                    end

                    ii = ii + obj(m).rank*nR;
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

                    for r=1:obj(m).rank % assign new set of weight to each component
                        if obj(m).Weights(d).constraint(r)~='n' % unless fixed weights
                            idx = ii + (1:nR(r,d)); % index of regressors for this component
                            FW{m}{r,d} = U(idx); % weight for corresponding adim and component
                        end
                        ii = ii + nR(r,d);

                    end

                end
            end
        end

        %% ORTHOGONALIZE WEIGHTS (FOR PWA)
        function obj = orthogonalize_weights(obj, d)
            if nargin<1
                constraint = [obj.Weights.constraint];
                FreeDims = find(all(constraint=='f',1));
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
                assert(length(HP)==sum(nHP{:}),'incorrect length for HP');
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
            nR = [W.nWeight]; % corresponding number of weights

            nRegTotal = sum(nR); % total number of regressors

            %% first process fixed set of weights
            ii=0;
            redo = 0;
            fixedw = false(1,nRegTotal);
            for m=1:nM
                if any(W(m).constraint == 'n') % if weight is fixed, remove entire projection
                    for r=find(W(m).constraint == 'n') % for each set fo fixed weight
                        idx  = ii + (nR(m)*(r-1)+1:nR(m)*r); % index of regressors
                        fixedw(idx) = true;
                    end
                    redo = 1;
                end
                ii = ii + obj(m).rank*nR(m);
            end

            if redo % compute the projection without fixed weights (if any)
                rho = rho - Phi(:,fixedw)*UU(fixedw)';
            end


            %% then weights with linear constraint
            Uc = cell(1,nM);

            ii=0;
            for m=1:nM % for each module
                U_const = zeros(1,obj(m).rank); % constant weight for each component
                for r=1:obj(m).rank
                    switch W(m).constraint(r)
                        case {'f','b','1','n'}
                            U_const(r) = 0;
                        case 'm'
                            U_const(r) = 1; % constant offset to maintain constraint
                        case 's' % sum to one
                            U_const(r) = 1/nR(m);
                    end

                    idx = ii + (1:nR(m)); % index of regressors

                    % removing portion of rho due to projections of weights perpendicular to constraints
                    rho = rho - U_const(r)*sum(Phi(:,idx),2);
                    if W(m).constraint(r) == '1' % if first weight constrained to one, treat as fixed offset
                        rho = rho - Phi(:,idx(1));
                    end
                    ii =idx(end);

                end
                Uc{m} = U_const;
            end
            Uc2 = [Uc{:}]; % concatenate over modules

            % fixed part of the weights
            Uconst = repelem(Uc2,repelem(nR',[obj.rank]))';

            % we treated fixed weights apart, let's add them here
            Uconst(fixedw) = Uconst(fixedw) + UU(fixedw);
        end

        %% DEFINE PRIOR FOR ARRAY DIMENSIONS
        function  obj = define_priors_across_dims(obj, all_dims, summing, prior, HP, scale, basis, single_tau, condthresh)
            % make sure these options are in cell format and have minimum
            % size

if isempty(all_dims)
    return;
end

            summing = tocell(summing, max(all_dims));
            HP = tocell(HP, max(all_dims));
             prior = tocell(prior, max(all_dims));
             single_tau(end+1:max(all_dims)) = true;

            for d= all_dims % for all specified dimensions
                if isempty(summing{d}) % by default: linear set of weights for each dimension
                    summing{d} = 'linear';
                end
                obj.Weights(d).type = summing{d};
                obj.Weights(d).scale = scale{d};
                obj.Weights(d).nWeight = size(scale{d},2);

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
                                obj.HP(d) = HPstruct_ard(nWeight, d, HP{d}, HPfit);
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
                if size(scale,2)>100 && ~any(strcmp(summing,{'split','separate'})) && size(scale,1)==1
                    % Fourier basis is the default if more than 100 levels
                    basis = 'fourier';
                else % otherwise no basis functions
                    basis = [];
                end
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

            elseif length(basis)>=7 && strcmpi(basis(1:7), 'fourier')

                if length(basis)>7 % 'fourierN'
                    nFreq = str2double(basis(8:end));
                    assert(~isnan(nFreq), 'incorrect basis label');
                    condthresh = nan;
                else % 'fourier': number of basis functions is calculated automatically
                    %  condthresh = 1e12; % threshold for removing low-variance regressors
                    nFreq = nan;
                end

                if any(strcmp(summing,{'split','separate'}))
                    error('spectral trick not coded for sum=''%s''', summing);
                end
                %if strcmpi(type,'periodic')
                %    tau = period/4; % initial time scale: period/4
                %else
                %    tau = scale(end)-scale(1); % initial time scale: span of values
                %end

                %obj.Prior(d).CovFun = @(HP,scale,Tcirc) covSquaredExp_Fourier(scale, exp(HP(1)), exp(HP(2)),params.Tcirc);
                obj.Prior(d).CovFun = @covSquaredExp_Fourier;

                B.fun = @fourier_basis;
                B.fixed = true; % although the actual number of regressors do change
                padding = ~strcmpi(type, 'periodic'); % virtual zero-padding if non-periodic transform

                B.params = struct('nFreq',nFreq,'condthresh',condthresh, 'padding', padding);
                if strcmpi(type,'periodic')
                    % !! for periodic shouldn't use SquaredExponential
                    % prior
                    B.params.Tcirc = period;
                end
                obj.Prior(d).type = 'SquaredExponential';


            elseif length(basis)>4 && strcmp(basis(1:4),'poly')

                %% POLYNOMIAL BASIS FUNCTIONS
                order = str2double(basis(5:end));

                obj.Prior(d).CovFun = @L2basis_covfun;

                B.nWeight = order;
                B.fun = @basis_poly;
                B.params = struct('order',order);
                obj.Prior(d).type = 'L2_polynomial';

            elseif length(basis)>=3 && strcmp(basis(1:3),'exp')

                %% EXPONENTIAL BASIS FUNCTIONS
                if length(basis)==3
                    nExp =1;
                else
                    nExp = str2double(basis(4:end));
                end

                obj.Prior(d).CovFun = @L2basis_covfun;

                B.nWeight = nExp;
                B.fun = @basis_exp;
                B.fixed = false;
                B.params = struct('order',nExp);
                obj.Prior(d).type = 'L2_exponential';

            elseif length(basis)>=12 && strcmp(basis(1:12),'raisedcosine')
                %% RAISED COSINE BASIS FUNCTIONS (Pillow et al., Nature 2008)
                if length(basis)==12
                    nCos =1;
                else
                    nCos = str2double(basis(13:end));
                end

                obj.Prior(d).CovFun = @L2basis_covfun;

                B.nWeight = nCos;
                B.fun = @basis_raisedcos;
                B.fixed = false;
                B.params = struct('nFunctions',nCos);
                obj.Prior(d).type = 'L2_raisedcosine';

            else
                error('incorrect basis type: %s', basis);

            end
            if ~isempty(basis)
                B.projected = false;
                obj.Weights(d).basis = B;
            end

            %% define hyperparameters
            HH = obj.HP(d);
            switch obj.Prior(d).type
                case 'SquaredExponential'
                    if strcmpi(type,'periodic')
                        tau = period/4; % initial time scale: period/4
                    else
                        dt =  mean(diff(scale,[],2),2)'; % data point resolution (mean difference between two points)
                        % tau = dt; % initial time scale: mean different between two points
                        span = max(scale,[],2)' - min(scale,[],2)'; % total span in each dimension
                        tau =  sqrt(dt.*span); % geometric mean between the two
                        if single_tau
                            tau = mean(tau);
                        end
                    end
                    nScale = length(tau);

                    HP = HPwithdefault(HP, [log(tau) 0]); % default values for log-scale and log-variance [tau 1 1];
                    HH.HP = HP;
                    if nScale>1
                        HH.label = num2cell("log \tau"+(1:nScale));
                    else
                        HH.label = {'log \tau'};
                    end
                    HH.label{end+1} = 'log \alpha';
                    HH.fit = true(1,nScale+1);

                    % upper and lower bounds on hyperparameters
                    if strcmp(basis, 'fourier')
                        HH.LB(1:nScale) = log(2*tau/length(scale)); % lower bound on log-scale: if using spatial trick, avoid scale smaller than resolution
                        HH.LB(nScale+1) = -max_log_var; % to avoid exp(HP) = 0
                        HH.UB = max_log_var*[1 1];  % to avoid exp(HP) = Inf

                    else
                        HH.UB(1:nScale) = log(101*tau); %log(5*tau); % if not using spatial trick, avoid too large scale that renders covariance matrix singular
                        HH.UB(nScale+1) = max_log_var; 1;
                        if ~isempty(binning)
                            HH.LB = [log(binning)-2 -max_log_var];
                        else
                            HH.LB = -max_log_var*[1 1];  % to avoid exp(HP) = Inf
                        end
                    end
                case 'L2_polynomial'
                    HH = HPstruct_L2(order+1, HP);

                case 'L2_exponential'
                    log_min_tau = log(min(diff(scale)));
                    log_max_tau = log(scale(end)-scale(1));
                    logtau = linspace(log_min_tau,log_max_tau,nExp); % default tau values: logarithmically spaced
                    HP = HPwithdefault(HP, [logtau 0]); % default values for log-scale and log-variance [tau 1 1];
                    HH.HP = HP;
                    HH.LB = [(log_min_tau-2)*ones(1,nExp)   -max_log_var];
                    HH.UB = [(log_max_tau+5)*ones(1,nExp)   max_log_var];

                    if nExp>1
                        HH.label = [num2cell("log \tau"+(1:nExp)), {'\log \alpha'}];
                    else
                        HH.label = {'log \tau'};
                    end
                    HH.fit = true(1,nExp+1);
                case 'L2_raisedcosine'

                    dt = min(diff(scale)); % time step
                    Ttot = scale(end)-scale(1); % total span
                    a = (2*nCos+5)*pi/4/log(2*Ttot/dt); % time power, this formula is to tile all time steps with
                    c = 0; % time shift
                    if scale(end)<0
                        c = mean(scale);
                    end
                    Phi_1 = pi+a*log(dt/2); % angle for first basis function
                    HH.HP = [a c Phi_1 0];
                    HH.LB = [a-2 -max(scale)  Phi_1-pi   -max_log_var];
                    HH.UB = [a+2 -min(scale)  Phi_1+pi    max_log_var];
                    HH.fit = true(1,4);
                    HH.label = {'power','timeshift', '\Phi_1','\log \alpha'};
            end

            obj.HP(d) =  HH ;
        end


        %% PROJECTION MATRIX from free set of parameters to complete set of
        % parameters
        function PP = ProjectionMatrix(obj)

            dd = obj.nDim;
            rr = obj.rank;
            ss = [obj.Weights.nWeight];
            cc = [obj.Weights.constraint];
            PP = cell(rr,dd);
            for d=1:dd % for each dimension
                for r=1:rr % for each component
                    switch cc(r,d)
                        case 'f' % free dimension
                            PP{r,d} = speye(ss(d)); % no projection for first set of weight
                        case {'b','m','s'} % sum to zero, average weight or sum set to one

                            PP{r,d} = zeros(ss(d)-1,ss(d)); % rows: free, col:complete
                            for i=1:ss(d)-1
                                PP{r,d}(i,:) = [ones(1,i) -i zeros(1,ss(d)-i-1)]/sqrt(i*(i+1)); % coordinates for i-th basis vector of free space
                            end

                            %                             %PP{r,d} = sparse(ss(d)-1,ss(d)); % rows: free, col:complete
                            %                             I = []; % rows: free parameters
                            %                             J = []; % columns: complete parameter set
                            %                             VV = []; % value
                            %                             for i=1:ss(d)-1
                            %                                 vv = [ones(1,i) -i]/sqrt(i*(i+1)); % coordinates for i-th basis vector of free space
                            %                                 I = [I i*ones(1,length(vv))];
                            %                                 J = [J 1:i+1];
                            %                                 VV = [VV vv];
                            %
                            %                             end
                            %                             PP{r,d} = sparse(I,J,VV,ss(d)-1,ss(d));
                        case '1' % first weight set to 1
                            PP{r,d} = [sparse(ss(d)-1,1) speye(ss(d)-1)]; % first col is just 0, then identity matrix
                        case 'n' % fixed set of weights (no free parameter)
                            PP{r,d} = sparse(0,ss(d));
                    end
                end
            end

        end

        %% COMPUTE DESIGN MATRIX
        function [Phi,nReg, dims] = design_matrix(obj,subset, dims, init_weight)
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
            nReg = zeros(1,nF);
            ii = 1;
            for m=1:nM
                for d=dims{m}
                    nReg(ii) = obj(m).rank*obj(m).Weights(d).nWeight; % add number of regressors for this module
                    ii = ii+1;
                end
            end

            %% initialize weights
            if init_weight
                for m=1:nM
                    % if empty weight, pre-allocate
                    for d=1:obj(m).nDim
                        if isempty(obj(m).Weights(d).PosteriorMean)
                            obj(m).Weights(d).PosteriorMean = zeros(obj(m).rank,obj(m).Weights(d).nWeight);
                        end
                    end

                    % initialize weight to default value
                    obj(m) = obj(m).initialize_weights();

                end
            end

            ii = 0; % index for regressors in design matrix

            % use sparse coding if any data array is sparse
            SpCode = any(cellfun(@issparse, {obj.Data}));
            if SpCode
                Phi = sparse(obj(1).nObs,sum(nReg));
            else
                Phi = zeros(obj(1).nObs,sum(nReg));
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
            ss = [obj.Weights.nWeight]; % size along each dimension in this module
            cc = [obj.Weights.constraint]; % constraint for this module
            rk = obj.rank;

            for r=1:rk % for each rank
                for d=1:nD % for each dimension
                    % if more than one rank and only defined for first, use
                    % same as first
                    if r>1 && ((size(obj.Prior,1)<r) || isempty(obj.Prior(r,d).PriorCovariance))
                        obj.Prior(r,d).PriorCovariance = obj.Prior(1,d).PriorCovariance;
                    end

                    if isempty(obj.Prior(r,d).PriorCovariance) % by default:
                        if d<=nD && cc(r,d) =='1' % no covariance for first weight (set to one), unit for the others
                            obj.Prior(r,d).PriorCovariance = diag([0 ones(1,ss(d)-1)]);
                        else % otherwise diagonal unit covariance
                            obj.Prior(r,d).PriorCovariance = speye(ss(d));
                        end
                    end
                    if ~isequal(size(obj.Prior(r,d).PriorCovariance),[ss(d) ss(d)])
                        error('Prior covariance for dimension %d should be a square matrix of size %d', d, ss(d));
                    end
                end
            end
        end


        %% PROJECT TO SPECTRAL SPACE
        function obj = project_to_basis(obj)

            for m=1:length(obj) % for each module

                for d=1:obj(m).nDim % each dimension
                    B = obj(m).Weights(d).basis;
                    if ~isempty(B) && ~B.projected % if we're using a set of basis functions

                        % hyperparameter values for this component
                        this_HP = obj(m).HP(d).HP; %fixed values
                        this_scale = obj(m).Weights(d).scale;

                        % compute projection matrix and levels in projected space
                        if isrow(this_scale)
                        [B.B,new_scale, B.params] = B.fun(this_scale, this_HP, B.params); % apply function (params is hyperparameter)
                        else
                            % more rows in scale means we fit different
                            % functions for each level of splitting
                            % variable
                            [id_list,~,split_id] = unique(this_scale(2:end,:)','rows'); % get id for each observation
                            B.B = zeros(0,size(this_scale,2)); % the matrix will be block-diagonal
                            new_scale = zeros(size(this_scale,1),0);
                            for g=1:length(id_list)
                                subset = split_id==g; % subset of weights for this level of splitting variable
                                [this_B,this_new_scale,B.params] = B.fun(this_scale(1,subset), this_HP, B.params);
                                n_new = size(this_B,1);
                                B.B(end+1 : end+n_new, subset) = this_B; % 
                                new_scale(1,end+1:end+n_new) = this_new_scale;
                                new_scale(2,end-n_new+1:end) = repmat(id_list(g,:)',1,n_new);
                            end

                        end

                        assert( size(B.B,2)==obj(m).Weights(d).nWeight, 'Incorrect number of columns (%d) for projection on basis in component %d (expected %d)',...
                            size(B.B,2), d, obj(m).Weights(d).nWeight);
                        new_nWeight = size(B.B,1);

                        % store values for full space in B
                        B.scale = obj(m).Weights(d).scale;
                        B.nWeight = obj(m).Weights(d).nWeight;
                        obj(m).Weights(d).nWeight = new_nWeight;
                        obj(m).Weights(d).scale = new_scale;

                        %% if weights are constrained to have mean or average one, change projection matrix so that sum of weights constrained to 1
                        cc = obj(m).Weights(d).constraint;
                        for r=1:obj(m).rank
                            if any(cc(r)=='msb')
                                if ~all(cc == cc(r))
                                    error('spectral dimension %d cannot have ''%s'' constraint for one order and other constraint for other orders',d, cc(r,d));
                                end
                                B2 = sum(B.B,2); % ( 1*(B*U) = const -> (1*B)*U = const
                                if cc=='m' % rescale
                                    B2 = B2/obj(m).Weights(d).nWeight;
                                end
                                invB = diag(1./B2);
                                B.B = invB * B.B; % change projection matrix so that constraint is sum of weight equal to one

                                % !! I'm commenting because I don't
                                % understand what it's supposed to be
                                % doing... go back to it later on
                                %                                 for r2=1:obj(m).rank
                                %                                     if cc(r)~='b'
                                %                                         obj(m).Weights(d).constraint(r2) = 's';
                                %                                     end
                                %                                     obj(m).Prior(r2,d).CovFun = @(sc,hp) covfun_transfo(sc,hp,  diag(B.B) , obj(m).Prior(r2,d).CovFun);
                                %                                 end
                                break; % do not do it for other orders
                            end
                        end


                        %% if initial value of weights are provided, compute in new basis
                        U = obj(m).Weights(d).PosteriorMean;
                        if ~isempty(U)
                            B.PosteriorMean = U;
                            for r=1:obj(m).rank
                                % if some hyperparameters associated with other
                                % rank
                                % need to work this case!
                                % if ~isempty(idx{m}{r,d}) % if some hyperparameters associated with other rank
                                %     obj.covfun{r,d} =  obj.covfun{1,d};
                                % end
                                % if ~isempty( obj.U{r,d})
                                obj(m).Weights(d).PosteriorMean = U / B.B;
                                % end
                            end
                        end

                        % store copy of original data
                        if isempty(obj(m).DataOriginal)
                            obj(m).DataOriginal = obj(m).Data;
                        end

                        % apply change of basis to data (tensor product)
                        Bcell = cell(1,obj(m).nDim+1);
                        Bcell{d+1} = B.B;
                        obj(m).Data = tensorprod(obj(m).Data, Bcell);

                        B.projected = true;
                        obj(m).Weights(d).basis = B;
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
                    if ~isempty(B) && B.projected

                        tmp_nWeight = B.nWeight;
                        tmp_scale = B.scale;
                        B.nWeight = obj(m).Weights(d).nWeight;
                        obj(m).Weights(d).nWeight = tmp_nWeight;
                        obj(m).Weights(d).scale = tmp_scale;

                        W = obj(m).Weights(d); % weight structure

                         % save weights (mean, covariance, T-stat, p-value) in basis space in basis structure
                        B.PosteriorMean =  W.PosteriorMean;
                        B.PosteriorStd =  W.PosteriorStd;
                        B.PosteriorCov = W.PosteriorCov;
                        B.T = W.T;
                        B.p = W.p;

                        % compute posterior back in original domain
                        obj(m).Weights(d).PosteriorMean = W.PosteriorMean * B.B;

                        % compute posterior covariance, standard error
                        % of weights in original domain
                        W.PosteriorCov = zeros(W.nWeight, W.nWeight,obj(m).rank);
                        W.PosteriorStd = zeros(obj(m).rank, W.nWeight);
                        for r=1:obj(m).rank
                            PCov  = B.B' * B.PosteriorCov(:,:,r) * B.B;
                            W.PosteriorCov(:,:,r) = PCov;
                            W.PosteriorStd(r,:) = sqrt(diag(PCov))'; % standard deviation of posterior covariance in original domain
                        end

                        % compute T-statistic and p-value
                        W.T = W.PosteriorMean ./ W.PosteriorStd; % wald T value
                        W.p = 2*normcdf(-abs(W.T)); % two-tailed T-test w.r.t 0                       

                        % replace basis structure
                        B.projected = false;
                        W.basis = B;

                        obj(m).Weights(d) = W;

                        % recover original data
                        obj(m).Data = obj(m).DataOriginal;
                    end

                end
            end
        end

        %% COMPUTE BASIS FUNCTIONS
        function [Bmat,scale] = compute_basis_functions(obj,d)
            % [Bmat,scale] = compute_basis_functions(obj,d)
            % compute basis functions for regressor in dimension d

            if nargin<2
                Bcell =    {obj.Weights.basis};
                d = find(~cellfun(@isempty, Bcell),1);
                if isempty(d)
                    d = 1;
                end
            end
            B = obj.Weights(d).basis;
            if isempty(B)
                Bmat = [];
                scale = [];
            else
                this_HP = obj.HP(d).HP; %fixed values
                this_scale = obj.Weights(d).scale;
                if size(this_scale,1)>1 % if splitting variable is used
this_scale = unique(this_scale(1,:));
                end
                this_params = B.params;

                % compute projection matrix and levels in projected space
                [Bmat,scale] = B.fun(this_scale, this_HP, this_params);
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
                    Bcell = {obj(m).Weights.basis};
                    dims{m} = find(~cellfun(@isempty, Bcell));
                end
            end

            nSub = sum(cellfun(@length, dims)); % total number of plots
            iSub = 1; % subplot counter
            h = cell(1,nSub);

            for m=1:numel(obj)
                for d=dims{m}
                    subplot2(nSub,iSub);
                    [Bmat,scale_basis] = compute_basis_functions(obj(m),d);

                    h{iSub} =  plot(obj(m).Weights(d).scale, Bmat);
                    if isnumeric(scale_basis)
                        scale_basis = num2strcell(scale_basis);
                    end
                    box off;
                    legend(scale_basis);
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

            for m=1:length(obj)
                with_grad = (nargout>1); % compute gradient
                nD = obj(m).nDim; % number of dimensions for this module
                rk = obj(m).rank; % rank
                ss = [obj(m).Weights.nWeight]; % size of each dimension

                %% evaluate prior covariance
                P = obj(m).Prior; % prior structure

                gsf = cell(rk,nD);
                for d=1:nD
                    if size(P,1)==1 || isempty(P(2,d).CovFun)
                        % if more than rank 1 but same covariance prior for
                        % all
                        rrr = 1;
                    else
                        rrr = rk;
                    end
                    % r=1;
                    for r=1:rrr
                        if ~recompute && ~isempty(P(r,d).PriorCovariance)
                            % do nothing we already have the covariance
                            % prior
                            Sigma = P(r,d).PriorCovariance;
                        else
                            if isa(P(r,d).CovFun, 'function_handle') % function handle


                                % hyperparameter values for this component
                                this_HP = obj(m).HP(r,d).HP; %hyperparameter values
                                this_nHP = length(this_HP); % number of hyperparameters
                                this_scale = obj(m).Weights(d).scale;
                                nRep = P(r,d).replicate;
                                if  ~isempty(nRep) && nRep>1 % if we replicate the covariance matrix, we need to remove splitting dimensions are repetitions
                                    % we don't know which of the last
                                    % rows(s) in scale code for the
                                    % splitting variable, so let's find out
                                    nRep_check = 1;
                                    iRow = size(this_scale,1)+1; % let's start from no row at all

                                    while nRep_check<nRep && iRow>1
                                        iRow=iRow-1; % move one row up
                                       [id_list,~,split_id] = unique(this_scale(iRow:end,:)','rows'); % get id using last rows
                                        
                                        nRep_check = size(id_list,1); % number of unique sets
                                    end
                                    assert(nRep_check==nRep, 'weird error, could not find out splitting rows in scale');

                                    this_scale(iRow:end,:) =[]; % remove these rows for computing prior covariance
                                    this_scale = this_scale(:,split_id==1); % to avoid repetitions of value

                                end
                                this_basis = obj(m).Weights(d).basis;

                                % compute associated covariance matrix
                                CovFun = P(r,d).CovFun;
                                % [Sigma, gg]= CovFun(this_scale,this_HP, this_params);
                                if with_grad % need gradient
                                    [Sigma, gg]= CovFun(this_scale,this_HP, this_basis);

                                    % replicate covariance matrix if needed
                                    [Sigma, gg]= replicate_covariance(nRep, Sigma,gg);
                                    if isstruct(gg)
                                        gg = gg.grad;
                                    end
                                    if size(gg,3) ~= this_nHP
                                        error('For component %d and rank %d, size of covariance matrix gradient along dimension 3 (%d) does not match corresponding number of hyperparameters (%d)',...
                                            d,r, size(gg,3),this_nHP);
                                    end

                                    %% compute gradient now
                                    gsf{r,d} = zeros(size(gg)); % gradient of covariance w.r.t hyperparameters
                                    for l=1:this_nHP
                                        if obj(m).Weights(d).constraint(r)=='1' % if first weight fixed to 1, invert only the rest
                                            gsf{r,d}(:,:,l) = - blkdiag(0, (Sigma(2:end,2:end) \ gg(2:end,2:end,l)) / Sigma(2:end,2:end));% gradient of precision matrix
                                        else

                                            gsf{r,d}(:,:,l) = - (Sigma \ gg(:,:,l)) / Sigma;% gradient of precision matrix
                                        end
                                    end

                                    % select gradient only for fittable HPs
                                    HP_fittable = logical(obj(m).HP(r,d).fit);
                                    gsf{r,d} = gsf{r,d}(:,:,HP_fittable);
                                else
                                    Sigma= CovFun(this_scale,this_HP, this_basis);

                                    % replicate covariance matrix if needed
                                    Sigma= replicate_covariance(nRep, Sigma);
                                end
                            elseif isempty(P(r,d).CovFun) % default (fix covariance)
                                Sigma = [];
                                gsf{r,d} = zeros(ss(d),ss(d),0);
                            else % fixed custom covariance
                                Sigma = P(r,d).CovFun;
                                gsf{r,d} = zeros(ss(r,d),ss(d),0);
                            end
                        end
                        nW = obj(m).Weights(d).nWeight; % corresponding number of weights
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
                            % for d=1:DD
                            % this_nHP = length(obj(m).HP(d).HP);
                            gsf_new = zeros(ss(d)*rk,ss(d)*rk,this_nHP);
                            for l=1:this_nHP
                                gg = repmat({gsf{1,d}(:,:,l)},rk,1);
                                gsf_new(:,:,l) = blkdiag( gg{:});
                            end
                            gsf{1,d} = gsf_new;
                            % end
                        end
                    end

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
        function obj = compute_prior_mean(obj)
            % if more than one object, compute iteratively
            if length(obj)>1
                for i=1:numel(obj)
                    obj(i) = compute_prior_mean(obj(i));
                end
                return;
            end

            nD = obj.nDim; % dimension in this module
            ss = [obj.Weights.nWeight]; % size along each dimension in this module
            cc = [obj.Weights.constraint]; % constraint for this module
            rk = obj.rank;

            for d=1:nD % for each dimension
                for r=1:rk
                    if  ~isempty(obj.Prior(r,d).PriorMean)
                        mu = obj.Prior(r,d).PriorMean;
                        assert( size(mu,2) == ss(d), 'incorrect length for prior mean');
                        %if isvector(Mu) && rk>1
                        %    Mu = repmat(Mu,rk,1);
                        %end
                    else

                        switch cc(r,d) % what constraint on set of weights
                            case {'f','n','b'} % mean 0
                                mu = zeros(1,ss(d));
                            case '1' % first weight 1, then 0
                                mu = [1 zeros(1,ss(d)-1)];
                            case 's' % sum-one constraint
                                mu = ones(1,ss(d))/ss(d);
                            case 'm' % mean-one
                                mu = ones(1,ss(d));
                        end
                        obj.Prior(r,d).PriorMean = mu;
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

        %% CONSTRAINT CELL
        function C = constraint_cell(obj)
            % extract all constraints from different regressor objects into
            % a single cell array
            C = cell(size(obj));
            for i=1:numel(obj)
                for d=1:numel(obj(i).Weights)
                    C{i}(:,d) = obj(i).Weights(d).constraint;
                end
            end
        end

        %% INITIALIZE WEIGHTS
        function obj = initialize_weights(obj, obs)
            % if more than one object, compute iteratively
            if length(obj)>1
                for i=1:numel(obj)
                    obj(i) = initialize_weights(obj(i), obs);
                end
                return;
            end

            ss = [obj.Weights.nWeight]; % size along each dimension in this module
            cc = [obj.Weights.constraint]; % constraint for this module
            for r=1:obj.rank
                for d=1: obj.nDim
                    UU = obj.Weights(d).PosteriorMean;
                    if isempty(UU) || size(UU,1)<r
                        if first_update_dimension(obj)==d && cc(r,d)~='n' && ~strcmp(obs, 'normal')
                            % if first dimension to update, leave as nan to
                            % initialize by predictor and not by weights (for
                            % stability of IRLS)
                            UU = nan(1,ss(d));
                        else
                            switch cc(r,d)
                                case 'f' %free basis
                                    if any(cc(r,1:d-1)=='f')
                                        % sample from prior (multivariate
                                        % gaussian)
                                        P = obj.Prior(r,d);
                                        UU = mvnrnd(P.PriorMean,P.PriorCovariance);
                                    else % only the first free basis is set to 0 (otherwise just stays at 0)
                                        UU = zeros(1,ss(d));
                                    end
                                case 'b' % sum equal to zero
                                    UU = zeros(1,ss(d));
                                case 'm' % mean equal to one
                                    UU = ones(1,ss(d)); % all weight set to one
                                    % Uini{r,d} = rand(1,m(d)); %random intial weights
                                    % Uini{r,d} = Uini{r,d}/mean(Uini{r,d}); % enforce mean equal to one
                                case 's' % sun of weights equal to one
                                    UU = ones(1,ss(d))/ss(d); % all weights equal summing to one
                                    %Uini{r,d} = rand(1,m(d)); %random intial weights
                                    %Uini{r,d} = Uini{r,d}/sum(Uini{r,d}); % enforce mean equal to one
                                case '1' % first weight set to one
                                    UU = [1 zeros(1,ss(d)-1)];
                                case 'n' % fixed weights (if not provided by user, ones)
                                    % warning('Fixed set of weights for component %d and rank %r not provided, will use zeros',d,r);
                                    UU = ones(1,ss(d));
                            end
                        end
                        obj.Weights(d).PosteriorMean(r,:) = UU;
                    elseif size(UU,2) ~= ss(d)
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
                        ct = obj(i).Weights(d).constraint(r);
                        if ct ~= 'n'
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

                            switch ct
                                case 'b' %mean zero
                                    UU = UU - mean(UU);
                                case 'm' % mean equal to one
                                    UU = UU/mean(UU); % all weight set to one
                                case 's' % sun of weights equal to one
                                    UU = UU/sum(UU); % all weights equal summing to one
                                case '1' % first weight set to one
                                    UU(1) = 1;

                            end
                            obj(i).Weights(d).PosteriorMean(r,:) = UU;
                        else % fixed weights: set to 1
                            obj(i).Weights(d).PosteriorMean(r,:) = ones(1,obj(i).Weights(d).nWeight);
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
                        if isempty(W.PosteriorCov) && W.nWeights>0
                            error('Posterior has not been computed. Infer model first');
                        end
                        ct = W.constraint(r);
                        if ct ~= 'n' % except for fixed weights
                        
                            % sampled for mvn distribution with linear constraint
%                                 pp = PP{r,d};
%                                 Sigma = pp*P.PriorCovariance*pp';
%                                 if ~issymmetric(Sigma) % sometimes not symmetric for numerical reasons
%                                     Sigma = (Sigma+Sigma')/2;
%                                 end
%                                 UU =  P.PriorMean + mvnrnd(zeros(1,size(pp,1)), Sigma)*pp;
% 
%                             switch ct
%                                 case 'b' %mean zero
%                                     UU = UU - mean(UU);
%                                 case 'm' % mean equal to one
%                                     UU = UU/mean(UU); % all weight set to one
%                                 case 's' % sun of weights equal to one
%                                     UU = UU/sum(UU); % all weights equal summing to one
%                                 case '1' % first weight set to one
%                                     UU(1) = 1;
%
%                            end
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

            if nargin<3
                d=1;
            end

            assert(length(obj)==1);
            assert(isscalar(d) && d<=obj.nDim, 'd must be a scalar integer no larger than the dimensionality of M');
            nW = size(scale,2); % number of prediction data points

            obj = obj.project_from_basis;

            W = obj.Weights(d);
            assert(  any(strcmp(W.type, {'continuous','periodic'})), 'test data only for continuous or periodic regressors');
            P = obj.Prior(d);
            this_HP = obj.HP(d).HP;
            assert(size(scale,1)==size(W.scale,1), 'scale must has the same number of rows as the fitting scale data');

            B = W.basis;

            if isempty(B) % no basis functions: use equation 15
                % evaluate prior covariance between train and test set
                KK = P.CovFun({W.scale,scale},this_HP, []);

                mu = (W.PosteriorMean / P.PriorCovariance)*KK;

                if nargout>1
                    Kin = P.CovFun(scale,this_HP, []);
                    K = zeros(nW, nW,obj.rank);
                    S = zeros(obj.rank, nW);
                    for r=1:obj.rank
                        K(:,:,r) = Kin - KK'*W.invHinvK(:,:,r)*KK;
                        S(r,:) = sqrt(diag(K))';
                    end
                end

            else
                %% if using basis functions
                this_HP = obj.HP(d).HP; %fixed values

                % compute projection matrix
                BB = B.fun(scale, this_HP, B.params);

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
                    if obj.Weights(d2).constraint(r)~='n' % unless fixed weights
                        P = obj.Prior(r,d2); % corresponding prior
                        dif =  obj.Weights(d2).PosteriorMean(r,:) - P.PriorMean; % distance between MAP and prior mean

                        % ignore weights with infinite variance (their prior will always be null)
                        inf_var = isinf(diag(P.PriorCovariance));
                        if any(~inf_var)
                            dif = dif(~inf_var);
                            LP(r,d) = - dif / P.PriorCovariance(~inf_var,~inf_var) * dif'/2; % log-prior for this weight
                        end
                    end

                end
            end

        end

        %% first dimension to be updated in IRLS
        function d = first_update_dimension(obj)
            d = find(any([obj.Weights.constraint]=='f',1),1); % start update with first free dimension
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
            if any([obj.Weights.nWeight]==0) %
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
        function P = ProjectDimension(obj,r,d, do_squeeze, Uobs, to_double)
            %  P = projdim(obj.Data, obj.U(r,:), obj.sparse, d, varargin{:});  % add activation for component r

            if nargin<6 % by default, always convert output to double if sparse array
                to_double = true;
            end

            X = obj.Data;
            VV = cell(1,obj.nDim);
            for dd = setdiff(1:obj.nDim,d) % dimensions to be collapsed (over which tensor product must be computed)
                VV{dd} = obj.Weights(dd).PosteriorMean(r,:);
            end
            %   VV = obj.U(r,:);
            %   for dd=d % dimensions over which we project (no tensor product over this one)
            %       VV{dd} = [];
            %   end

            % spd = obj.sparse;
            if nargin<4 % by default, squeeze resulting matrix
                do_squeeze = 1;
            end
            if nargin<5 % by default, do not collapse over observations
                Uobs = [];
            end

            VV = [{Uobs} VV];

            P = tensorprod(X,VV);
            %  if isa(X, 'sparsearray')
            %      P = tensorprod(X,V);
            %  else
            %      P = tprod(X,V);
            %  end

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


        %% CLEAR DATA (E.G. TO FREE UP MEMORY)
        function obj = clear_data(obj)
            for i=1:numel(obj)
                obj(i).Data = [];
            end
        end


        %% EXTRACT REGRESSOR FOR A SUBSET OF OBSERVATIONS
        function obj = extract_observations(obj,subset) %,dim)
            % M = M.extract_observations(subset); extracts observations for regressor
            % object

            for i=1:numel(obj)
                S = size(obj(i).Data);
                %  obj.Data = reshape(obj.Data, S(1), prod(S(2:end))); % reshape as matrix to make subindex easier to call
                obj(i).Data = obj(i).Data(subset,:);
                n_Obs = size(obj(i).Data,1); % update number of observations
                %    if issparse(obj.Data) && length(S)>2
                %        obj.Data = sparsearray(obj.Data);
                %    end
                obj(i).Data = reshape(obj(i).Data, [n_Obs, S(2:end)]); % back to nd array
                obj(i).nObs = n_Obs;
            end
        end

        %% PRODUCT OF REGRESSORS
        function obj = times(obj1,obj2, varargin)
            %M = regressor_multiplier( M1, M2,... )
            %   multiplies regressors for GUM
            %
            % M = regressor_multiplier( M1, M2, 'SharedDim') if M1 and M2
            % share all dimensions. Weights, priors and hyperparameters for
            % new object are inherited from M1.
            %
            % M = regressor_multiplier( M1, M2, 'SharedDim', D)
            %to specify dimensions shared by M1 and M2 (D is a vector of integers)
            %
            % See also gum, regressor

            SharedDim = [];
            if not(isempty(varargin))
                if isequal(varargin{1}, 'ShareDim')
                    if length(varargin)==1
                        assert(obj1.nDim==obj2.nDim, 'number of dimensions of both regressors should be equal');
                        SharedDim = 1:obj1.nDim;
                    elseif length(varargin)==2
                        SharedDim = varargin{2};
                    end
                    SharedDimOk = isnumeric(SharedDim) && isvector(SharedDim) && all(isinteger(SharedDim))...
                        && all(SharedDim>0) && all(SharedDim<obj1.nDim) && all(SharedDim<obj2.nDim);
                    assert(SharedDimOk, 'SharedDim must be a vector of dimensions');
                    assert(all([obj1.Weights(SharedDim).nWeight]== [obj2.Weights(SharedDim).nWeight]),...
                        'the number of weights in both regressors are different along shared dimensions');
                    assert(length(varargin)<3, 'too many inputs');
                else % M1 * M2 * M3...
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
            OldDims = find(NotSharedDim)+1;
            NewDims = obj1.nDim+1+(1:sum(NotSharedDim));
            P(OldDims) = NewDims; % send non-shared dimensions at the end
            P(NewDims) = OldDims; % and use singletons
            %  P = [1 obj2.nDim+1+(1:obj1.nDim) 2:obj2.nDim+1];
            obj2.Data = permute(obj2.Data, P);

            % pairwise multiplication
            if isa(obj2.Data, 'double') && ~isdouble(obj1.Data, 'double')
                % make sure we don't run into problems if multiplying
                % integers with doubles
                obj1.Data = double(obj1.Data);
            end
            obj.Data = obj1.Data .* obj2.Data;

            %% other properties (if present in at least one regressor)
            obj.Weights = [obj1.Weights obj2.Weights(NotSharedDim)];
            obj.Prior = [obj1.Prior obj2.Prior(NotSharedDim)];
            obj.HP = [obj1.HP obj2.HP(NotSharedDim)];
            %             if ~isempty(obj2.spectral)
            %                 spk = obj2.spectral(1); % create empty structure
            %                 spk.fun = [];
            %                 obj2.spectral = [repmat(spk,1, obj1.nDim) obj2.spectral];
            %                 if isempty(obj1.spectral)
            %                     obj.spectral = obj2.spectral;
            %                 else
            %                     obj.spectral = [obj1.spectral obj2.spectral];
            %                 end
            %             else
            %                 obj.spectral = obj1.spectral;
            %             end
            %             obj.spectral = [obj1.spectral obj2.spectral];
            % obj.label = [obj1.label obj2.label];
            obj.formula = [obj1.formula ' * ' obj2.formula];

            FreeRegressors = isFreeWeights(obj);
            if sum(FreeRegressors)>1
                % fprintf('multiplying free regressors ... now only first will be free\n');

                % turning free regressors in second object to one-mean regressor
                for d=find(FreeRegressors(obj1.nDim+1:end))
                    obj.Weights(obj1.nDim+d).constraint = 'm';
                end
            end
        end

        %% WHETHER WEIGHTS ARE FREE (NOT CONSTRAINED)
        function bool = isFreeWeights(obj)
            assert(numel(obj)==1, 'only for scalar regressor object');
            bool = [obj.Weights.constraint]=='f';
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
                free_continuous1 = strcmp({W1.type}, 'continuous') & strcmp({W1.constraint}, 'f');
                free_continuous2 = strcmp({W2.type}, 'continuous') & strcmp({W2.constraint}, 'f');

                if    any(free_continuous1) && any(free_continuous2)
                    %  change second constraint to zero-sum to avoid identifiability issues with the likelihood.
                    chg = i2(free_continuous2);
                    for i=chg
                        obj2(i).Weights.constraint = 'b';
                    end
                end
            end

            % concatenate
            obj = [obj1 obj2];
        end

        %% INTERACTIONS (FOR CATEGORICAL REGRESSORS)--> x1:x2
        function obj = colon(obj1, obj2)
            if ~isa(obj2,'gum')
                obj2 = regressor(obj2, 'categorical');
            end

            assert(obj2.nObs==obj1.nObs, 'number of observations do not match');
            assert(obj1.nDim ==1 && obj2.nDim ==1 && strcmp(obj1.weights.type,'categorical') && strcmp(obj2.weights.type,'categorical')...
                , 'regressors must be one-dimensional categorical');

            obj = obj1;

            nLevel1 = obj1.weights.nWeight; % number of levels for each regressor
            nLevel2 = obj2.weights.nWeight;
            nLevel = nLevel1 * nLevel2; % number of interaction inf_terms

            % build design matrix
            X = zeros(obj1.nObs, nLevel);
            for i=1:nLevel1
                idx = (i-1)*nLevel2 + (1:nLevel1);
                X(:,idx) = obj1.Data(:,i) .* obj2.Data; % multiply covariates
            end
            obj1.Data = X;

            % scale
            scale = interaction_levels(obj1.weights.scale, obj2.weights.scale);

            % constraint (zeros for all default)
            warning('constraint for interactions not coded yet');

            obj.weights.nWeight  = empty_weight_structure(1, [obj1.nObs nLevel], ...
                scale, obj1.weights.color);

        end

        %% SPLIT/CONDITION ON REGRESSORS
        function obj = split(obj,X, scale, label)
            % obj = split(obj,X, scale, label)

            %  if nargin<3
            %      subindex_coding = false;
            %  end

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
            unq(isnan(unq)) = []; % remove nan values
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

                    obj.Data.sub{dd+1} = obj.Data.sub{dd+1} +  shift;
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
                obj.Weights(dd).scale = interaction_levels(obj.Weights(dd).scale, label);

                %% build covariance function as block diagonal
                if ~isempty(obj.Prior(dd).CovFun)
                    obj.Prior(dd).replicate = nVal * obj.Prior(dd).replicate;
                    %  obj.Prior(dd).CovFun = @(P)  replicate_covariance(obj.Prior(dd).CovFun, P, nVal);
                end

                obj.Weights(dd).nWeight = obj.Weights(dd).nWeight * nVal; % one set of weights for each level of splitting variable
            end


            %             % set this dimension as constant
            %             %  obj.nWeight(d) = 1;
            %             obj.HP(d) = HPstruct;
            %             obj.covfun{d} = {};
            %             obj.constraint(d) = 'n';

            %  % add label if required
            %  if nargin>=4
            %      if length(label) ~= length(scale)
            %          error('length of label does not match number of values');
            %      end
            %      obj.Weights(d).scale = label;
            %
            %            end
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

            if length(SplitDims)==1 && SplitDims==obj.nDim && obj.nDim==2 && strcmpi(summing{obj.nDim}, 'split') && ~(isa(obj.Data,'sparsearray') && subcoding(obj.Data,3))
                %% exactly the same as below but should be way faster for this special case

                % reshape regressors as matrices
                nWeightCombination = prod([obj.Weights.nWeight]);
                obj.Data = reshape(obj.Data,[obj.nObs nWeightCombination]);

                if isa(obj.Data, 'sparsearray') && ismatrix(obj.Data)
                    obj.Data = matrix(obj.Data); % convert to basic sparse matrix if it is 2d now
                end

                % how many times we need to replicate covariance function
                obj.Prior(1).replicate = obj.Weights(2).nWeight * obj.Prior(1).replicate;

                obj.Weights(1).nWeight = obj.Weights(1).nWeight*obj.Weights(2).nWeight; % one set of weights in dim 1 for each level of dim 2
                obj.Weights(1).scale = interaction_levels(obj.Weights(1).scale, obj.Weights(2).scale);

            else

                for d=SplitDims
                    nRep = obj.Weights(d).nWeight;

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

                        obj.Weights(dd).scale = interaction_levels(obj.Weights(dd).scale, obj.Weights(d).scale);

                        % build covariance function as block diagonal
                        obj.Prior(dd).replicate = nRep * obj.Prior(dd).replicate;
                        %  obj.Prior(dd).CovFun = @(P)  replicate_covariance(obj.Prior(dd).CovFun, P, nRep);

                        obj.Weights(dd).nWeight = obj.Weights(dd).nWeight*nRep; % one set of weights in dim dd for each level of splitting dimension
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
            obj.Weights(nDim_new).label = 'lag';
            if any(SplitValue) || isempty(SplitValue)
                obj.Weights(nDim_new).constraint = 'f';
            else % if no splitting, i.e multidimensional regressor, need to set some constraint
                obj.Weights(nDim_new).constraint = 'm';
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
            for i=1:n
                if obj(i).nDim<nD
                    obj(i) = add_dummy_dimension(obj(i), nD);
                end
            end

            % make sure that dimensionality matches along all other
            % dimensions
            S = ones(n,nD);
            for i=1:n
                S(i,1:obj(i).nDim) = [obj(i).Weights.nWeight];
            end
            otherdims = setdiff(1:nD,D);
            check_dims = all(S(:,otherdims)==S(1,otherdims),'all'); % check that dimensions allow for concatenation
            if check_dims_only % return only boolean telling whether regressors can be concatenated
                obj = check_dims;
                return;
            end

            assert(check_dims, 'number of elements in regressor does not match along non-concatenating dimensions');
            S(1,D) = sum(S(:,D)); % number of elements along dimension D are summed
            S = S(1,:); % select from first regressor

            % update properties of first element
            for d=1:nD
                obj(1).Weights(d).nWeight = S(d);
            end
            obj(1).nDim = nD;
            if nD>2 && ~isa(obj(1).val, 'sparsearray')
                obj(1).Data = sparsearray(obj(1).Data);
            end
            obj(1).Data = cat(D+1, obj.Data); % concatenate all regressors

            if D==nD && isempty(obj(1).Weights(D).constraint)
                obj(1).Weights(D).constraint = 'f';
            end

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
            F = cellfun(@(x) x(D).CovFun ,allPrior, 'unif',0);
            F = cellfun(@(x) x{D}, F, 'unif',0); % select D-th element for each regressor
            PP.CovFun = @(s,x) concatenate_covariance(S(:,D), F{:}, s, nHP, x);
            PP.label = 'mix';

            % deal with prior covariance
            F = cellfun(@(x) x(D).PriorCovariance ,allPrior, 'unif',0);
            if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                PP.PriorCovariance = {};
            else % otherwise build block diagonal matrix
                PP.PriorCovariance = blkdiag(F{:});
            end
            obj(1).Prior(D) =    PP;

            %% merge fields of weight structure
            obj(1).Weights = rmfield(obj(1).Weights, {'U_allstarting','U_CV'});

            fild = { 'PosteriorMean','PosteriorStd','T','p', 'scale'};
            WW = obj1.Weights(D);
            allWeights = {obj.Weights};
            for f = 1:length(fild)
                fn = fild{f};
                F = cellfun(@(x) x(D).(fn) , allWeights,'unif',0);
                %    F = {obj.(fn)};
                if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                    WW.(fn) = [];
                    %  else
                    %      F = cellfun(@(x) x{D}, F, 'unif',0);
                    %      if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                    %          obj(1).(fn){D} = {};
                else % otherwise concatenate
                    WW.(fn) = cat(2,F{:});
                end
            end

            % deal with scale
            F = cellfun(@(x) x(D).scale, allWeights,'unif',0);
            F = cellfun(@(x) x{D}, F, 'unif',0); % select D-th element for each regressor
            if any(cellfun(@isempty,F))
                idx = 0;
                iEmpty = find(cellfun(@isempty,F));
                for i=iEmpty
                    F{i} = idx + (1:obj(i).nWeight(D));
                    idx = idx + obj(i).nWeight(D);
                end
                WW.scale = cat(2,F{:});
            end

            % deal with posterior covariance
            F = cellfun(@(x) x(D).PosteriorCov , allWeights,'unif',0);
            if any(cellfun(@isempty,F)) % if any is empty, then overall is empty
                WW.PosteriorCov = {};
            else % otherwise build block diagonal matrix
                WW.PosteriorCov = blkdiag(F{:});
            end


            obj(1).Weights(D) = WW;

            %% deal with hyperparameters
            F = {obj.HP};
            F = cellfun(@(x) x(D), F); % select D-th element for each regressor
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
            obj.Weights(obj.nDim+1).constraint = 'n';
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

        %% EXPORT WEIGHTS TO TABLE
        function T = export_weights_to_table(obj)
            % T = M.export_weights_to_table(obj);
            % creates a table with metrics for all weights (scale, PosteriorMean, PosteriorStd, T-statistics, etc.)
            W = [obj.Weights]; % concatenate weights over regressors

            nWeight = [W.nWeight]; % number of weights for each set
            reg_idx = repelem(1:length(W), nWeight)'; % regressor index
            label = repelem({W.label},nWeight)'; % weight labels

            T = table(reg_idx, label, 'VariableNames',{'regressor','label'});


            filds = {'scale','PosteriorStd','T','p'};
            for f=1:length(filds)
                ff =  concatenate_weights(obj,0,filds{f}); % concatenate values over all regressors
                if length(ff) == height(T)
                    T.(filds{f}) = ff';
                end
            end

            % prior mea
            P = [obj.Prior];
            PM = {P.PriorMean};
            for i=1:length(W) % if using basis function, project prior mean on full space
                if ~isempty(W(i).basis)
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
            transform = {W.label};
            nHP = cellfun(@length, {H.HP}); % number of HPs for each transformatio
            transform = repelem(transform, nHP)';

            nW = cellfun(@length, {obj.Weights}); % number of weight set per regressor
            nW_cum = [0 cumsum(nW)];
            nHP_cum = [0 cumsum(nHP)];
            nHP_reg = nHP_cum(nW_cum(2:end)+1) - nHP_cum(nW_cum(1:end-1)+1); % total number of HPs per regressor
            reg_idx = repelem(1:length(obj), nHP_reg)';

            value = [H.HP]';
            label = [H.label]';
            fittable = [H.fit]';
            UpperBound = [H.UB]';
            LowerBound = [H.LB]';
            if isfield(H, 'std')
                standardDeviation = [H.std]';
                T = table(reg_idx, transform, label, value, fittable, LowerBound, UpperBound, standardDeviation);
            else
                T = table(reg_idx, transform, label, value, fittable, LowerBound, UpperBound);
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

%% CREATES A DESIGN MATRIX (OR ARRAY) WITH ONE-HOT ENCODING - ONE COLUMN PER VALUE OF X
function val = one_hot_encoding(X, scale,OneHotEncoding, nD)

nVal = size(scale,2);
%siz = size(X);
%if any(strcmp(summing,{'split','separate'})) % different function for each column
%    f_ind = repmat(0:prod(siz(2:end))-1,siz(1),1); % indicator function
%    f_ind = reshape(f_ind, siz);
%    nFun = prod(siz(2:end)); % total number of weights
%else

%end

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


%% L2-regression covariance function
%function [K, grad] = L2_covfun(nreg, loglambda,HPdefault, which_par)
function [K, grad] = L2_covfun(scale, loglambda, B)
if nargin<2
    K = 1; % number of hyperparameters
    return;
end

nReg = length(scale);
lambda2 = exp(2*loglambda);
if isinf(lambda2)
    K = diag(lambda2 * ones(1,nReg));
else
    K = lambda2 * eye(nReg); % diagonal covariance
end

grad.grad = 2*K; %eye(nreg);
grad.EM = @(m,Sigma) log(mean(m(:).^2 + diag(Sigma)))/2; % M-step of EM to optimize L2-parameter given posterior on weights
end

%% L2 covariance function when using basis (L2 on last hyperparameter)
function varargout = L2basis_covfun(scale, HP, B)
if nargin<2
    varargout = {1}; % number of hyperparameters
else
    varargout = cell(1,nargout);
    loglambda = HP(end);
    [varargout{:}] = L2_covfun(scale,loglambda, B);
    if nargout>1
        % place gradient over variance HP as last matrix in 3-D array
        nR = size(varargout{1},1);
        grad = varargout{2}.grad;
        varargout{2} = cat(3,zeros(nR,nR,length(HP)-1),grad);
    end
end
end

%% infinite covariance
function [K, grad] = infinite_cov(scale, HP, B)
if nargin<2
    K = 0; % number of hyperparameters
    return;
end

nReg = length(scale);
K = diag(inf(1,nReg));

grad = zeros(nReg,nReg,0);
end

%% Automatic Relevance Discrimination (ARD) covariance function
function [K, grad] = ard_covfun(scale, loglambda, B)
nReg = length(scale);

if nargin<2
    K = nReg; % number of hyperparameters
    return;
end

lambda2 = exp(2*loglambda); % variance for each regressor
K = diag(lambda2);

% Jacobian
G = zeros(nReg,nReg,nReg);
for i=1:nReg
    G(i,i,i) = 2*lambda2(i);
end
grad.grad = G;

% M-step of EM to optimize hyperparameters given posterior on weights
grad.EM = @(m,V) log(m(:).^2 + diag(V))/2;

end

%% Squared Exponential covariance
function [K, grad] = covSquaredExp(X, HP, B)
% radial basis function covariance matrix:
%  K = covSquaredExp(X, [log(tau), log(rho)])
% K(i,j) = rho^2*exp(- [(x(i,1)-x(j,1))^2 + ...
% (x(i,d)-x(j,d))^2]/(2*tau^2))
%
% tau can be scalar (same scale for each dimension) or a vector of length D
%
% [K, grad] = covSquaredExp(...) get gradient over
% hyperparameters, incl is a vector of two booleans indicating which
% hyperparameter to optimize

if nargin<2
    K = 2; % number of hyperparameters
    return;
end

within = ~iscell(X); % distance within set of points
if  within
    X = {X,X};
end
X{1} = X{1}';
X{2} = X{2}';

tau = exp(HP(1:end-1));
rho2 = exp(2*HP(end));

n_tau = length(tau);
if n_tau ~=1 && n_tau ~= size(X{1},2)
    error('tau should be scalar or a vector whose length matches the number of column in X');
end
tau = tau(:)';
if length(tau)==1
    tau = repmat(tau,1,size(X{1},2));
end

m = size(X{1},1); % number of data points
n = size(X{2},1); % number of data points

Xnorm = X{1}./tau; % normalize by scale
Ynorm = X{2}./tau;

% exclude scale-0 dimensions
nulltau = tau==0;
xx = Xnorm(:,~nulltau);
yy = Ynorm(:,~nulltau);

D = zeros(size(xx,1), size(yy,1));  % cartesian distance matrix between each column vectors of X
for i=1:size(xx,1)
    for j=1:size(yy,1)
        D(i,j) = sqrt(sum((xx(i,:)-yy(j,:)).^2));
        %  D(j,i) = D(i,j);
    end
end
if all(nulltau)
    D = zeros(m,n);
end

% treat separately for null scale
for tt=1:find(nulltau)
    Dinf = inf(m,n); % distance is infinite for all pairs
    Dinf( dist(X(:,tt)')==0) = 0; % unless values coincide
    D = D + Dinf;
end

K = rho2* exp(-D.^2/2);
if within
    K = force_definite_positive(K);
end

% compute gradient
if nargout>1
    grad = zeros(m,n,n_tau+1); % pre-allocate
    if n_tau == 1
        grad(:,:,1) = rho2* D.^2 .* exp(-D.^2/2); % derivative w.r.t GP scale
    else
        for t=1:n_tau % gradient w.r.t scale for each dimension
            dd = bsxfun(@minus, X{1}(:,t),X{2}(:,t)').^2; % distance along dimension
            grad(:,:,t) = log(rho2) * dd .* exp(-D.^2/2)/tau(t)^2; % derivative w.r.t GP scale
        end
    end
    grad(:,:,n_tau+1) = 2*K; % derivative w.r.t GP log-variance (log-rho)
    % grad(:,:,n_tau+1) = exp(-D.^2/2); % derivative w.r.t GP variance (rho)
    % grad(:,:,n_tau+2) = eye(n); % derivative w.r.t. innovation noise
    % grad = grad(:,:,which_par);
end
end

%% SQUARED EXPONENTIAL KERNEL COVARIANCE IN FOURIER DOMAIN
function [K, grad] = covSquaredExp_Fourier(scale, HP, B)
%
% [K, grad] = covSquaredExp_Fourier(HP, scale, params)
%
% Squared Exponential RBF covariance matrix, in the Fourier domain:
%  K = RBF(X, tau, rho, sigma)
% K(i,j) = rho*exp(- [(x(i,1)-x(j,1))^2 + ...
% (x(i,d)-x(j,d))^2]/(2*tau^2)) + sigma*delta(i,j)
%
% tau can be scalar (same scale for each dimension) or a fecor or length D
%omit sigma if there is no innovation noise
%
% [K, grad] = RBF_Fourier(X, tau, rho, sigma, incl) get gradient over
% hyperparameters, incl is a vector of three booleans indicating which
% hyperparameter to optimize


tau = exp(HP(1:end-1));
rho = exp(HP(end));

Tcirc = B.params.Tcirc;

% covariance matrix is diagonal in fourier domain
% !! check the formula !!
eexp = exp(-(2*pi^2/Tcirc^2)*tau^2*scale.^2);
kfdiag = sqrt(2*pi)*rho*tau*eexp;
K = diag(kfdiag);

% n_tau = length(tau);
% if n_tau ~=1 && n_tau ~= size(X,2)
%     error('tau should be scalar or a vector whose length matches the number of column in X');
% end
% tau = tau(:);
n = size(scale,2); % number of data points

% compute gradient
if nargout>1
    grad = zeros(n,n,2); % pre-allocate
    % if n_tau == 1
    grad_scale = sqrt(2*pi)*rho*  (1 - 4*pi^2/Tcirc^2*tau^2*scale.^2) .* eexp; % derivative w.r.t GP scale (tau)
    grad(:,:,1) = diag(grad_scale);
    % else
    %     for t=1:n_tau % gradient w.r.t scale for each dimension
    %             grad(:,:,t) = rho * dist(X(:,t)').^2 .* exp(-D.^2/2)/tau(t)^3; % derivative w.r.t GP scale
    %     end
    % end
    grad(:,:,2) = K/rho; % derivative w.r.t GP weight (rho)
    %    grad = grad(:,:,incl);
end

end



%% Periodic covariance function
function [K, grad] = cov_periodic(x, HP, period) %, nrep)
if nargin<2
    K = 2; % number of hyperparameters
    return;
end
%HPfull = [period 1]; % default values
%HPfull(which_par) = HP; % active hyperparameters

ell = exp(HP(1));
sf = exp(2*HP(2));

% use function from GPML
%[cov, dK] = covPeriodic([log(ell) log(period) log(sf)], x);

%adapted from GPML (see 4.31 from Rasmussen & Williams)
within = ~iscell(x);
if within
    x = {x,x};
end
T = pi/period*bsxfun(@plus,x{1},-x{2}');
S2 = (sin(T)/ell).^2;

K = sf*exp( -2*S2 );
K = force_definite_positive(K);

% if separate functions for each column, full covariance matrix is block
% diagonal
% if nrep>1
%     K = repmat({K}, 1, nrep);
%     K = blkdiag(K{:});
% end

if nargout>1
    % turn dK into gradient tensor (weight x weight x HP)
    %gradK = gradient(dK, size(cov,1), 3);

    %  P = sin(2*T).*cov;

    grad(:,:,1) = 4*S2.*K; % / ell; % grad w.r.t ell ( covPeriodic provides grad w.r.t log(ell))
    %  gradK2(:,:,2) = 2/ell^2*P.*T; % grad w.r.t sf ( covPeriodic provides grad w.r.t log(sf))
    grad(:,:,2) = 2*K; % / sf;

    % dhyp = [4*(S2(:)'*Q(:)); 2/ell^2*(P(:)'*T(:)); 2*sum(Q(:))];

    %grad(:,:,1) = gradK(:,:,1) / ell;
    %grad(:,:,2) = gradK(:,:,3) / sf;

    %grad = grad(:,:,which_par); % select hyperparameters to be fitted

    %     if nrep>1
    %         GG = grad;
    %         for hp=1:size(GG,3)
    %             gc = repmat({grad(:,:,hp)}, 1, nrep);
    %             GG(:,:,hp) = blkdiag(gc{:});
    %         end
    %         grad = GG;
    %
    %     end
end

end


%% CONCATENATE COVARIANCE MATRIX (AND GRADIENT)
function [cov, grad] = concatenate_covariance(S, varargin)
% concatenate_covariance(S, CovFun1, CovFun2, ... CovFunn, scale, nHP,HP)


n = length(varargin)-3;
P = varargin{end}-1;
nHP = varargin{end}-2;
scale = varargin{end-3};
C = cell(n,nargout);

Scum = cumsum([0 S]); % to get indices of levels for each prior function

cnt_hp = 0;
for i=1:n
    % nHP = varargin{i}; % number of hps for this covariance matrix
    this_level = scale(Scum(i)+1:Scum(i+1));  % cirresponding levels
    this_HP = P(cnt_hp+(1:nHP(i))); % corresponding HPs
    [C{i,:}] = varargin{i}(this_level, this_HP);
    cnt_hp = cnt_hp + nHP(i);
end

% full covariance matrix is block
% diagonal
cov = blkdiag(C{:,1});

% compute gradient
if nargout>1
    siz = zeros(3,n);
    for i=1:n
        if isstruct( C{i,2})
            C{i,2} = C{i,2}.grad;
        end
        for dd=1:3
            siz(dd,i) = size(C{i,2}, dd);
        end
    end
    siz_tot = sum(siz,2); % sum over regressors

    grad = zeros(siz_tot);

    cnt_hp = 0; % hyperparameter count
    cnt_w = 0; % parameter count

    for i=1:n
        for hp=1:nHP(i)
            idx = cnt_w + (1:siz(1,i));
            grad(idx,idx,cnt_hp+hp) = C{i,2}(:,:,hp);
        end
        cnt_hp = cnt_hp + nHP(i);
        cnt_w = cnt_w + siz(1,i);


    end
end
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

%% compute basis functions as polynomial
function [B, scale, params] = basis_poly(X,HP, params)
order = params.order;
B = zeros(order, length(X));
X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)

for p=1:order+1
    B(p,:) = X.^(p-1);
end
scale = 0:order;
end

%% compute basis functions as polynomial
function [B, scale, params, gradB] = basis_exp(X,HP, params)
nExp = length(HP)-1;
B = zeros(nExp,length(X));
X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)
for p=1:nExp
    tau = exp(HP(p));
    B(p,:) = exp(-X/tau);
end
scale = 1:nExp;

if nargout>3
    % gradient of matrix w.r.t hyperparameters
    gradB = zeros(nExp, length(X),length(HP));
    for p=1:nExp
        tau = exp(HP(p));
        gradB(p,:,p) = B(p,:) .* X/tau;
    end

end
end

%% compute basis functions as raised cosine
function [B, scale, params, gradB] = basis_raisedcos(X,HP, params)
nCos = params.nFunctions;
a = HP(1);
c = HP(2);
Phi_1 = HP(3);
X = X(1,:); % x is on first row (extra rows may be used e.g. if splitted)

B = zeros(nCos,length(X));
Phi = Phi_1 + pi/2*(0:nCos-1); % Pi/2 spacing between each function
for p=1:nCos
    alog = a*log(X+c)-Phi(p);
    nz = (X>-c) & (alog>-pi) & (alog<pi); % time domain with non-null value
    B(p,nz) = cos(alog(nz))/2 + 1/2;
end
scale = 1:nCos;

if nargout>3
    gradB = zeros(nCos, length(X),length(HP));
    for p=1:nCos
        alog = a*log(X+c)-Phi(p);
        nz = (alog>-pi) & (alog<pi); % time domain with non-null value
        sin_alog = sin(alog(nz)) / 2;
        gradB(p,nz,1) = - sin_alog .* log(X(nz)+c); % gradient w.r.t a
        gradB(p,nz,2) = - sin_alog ./ (X(nz)+c) * a ; % gradient w.r.t c
        gradB(p,nz,3) = sin_alog; % gradient w.r.t Phi_1
    end
end
end


% %% turn dK function from GPML into matrix gradient
% function gradK = gradient(dK, n, p)
% gradK = zeros(n,n,p);
% for i=1:n
%     for j=1:n
%         Q = zeros(n);
%         Q(i,j) = 1;
%         gradK(i,j,:) = dK(Q);
%     end
% end
% end

% %% selection HP parameters
% function V = selectHP(V, d, which_par)
% V.HP{d} = V.HP{d}(which_par);
% if isfield(V, 'HP_LB')
%     V.HP_LB{d} = V.HP_LB{d}(which_par);
% end
% if isfield(V, 'HP_UB')
%     V.HP_UB{d} = V.HP_UB{d}(which_par);
% end
% V.HP_labels{d} = V.HP_labels{d}(which_par);
% end

%% CREATE EMPTY WEIGHT STRUCTURE
function W = empty_weight_structure(nD, DataSize, scale, color)
W = struct('type','','label', '', 'nWeight',0, 'nFreeWeight',0,...
    'PosteriorMean',[], 'PosteriorStd',[],'V',[],...
    'PosteriorCov', [], 'T',[],'p',[],...
    'scale',[],'constraint','f','plot',[],'basis',[],'invHinvK',[]);
% other possible fields: U_CV, U_allstarting

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
    'PriorMean',[], 'replicate',1, 'spectral',[]);

% make it size nD
P = repmat(P, 1, nD);

end

%% combine default and specified HP values
function   HP = HPwithdefault(HPspecified, HPdefault)
HP = HPdefault;
HP(~isnan(HPspecified)) = HPspecified(~isnan(HPspecified)); % use specified values
end

%% void HP structure
function S = HPstruct()
S.HP = []; % value of hyperparameter
S.label = {}; % labels
S.fit = []; % which ones are fittable
S.LB = []; % lower bound
S.UB = []; % upper bound
end

%% HP structure for L2 regularization
function S = HPstruct_L2(d, HP, HPfit)

% value of hyperparameter
if nargin>1
    S.HP = HPwithdefault(HP, 0); % default value if not specified
else
    S.HP = 0;
end
S.label = {['log \lambda' num2str(d)]};  % HP labels
if nargin>2
    S.fit = HPfit; % if HP is fittable
else
    S.fit = true;
end
S.LB = -max_log_var; % lower bound: to avoid exp(HP) = 0
S.UB = max_log_var;  % upper bound: to avoid exp(HP) = Inf
end

%% HP structure for ARD regularization
function S = HPstruct_ard(nReg,d, HP, HPfit)

% value of hyperparameter
if nargin>1
    S.HP = HPwithdefault(HP, 0); % default value if not specified
else
    S.HP = zeros(1,nReg);
end
for i=1:nReg
    S.label = {['log \lambda' num2str(d) '_' num2str(i)]};  % HP labels
end
if nargin>2
    S.fit = HPfit; % if HP is fittable
else
    S.fit = true(1,nReg);
end
S.LB = -max_log_var*ones(1,nReg); % lower bound: to avoid exp(HP) = 0
S.UB = max_log_var*ones(1,nReg);  % upper bound: to avoid exp(HP) = Inf
end


%% concatenate hyperparameter structure
function S = cat_hyperparameters(Sall)
S=struct;
S.HP = cat(2, Sall.HP);
S.label = cat(2, Sall.label);
S.fit = cat(2, Sall.fit);
S.LB = cat(2, Sall.LB);
S.UB = cat(2, Sall.UB);

end

%% maximum log variance (to avoid non-invertible covariance matrices)
function mlv = max_log_var()  %
mlv = 12;
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

%% new covariance prior function from change of coordinates
function [K, grad] = covfun_transfo(scale, HP, M,covfun, B)
[K,grad] = covfun(scale,HP, B);
K = M*K*M';
for i=1:size(grad,3)
    grad(:,:,i) = M*grad(:,:,i)*M';
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

