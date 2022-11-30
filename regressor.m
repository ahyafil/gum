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
    % V = regressor(..., 'binning', B) bins values of X with size B (for
    % continuous and periodic variables only) to reduce size of covariance
    % matrix. Use regressor(..., 'binning', 'auto') to ensure fast approximate
    % inference.
    %
    %
    % V = regressor(..., 'HPfit',  which_par) is a boolean vector defining which
    % hyperparametes are fitted:
    % - for 'linear' regressor: one hyperparameter (L2-regularization) per dimension
    % - for 'categorical', one hyperparameter (L2-regularization)
    % - for 'continuous', three parameters of the gaussian kernel (log(scale), log(variance),
    % additive noise)
    % - for 'periodic' , two hyperparameters (log(scale)), log(standard deviation of
    % variance)). See covPeriodic from GPML.
    % V = regressor(...,'period',p) to define
    % period (default: 2pi)
    %
    % V = regressor(...,'hyperparameter') is a vector defining value of hyperparameter
    % (initial value if parameter is fitted, fixed value otherwise). See field
    % 'HPfit' for description of each hyperparameter. Use NaN to use default
    % values.
    %
    % V = regressor(..., 'SubindexCoding',bool) will use SubindexCoding regressor if bool is
    % set to true (false by default)
    %
    % V = regressor(...,'spectral', bool) uses spectral trick if bool is set to
    % true [default: false]
    %
    % V = regressor(...,'scale',sc) for categorical regressor to specify which
    % values to code (others have zero weight)
    %
    % V = regressor(...,'condthresh', c) to define conditional threshold for low rank spectral trick (default:1e12)
    %
    % V = regressor(...,'sum', S)
    % If X is a matrix, you may define how f(x) for each columns are summed
    % into factor. Possible values for S are:
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
    %
    % V = regressor(...,'label', N) to add label to regressor
    
    % By default, first dimension is free and all others are
    %
    %
    %
    %
    %See also gum, covFunctions, regressor_plus, regressor_multiplier
    
    % add details of output field, and input variance and tau
    
    properties
        val
        label = {}
        formula = ''
        nDim
        nObs
        % sparse = 0
        nRegressor
        scale = {}
        spectral
        HP = HPstruct
        covfun = {[]}
        constraint = ''
        mu
        sigma
        U
        V
        se
        T
        p
        rank = 1
        ordercomponent = false
        U_CV
        U_allstarting
        % Bfft
        % U_Fourier
        % se_Fourier
        plot = {}
    end
    properties (Dependent)
        nParameters
        nFreeParameters
    end
    
    methods
        %% DISPLAY
        % function disp(obj)
        %     if obj.nDim ==1
        %         str = sprintf('one-dimensional module with %d regressors and %d observations', obj.size, obj.nObs);
        %     else
        %         str = sprintf('%d-dimensional module with %d observations', obj.nDim, obj.nObs);
        %     end
        %     disp(str);
        %
        % end
        
        
        %% CONSTRUCTOR %%%%%
        function obj = regressor(X,type, varargin)
            
            if nargin==0
                return;
            end
            
            % default values
            % condthresh = 1e12; % threshold for spectral trick
            SubindexCoding = [];
            scale = [];
            do_spectral =0;
            period = 2*pi;
            HPfit = [];
            HP = [];
            summing = 'weighted';
            constraint = '';
            binning = [];
            label = '';
            variance = [];
            tau = [];
            color = [];
            
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
                    case 'hpfit'
                        HPfit = varargin{v+1};
                    case 'hyperparameter'
                        HP = varargin{v+1};
                    case 'subindexcoding'
                        SubindexCoding  = varargin{v+1};
                    case 'spectral'
                        do_spectral  = varargin{v+1};
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
                        obj.plot = varargin{v+1};
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
            else % data type
                switch type
                    case 'constant'
                        HPfit_def = [];
                    case 'linear'
                        HPfit_def = ones(1,ndims(X)-1);
                    case 'categorical'
                        HPfit_def = 1;
                    case 'continuous'
                        HPfit_def = [1 1];
                    case 'periodic'
                        HPfit_def = [1 1];
                    otherwise
                        error('incorrect type');
                end
            end
            if isempty(HPfit)
                HPfit = HPfit_def;
            elseif length(HPfit)==1
                HPfit = HPfit*ones(size(HPfit_def));
            end
            
            HPfit = logical(HPfit);
            
            %if ~isstruct(V)
            %    X = X;
            %else
            %    assert(isfield(V','vect'), 'if V is a structure, it should have a field ''vect''');
            %    xx = V.vect;
            %    obj = rmfield(V, 'vect');
            %end
            
            if isrow(X)
                X = X';
            end
            
            % dimension for corresponding regressor
            if iscolumn(X), nD = 1;
            elseif strcmpi(type, 'linear')
                nD = ndims(X)-1;
            else
                nD = ndims(X);
            end
            siz = size(X);
            obj.nObs = siz(1);
            obj.nRegressor = siz(2:end);
            obj.scale = cell(1,nD);
            if ~isempty(scale)
                
                if iscell(scale)
                    obj.scale = scale;
                    if length(scale)<nD
                        obj.scale(length(scale)+1:nD) = {[]};
                    end
                else
                    obj.scale{nD} = scale;
                end
            end
            
            if ~isempty(binning)
                if ischar(binning) && strcmp(binning,'auto')
                    binning = (max(X(:))- min(X(:)))/100; % ensore more or less 100 points spanning range of input values
                end
                X = binning*round(X/binning); % change value of X to closest bin
            end
            
            if isempty(SubindexCoding) % by default non SubindexCoding, except if categorical
                SubindexCoding = false(1,max(nD-1,1));
                if any(strcmp(type,{'categorical','continuous','periodic'}))
                    SubindexCoding(nD) = true;
                end
            elseif isscalar(SubindexCoding)
                SubindexCoding = [false(1,nD-1) SubindexCoding];
            end
            % obj.sparse = SubindexCoding;
            
            if ~iscell(summing)
                summing = {summing};
            end
            SplitOrSeparate = strcmpi(summing, 'split') | strcmpi(summing, 'separate');
            SplitOrSeparate(end+1:nD) = false;
            
            %% build structure
            switch lower(type)
                %% constant regressor (no fitted weight)
                case 'constant'
                    obj.val = X;
                    obj.nDim = 1;
                    obj.constraint = 'n';
                    obj.U = {}; %{ones(1,obj.nRegressor)};
                    
                    %% linear regressor
                case 'linear'
                    
                    obj.val = X;
                    obj.val(isnan(obj.val)) = 0; % convert nans to 0: missing regressor with no influence on model
                    % if any(strcmp(summing,{'split','separate'}))
                    %     nWeights = nWeights * prod(siz(3:end));
                    %     obj.val = reshape(X, siz(1), nWeights); % vector set replicated along dimension 3 and beyond
                    %     obj.nDim = 1;
                    % else
                    obj.nDim = nD;
                    %  nD = obj.nDim;
                    
                    % end
                    
                    HP = tocell(HP,nD);
                    if ~isempty(variance), HP{1} = log(variance)/2; end
                    
                    % hyperpameters for first dimension
                    obj.HP(nD) = HPstruct_L2(1, HP{1}, HPfit);
                    
                    nWeights = siz(1+nD);
                    obj.covfun{nD} = @(x) L2_covfun(nWeights, x); % L2-regularization (diagonal covariance prior)
                    obj.HP(nD).fit = HPfit;
                    
                    %                     %hyperparameters
                    %                     for d=2:obj.nDim
                    %                         switch summing
                    %                             case 'weighted'
                    %                                 obj.covfun{d} = @(x) L2_covfun(siz(d+1), x); % L2-regularization (diagonal covariance prior)
                    %                                 obj.HP(d) = HPstruct_L2(d);
                    %
                    %                             case {'equal','split'}
                    %                                 obj.U{d} = ones(1,siz(d+1));
                    %                                 obj.HP(d) = HPstruct;
                    %                         end
                    %                     end
                    
                    % prior for other dimensions
                  %  obj = define_priors_across_dims(obj, nD-1, summing, HP);
                                        obj = define_priors_across_dims(obj, nD, summing, HP);
                                        % !!! warning it seems should be
                                        % nD-1 in certain circ

                    
                    %% categorical regressor
                case 'categorical'
                    
                    if ~SubindexCoding(nD)
                        warning('categorical variables should be coded by subindex');
                        SubindexCoding(nD) = 1;
                    end
                    
                    % build design matrix
                    unq = unique(X); % unique values
                    unq(isnan(unq)) = []; % remove nan values
                    if ~isempty(obj.scale{nD})
                        if any(~ismember(obj.scale{nD},unq))
                            warning('exclude scale values not present in data');
                            obj.scale{nD}(~ismember(obj.scale{nD},unq)) = [];
                        end
                    else
                        obj.scale{nD} = unq';
                    end
                    
                    nVal = length(obj.scale{nD});
                    obj.val = indicator_regressors(X, obj.scale{nD}, SubindexCoding(nD),nD, summing);
                    obj.nDim = nD ;
                    obj.nRegressor(nD) = nVal;
                    
                    if ~iscell(HP)
                        HP = [cell(1,nD) {HP}];
                    end
                    
                    % hyperparameters (L2-regularization)
                    if ~isempty(variance), HP{nD} = log(variance)/2; end
                    HP_L2 = HPwithdefault(HP{nD}, 0);
                    obj.HP(nD) = HPstruct_L2(nD, HP_L2, HPfit);
                    obj.covfun{nD} = @(x) L2_covfun(nVal, x); % L2-regularization (diagonal covariance prior)
                    obj.HP(nD).fit = HPfit;
                    
                    
                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, nD-1, summing, HP);
                    
                    %% CONTINUOUS OR PERIODIC VARIABLE
                case {'continuous','periodic'}
                    
                    if strcmpi(type, 'periodic')
                        X = mod(X,period);
                    end
                    
                    
                    % build design matrix
                    unq = unique(X)'; % unique values
                    unq(isnan(unq)) = []; % remove nan values
                    nVal = length(unq);
                    obj.val = indicator_regressors(X, unq, SubindexCoding(nD),nD, summing);
                    obj.nDim = nD;
                    obj.nRegressor(nD) = nVal;
                    
                    if ~iscell(HP)
                        HP = [cell(1,nD-1) {HP}];
                    end
                    
                    if ~isempty(tau), HP{nD}(1:length(tau)) = log(tau); end
                    if ~isempty(variance), HP{nD}(2) = log(variance)/2; end
                    
                    % define continuous prior
                    obj = define_continuous_prior(obj,type, nD,unq, HP{nD}, do_spectral, binning,summing, period);
                    obj.HP(nD).fit = HPfit;
                    %obj.sparse(nD) = SubindexCoding;
                    
                    % prior for other dimensions
                    obj = define_priors_across_dims(obj, nD-1, summing, HP);
                    
                    %                     %% PERIODIC VARIABLE
                    %                 case 'periodic'
                    %
                    %                     if strcmpi(type, 'periodic')
                    %                         X = mod(X,period);
                    %                     end
                    %                     unq = unique(X); % unique values
                    %                     unq(isnan(unq)) = []; % remove nan values
                    %                     nVal = length(unq);
                    %
                    %                     obj.val = indicator_regressors(X, unq, SubindexCoding(nD),nD, summing);
                    %
                    %                     obj.scale{nD} = unq;
                    %
                    %
                    %
                    %                     if isempty(tau)
                    %                         tau = period/4; % initial time scale: period/4
                    %                     end
                    %                     if ~isempty(variance), HP(2) = log(variance)/2; end
                    %
                    %                     %  if any(strcmp(summing,{'split','separate'}))
                    %                     %      nrep = prod(siz(2:end)); % covariance matrix is block diagonal with repeated covariance matrix for each matrix independently
                    %                     %  else
                    %                     %      nrep = 1;
                    %                     %  end
                    %
                    %
                    %                     if do_spectral
                    %                         error('not coded yet');
                    %                     end
                    %
                    %                     obj.HP(nD) = HPstruct; % initialize HP structure
                    %
                    %                     % initial value for parameters
                    %
                    %                     HP = HPwithdefault(HP, [log(tau) 0]); % default values for log-scale and log-variance [tau 1 1];
                    %                     obj.HP(nD).HP = HP;
                    %
                    %                     % covariate function
                    %                     obj.covfun{nD} = @(x) cov_periodic(unq, x, period, nrep); % session covariance prior
                    %                     obj.HP(nD).label = {'\log\tau','\log\alpha'};
                    %
                    %                     if do_spectral
                    %                         % V.HP_LB{newd}(1) = 2*tau/length(xx); % if using spatial trick, avoid scale smaller than resolution
                    %                     else
                    %                         obj.HP(nD).UB(1) = log(5*tau); % if not using spatial trick, avoid too large scale that renders covariance matrix singular
                    %                         obj.HP(nD).UB(2) = 1;
                    %                         if ~isempty(binning)
                    %                             obj.HP(nD).LB = [log(binning) 3+log(eps)];
                    %                         end
                    %                     end
                    %                     obj.HP(nD).fit = HPfit;
                    %
                    %                     % obj.sparse(nD) = SubindexCoding;
                    %                     obj.nDim = nD;
                    %                     obj.nRegressor(nD+1) = nVal;
                    %
                    %                     % prior for other dimensions
                    %                     obj = define_priors_across_dims(obj, nD-1, summing, HP);
                    
                otherwise
                    error('unknown type:%s', type);
            end
            
            % by default, first dimension (that does not split or separate) is free, other have sum 1
            if ~strcmpi(type, 'constant')
                FreeDim = find(~SplitOrSeparate,1);
                obj.constraint = repmat('s',1,obj.nDim);
                obj.constraint(FreeDim) = 'f';
            end
            
            % whether components should be reordered according to variance, default: reorder if all components have same constraints
            if obj.rank>1
                obj.ordercomponent = all(all(constraint==constraint(1,:)));
            end
            
            %% if splitting along one dimension (or separate into different observations)
            SplitDims = fliplr(find(SplitOrSeparate));
            
            if length(SplitDims)==1 && SplitDims==obj.nDim && obj.nDim==2 && strcmpi(summing{obj.nDim}, 'split') && ~(isa(obj.val,'sparsearray') && subcoding(obj.val,3))
                %% exactly the same as below but should be way faster for this special case
                
                % reshape regressors as matrices
                obj.val = reshape(obj.val,[obj.nObs prod(obj.nRegressor)]);
                
                if isa(obj.val, 'sparsearray') && ismatrix(obj.val)
                    obj.val = matrix(obj.val); % convert to basic sparse matrix if it is 2d now
                end
                
                nRep = obj.nRegressor(2);
                
                % build covariance function as block diagonal
                obj.covfun{1} = @(P)  replicate_covariance(obj.covfun{1}, P, nRep);
                
                obj.nRegressor(1) = obj.nRegressor(1)*nRep;
            else
                
                for d=SplitDims
                    nRep = obj.nRegressor(d);
                    %  obj.val = tocell(obj.val);
                    
                    % replicate design matrix along other dimensions
                    for dd = setdiff(1:obj.nDim,d)
                        
                        if isa(obj.val,'sparsearray') && subcoding(obj.val,dd+1) && strcmpi(summing{d}, 'split') %.sparse(dd)
                            shift_value = obj.val.siz(dd+1) * (0:nRep-1); % shift in each dimension (make sure we use a different set of indices for each value along dimension d
                            shift_size = ones(1,1+obj.nDim);
                            shift_size(d) = nRep;
                            shift = reshape(shift_value,shift_size);
                            
                            
                            obj.val.sub{dd+1} = obj.val.sub{dd+1} +  shift;
                        else
                            X = obj.val;
                            new_size = size(X);
                            new_size(dd+1) = new_size(dd+1)*nRep; % each regressor is duplicated for each value along dimension d
                            if strcmpi(summing{d}, 'separate')
                                new_size(1) = new_size(1)*nRep; % if seperate observation for each value along dimension d
                            end
                            
                            % preallocate
                            if ~issparse(X)
                                obj.val = zeros(new_size);
                            elseif length(new_size)<2
                                obj.val = spalloc(new_size(1),new_size(2), nnz(X));
                            else
                                obj.val = sparsearray('empty',new_size, nnz(X));
                            end
                            
                            
                            idx = cell(1,ndims(X));
                            for ee = 1:ndims(X)
                                idx{ee} = 1:size(X,ee);
                            end
                            
                            for r=1:nRep
                                idx{d+1} = r; % focus on r-th value along dimension d
                                idx2 = idx;
                                idx2{d+1} = 1;
                                idx2{dd+1} = (r-1)*size(X,dd+1) + idx{dd+1};
                                if strcmpi(summing{d}, 'separate')
                                    idx2{1} = (r-1)*size(X,1) + idx{1};
                                end
                                obj.val(idx2{:}) = X(idx{:}); % copy content
                            end
                        end
                        
                        
                        % build covariance function as block diagonal
                        obj.covfun{dd} = @(P)  replicate_covariance(obj.covfun{dd}, P, nRep);
                        
                        obj.nRegressor(dd) = obj.nRegressor(dd)*nRep;
                    end
                    
                    % remove dimension D
                    %                     idx = cell(1,ndims(obj.val));
                    %                         for ee = 1:ndims(obj.val)
                    %                             idx{ee} = 1:size(obj.val,ee);
                    %                         end
                    %                         idx{d+1}(1) = []; % keep only first dimension
                    idx = repmat({':'},1,ndims(obj.val));
                    idx{d+1} = 2:size(obj.val,d+1); % keep this dimension as singleton
                    Str = struct('type','()','subs',{idx});
                    obj.val = subsasgn(obj.val, Str, []);
                    
                    
                    
                    if isa(obj.val, 'sparsearray') && ismatrix(obj.val)
                        obj.val = matrix(obj.val); % convert to basic sparse matrix if it is 2d now
                    end
                    %  obj.val(idx{:}) = [];
                    
                    %                 % set this dimension as constant
                    %                 %  obj.size(d) = 1;
                    %                 obj.HP(d) = HPstruct;
                    %                 obj.covfun{d} = {};
                    %                 obj.constraint(d) = 'n';
                    
                    
                    
                end
                
                NonSplitDims = setdiff(1:obj.nDim, SplitDims);
                if ~isempty(SplitDims)
                    obj.val = reshape(obj.val, [size(obj.val,1)  obj.nRegressor(NonSplitDims)]);
                end
            end
            
            SeparateDims = strcmpi(summing, 'separate');
            obj.nObs = obj.nObs*prod(obj.nRegressor(SeparateDims));
            
            obj.scale(SplitDims) = [];
            obj.nRegressor(SplitDims) = [];
            obj.constraint(SplitDims) = [];
            obj.covfun(SplitDims) = [];
            obj.HP(SplitDims) = [];
            obj.nDim = obj.nDim - length(SplitDims);
            
            
            if ~isempty(constraint)
                assert(length(constraint)==nD, 'length of constraint C should match the number of dimensions in the regressor');
                assert(all(ismember(constraint, 'fbsm1n')), 'possible characters in constraint C are: ''fbsm1n''');
                obj.constraint = constraint;
            end
            
            
            
            %             % other dimensions correspond to different observations
            %             if strcmp(summing,'separate')
            %                 if SubindexCoding % convert
            %                     obj.val{1} = obj.val{1}(:);
            %                     obj.val{nD+1} = obj.val{nD+1}(:);
            %                 else
            %                     sz = size(obj.val);
            %                     obj.val = reshape(obj.val, prod(sz(1:end-1)), sz(end)); % convert to classical 2D design matrix
            %                 end
            %             end
            
            % regressor label
            if ~isempty(label)
                obj.label = label;
            end
            
            % define plotting color
            if ~isempty(color)
                obj.plot = repmat({{'color',color}}, 1, obj.nDim);
            end
            
        end
        
        %% OTHER METHODS %%%%
        
        %% set rank
        function obj = set.rank(obj, rank)
            if ~isnumeric(rank) || ~isscalar(rank) || rank<=0
                error('rank should be a positive integer');
            end
            obj.rank = rank;
        end
        
        
        %% get number of parameters
        function np = get.nParameters(obj)
            ss = obj.nRegressor;
            np = repelem(ss,obj.rank); % number of parameters for each component
            
        end
        
        %% get number of free parameters
        function nf = get.nFreeParameters(obj)
            cc = obj.constraint;
            ss = obj.nRegressor;
            % number of free parameters per set weight (remove one if there is a constraint)
            nf = repmat(ss,obj.rank,1) - (cc~='f') - (ss-1).*(cc=='n' | ~cc);
            % nf = nf(:)'; % into line vector
        end
        
        %% GET NUMBER OF FREE DIMENSIONS
        function fd = nFreeDimensions(obj)
            cc = obj.constraint;
            fd = max(sum(cc~='n',2));
        end
        
        %% IS MULTIPLE RANK FREE DIMENSIONS
        function bool = isFreeMultipleRank(obj)
            bool = obj.rank>1 && all(obj.constraint=='f','all')
        end
        
        %% TOTAL NUMBER OF PARAMETERS
        function nptot = nParametersTot(obj)
            nptot = sum(obj.nRegressor)*obj.rank;
        end
        
        %% CONCATENATE ALL WEIGHTS
        function U = concatenate_weights(obj, dims)
            if nargin==1 || isequal(dims,0) % all weights
                U = {obj.U};
                U = cellfun(@(x) [x{:}], U,'unif',0);
                U = [U{:}];
                
            else
                assert(length(dims)==length(obj));
                
                % total number of regressors/weights
                nRegTot = 0;
                nR = zeros(1,length(obj));
                for m=1:length(obj)
                    nR(m)  = obj(m).nRegressor(dims(m));
                    nRegTot = nRegTot + nR(m);
                end
                
                %%select weights on specific dimensions
                U = zeros(1,nRegTot); % concatenante set of weights for this dimension for each component and each module
                
                ii = 0; % index for regressors in design matrix
                
                for m=1:length(obj)
                    
                    d = dims(m); %which dimension we select
                    % project on all dimensions except the dimension to optimize
                    
                    for r=1:obj(m).rank
                        idx = ii + (1:nR(m)); % index of regressors
                        U(idx) = obj(m).U{r,d};
                        
                        
                        ii = idx(end); %
                    end
                end
            end
        end
        
        %% SET WEIGHTS FOR SET OF MODULES
        function obj = set_weights(obj,U, dims)
            
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
                    nR = obj(m).nRegressor(d); % number of regressors
                    
                    for r=1:obj(m).rank % assign new set of weight to each component
                        if obj(m).constraint(r,d)~='n' % unless fixed weights
                            
                            idx = ii + (nR*(r-1)+1 : nR*r); % index of regressors for this component
                            obj(m).U{r,d} = U(idx); % weight for corresponding adim and component
                            if obj(m).constraint(r,d)=='1'
                                obj(m).U{r,d}(1)=1;
                            end
                        end
                    end
                    
                    ii = ii + obj(m).rank*nR;
                end
            end
        end
        
        %% SET FREE WEIGHTS (or anything of same size - used only for gradient i think)
        function FW = set_free_weights(obj,U, FW, dims)
            
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
                        if obj(m).constraint(r,d)~='n' % unless fixed weights
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
                free_dims = find(all(obj.constraint=='f',1));
                [~,i_d] = min(obj.nRegressor(free_dims));
                d = free_dims(i_d);
            end
            
            other_dims = setdiff(1:obj.nDim,d);
            
            UU = cat(1,obj.U{:,d});
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
                    obj.U{r,dd} = xxx;
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
            
            if nargin && any(bool)
                if isscalar(bool)
                    obj.HP = repmat(obj.HP, rank,1);
                    obj.covfun = repmat(obj.covfun, rank, 1);
                else
                    for d=find(bool)
                        obj.HP(:,d) = repmat(obj.HP(1,d), rank,1);
                        obj.covfun(:,d) = repmat(obj.covfun(1,d), rank, 1);
                    end
                    
                end
                
            end
            
            
        end
        
        %% REMOVE CONSTRAINED PART FROM PREDICTOR
        function  [rho, Uconst] = remove_constrained_from_predictor(obj, dims, rho, Phi, UU)
            
            nM = length(obj);
            
            % size of regressor per module
            nR = zeros(1,nM);
            for m=1:nM % for each module
                nR(m) = obj(m).nRegressor(dims(m));
            end
            nRegTotal = sum(nR);
            
            %% first process fixed set of weights
            ii=0;
            redo = 0;
            fixedw = false(1,nRegTotal);
            for m=1:nM
                d = dims(m); % dimension
                
                if any(obj(m).constraint(:,d) == 'n') % if weight is fixed, remove entire projection
                    for r=find(obj(m).constraint(:,dims(m)) == 'n') % for each set fo fixed weight
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
                    switch obj(m).constraint(r,dims(m))
                        case {'f','b','1','n'}
                            U_const(r) = 0;
                        case 'm'
                            U_const(r) = 1; % constant offset to maintain constraint
                        case 's' % sum to one
                            U_const(r) = 1/nR(m);
                    end
                    
                    idx = ii + (1:nR(m)); % index of regressors
                    
                    rho = rho - U_const(r)*sum(Phi(:,idx),2); % removing portion of A due to projections of weights perpendicular to constraints
                    if obj(m).constraint(r,dims(m)) == '1' % if first weight constrained to one, treat as fixed offset
                        rho = rho - Phi(:,idx(1));
                    end
                    ii =idx(end);
                    
                end
                Uc{m} = U_const;
            end
            Uc2 = [Uc{:}]; % concatenate over modules
            Uconst = repelem(Uc2,repelem(nR',[obj.rank]))'; % fixed part of the weights
        end
        
        %% DEFINE PRIOR FOR ARRAY DIMENSIONS
        function  obj = define_priors_across_dims(obj, dim_max, summing, HP)
            summing = tocell(summing);
            HP = tocell(HP, obj.nDim-1);
            
            for d=1:dim_max
                if length(summing)<d
                    summing{d} = 'linear';
                end
                switch summing{d}
                    case {'weighted','linear'}
                        obj.covfun{d} = @(x) L2_covfun(obj.nRegressor(d), x); % L2-regularization (diagonal covariance prior)
                        obj.HP(d) = HPstruct_L2(d);
                        obj.HP(d).HP = HPwithdefault(HP{d}, 0); % log-variance hyperparameter
                        
                        
                    case 'equal'
                        obj.U{d} = ones(1,obj.nRegressor(d));
                        obj.HP(d) = HPstruct;
                        
                    case {'continuous','periodic'}
                        if isempty(obj.scale{d})
                            scl = 1:obj.nRegressor(d);
                        else
                            scl = obj.scale{d};
                        end
                        
                        % define continuous prior
                        obj = define_continuous_prior(obj,summing{d}, d,scl,HP{d}, 0, [],[],2*pi);
                end
                
            end
        end
        
        %% DEFINE CONTINUOUS PRIOR
        function obj = define_continuous_prior(obj, type, d, scale, HP, do_spectral, binning, summing, period)
            
            
            if nargin<4
                HP = [];
            end
            obj.HP(d) = HPstruct; % initialize HP structure
            obj.scale{d} = scale; % values
            
            if nargin<6 || do_spectral
                if any(strcmp(summing,{'split','separate'}))
                    error('spectral trick not coded for sum=''%s''', summing);
                end
                if strcmpi(type,'periodic')
                    tau = period/4; % initial time scale: period/4
                else
                    tau = scale(end)-scale(1); % initial time scale: span of values
                end
                
                error('add HP HPfit');
                %covariance function
                if strcmpi(type,'periodic')
                    error('not coded yet');
                else
                    obj.covfun{d} = @(x,wvec,Tcirc) RBF_Fourier(wvec, exp(x(1)), exp(x(2)),Tcirc); % neuron firing spectral covariance prior
                    
                    obj.spectral(d).fun = @(x,ll) basis_fft_cat(x, exp(ll(1)), 1, condthresh); % FFT transformation matrix
                end
            else
                
                if strcmpi(type,'periodic')
                    tau = period/4; % initial time scale: period/4
                else
                    tau = mean(diff(scale,[],2),2)'; % initial time scale: mean different between two points
                end
                
                % if any(strcmp(summing,{'split','separate'}))
                %     nrep = prod(obj.nRegressor(2:end)); % covariance matrix is block diagonal with repeated covariance matrix for each matrix independently
                % else
                nrep = 1;
                % end
                
                %  scale = scale(:);
                
                % covariate function
                if strcmpi(type,'periodic')
                    obj.covfun{d} = @(x) cov_periodic(scale, x, period); % session covariance prior
                else
                    obj.covfun{d} = @(x) covSquaredExp(scale, x,  nrep); % session covariance prior
                end
            end
            
            nScale = length(tau);
            
            HP = HPwithdefault(HP, [log(tau) 0]); % default values for log-scale and log-variance [tau 1 1];
            obj.HP(d).HP = HP;
            if nScale>1
                obj.HP(d).label = num2cell("log \tau"+(1:nScale));
            else
                obj.HP(d).label = {'log \tau'};
            end
            obj.HP(d).label{end+1} = 'log \alpha';
            
            if do_spectral
                obj.HP(d).LB(1:nScale) = log(2*tau/length(X)); % lower bound on log-scale: if using spatial trick, avoid scale smaller than resolution
                obj.HP(d).LB(nScale+1) = -max_log_var; % to avoid exp(HP) = 0
                obj.HP(d).UB = max_log_var*[1 1];  % to avoid exp(HP) = Inf
                
            else
                obj.HP(d).UB(1:nScale) = log(101*tau); %log(5*tau); % if not using spatial trick, avoid too large scale that renders covariance matrix singular
                obj.HP(d).UB(nScale+1) = max_log_var; 1;
                if ~isempty(binning)
                    obj.HP(d).LB = [log(binning)-2 -max_log_var];
                else
                    obj.HP(d).LB = -max_log_var*[1 1];  % to avoid exp(HP) = Inf
                end
            end
        end
        
        
        
        %% PROJECTION MATRIX from free set of parameters to complete set of
        % parameters
        function PP = ProjectionMatrix(obj)
            
            dd = obj.nDim;
            rr = obj.rank;
            ss = obj.nRegressor;
            cc = obj.constraint;
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
        function [Phi,nReg] = design_matrix(obj,subset, dims, init_weight)
            
            nM = length(obj);
            
            
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
            
            %  rank = ones(1,obj.nMod); % all rank one
            nReg = zeros(1,nF);
            
            ii = 1;
            for m=1:nM
                for d=dims{m}
                    nReg(ii) = obj(m).rank*obj(m).nRegressor(d); % add number of regressors for this module
                    ii = ii+1;
                end
            end
            
            %% initialize weights
            if init_weight
                for m=1:nM
                    
                    if isempty(obj(m).U)
                        for d=1:obj(m).nDim
                            obj(m).U{d} = zeros(obj(m).rank,obj(m).nRegressor(d));
                        end
                    end
                    
                    % initialize weight to default value
                    obj(m) = obj(m).initialize_weights();
                    
                end
            end
            
            ii = 0; % index for regressors in design matrix
            
            spcode = any(cellfun(@issparse, {obj.val}));
            if spcode
                Phi = sparse(obj(1).nObs,sum(nReg));
            else
                Phi = zeros(obj(1).nObs,sum(nReg));
            end
            for m=1:nM
                
                % project on all dimensions except the dimension to optimize
                for d=dims{m}
                    for r=1:obj(m).rank
                        idx = ii + (1:obj(m).nRegressor(d)); % index of regressors
                        Phi(:,idx) = ProjectDimension(obj(m),r,d); % tensor product, and squeeze into observation x covariate matrix
                        ii = idx(end); %
                    end
                end
            end
            
            
        end
        
        
        %% CHECK PRIOR COVARIANCE and provide default covariance if needed
        function obj = check_prior_covariance(obj)
            dd = obj.nDim; % dimension in this module
            ss = obj.nRegressor; % size along each dimension in this module
            cc = obj.constraint; % constraint for this module
            rr = obj.rank;
            
            
            if ~isempty(obj.sigma)
                sig = obj.sigma;
            else
                sig = [];
            end
            
            if isempty(sig)
                sig = cell(rr,dd);
            elseif ~iscell(sig)
                sig = {sig};
            end
            
            if isvector(sig) && rr>1
                sig = repmat(sig(:)',rr,1);
            end
            
            for r=1:rr % for each rank
                for d=1:dd % for each dimension
                    if isempty(sig{r,d}) % by default:
                        if d<=dd && cc(r,d) =='1' % no covariance for first weight (set to one), unit for the others
                            sig{r,d} = diag([0 ones(1,ss(d)-1)]);
                        else % otherwise diagonal unit covariance
                            sig{r,d} = speye(ss(d));
                        end
                    end
                    if length(sig{r,d}) == 1
                        assert(sig{r,d}>0,'ridge parameter should be positive');
                        sig{r,d} = speye(ss(d)) * sig{r,d};
                    elseif (length(sig{r,d}(:)) == ss(d))
                        assert(all(sig{r,d}>0),'ridge vector should contain only positive values');
                        sig{r,d} = spdiags(sig{r,d}(:), 0, ss(d), ss(d));
                        
                    else
                        if ~isequal(size(sig{r,d}),[ss(d) ss(d)])
                            error('covariance for dimension %d should be scalar, vector of length %d or square matrixof size %d', d, ss(d), ss(d));
                        end
                        
                    end
                    
                end
            end
            obj.sigma = sig;
            
        end
        
        
        %% PROJECT TO SPECTRAL SPACE
        function obj = project_to_spectral(obj) % (M,P,idx, rank)
            
            S = obj.spectral;
            if isempty(S)
                return;
            end
            
            %  DD = obj.nDim; % number of dimensions for this module
            rr = obj.rank; % rank
            ss = obj.nRegressor; % size of each dimension
            cc = obj.constraint;
            
            %obj.spectral = tocell(obj.spectral);
            with_spectral = ~cellfun(@isempty,  {obj.spectral.fun});
            SpectralDims = find(with_spectral); % dimension with spectral transofrmation
            
            % remove dimensions where covariance matrix is already provided
            if ~isempty(obj.sigma)
                SpectralDims(~cellfun(@isempty, obj.sigma(1,SpectralDims))) = [];
            end
            % spectral = ~isempty(spk); % if any dimension with this module uses the spectral trick
            % if spectral
            
            % wvec = cell(1,length(spk)); % vector of Fourier frequencies for each component
            % B_fft = cell(1,DD); % basis for each component
            % B_fft2 = cell(1,DD); % same, excluding index dimensions
            
            
            % for f= 1:length(spk) % for each component to be converted
            %     d = spk(f);
            for d = SpectralDims
                
                % hyperparameter values for this component
                this_HP = obj.HP(d).HP; %fixed values
                
                if isa(obj.spectral{d}, 'function_handle') % directly provides function for basis change
                    
                    [S(d).B_fft,S(d).wvec,Tcirc] = S(d).fun(obj.scale{d}, this_HP); % apply user-provided function (params is hyperparameter)
                    assert( size(S(d).B_fft,2)==ss(d), 'Incorrect number of columns (%d) for spectral transformation matrix in component %d (expected %d)', size(S(d).B_fft,2), d, ss(d));
                    ss(d) = size(S(d).B_fft,1);
                else % compute basis for basis change
                    
                    minl = this_HP(1); % temporal scale
                    
                    range = max(obj.scale{d}) - min(obj.scale{d});  % range of values for given component
                    Tcirc = range + 3*minl; %   range + 3*minscale (for virtual padding to get periodic covariance)
                    
                    % if isfield(obj,'condthreshold') % threshold on condition number of covariance matrix
                    %     condthresh = obj.condthreshold(f);
                    % else
                    condthresh = 1e12;
                    % end
                    
                    % set up Fourier frequencies
                    maxw = floor((Tcirc/(pi*minl))*sqrt(.5*log(condthresh)));  % max freq to use
                    ss(d) = maxw*2+1; % number of fourier frequencies
                    
                    [S(d).B_fft,S(d).wvec] = realnufftbasis(obj.scale{d},Tcirc,ss(d)); % make Fourier basis
                    S(d).B_fft = S(d).B_fft';
                    warning('the covfun below looks weird, shouldnt i use RBF_Fourier instead?');
                end
                obj.covfun{1,d} = @(x) obj.covfun{1,d}(x, S(d).wvec,Tcirc); % use frequency scale for covariance prior function
                
                % convert design matrix to spectral decomposition along
                % required dimension
                % if d<=DD % component from main design matrix
                %                     if obj.sparse(d)
                %                         siz = size(obj.val{d+1});
                %                         siz(d+1) = ss(d);
                %                         VV = zeros(siz);
                %                         dimdim = 1:obj.nDim+1;
                %                         dimdim(d+1) = 1; dimdim(1) = d+1; % put along d+1 dimensions in design matrix
                %                         for i=1:obj.size(d)
                %                             VV = VV + (obj.val{d+1}==i) .* permute(B_fft{d}(:,i),dimdim); % add spectral component for this value
                %                         end
                %                         obj.val{1} =  obj.val{1} .* VV;
                %                         obj.sparse(d) = 0;
                %                         obj.val{d+1} = [];
                %                     else
                S(d).B_fft2 = S(d).B_fft;
                %                   end
                
                %% if weights are constraint to have mean or average one, change projection matrix so that sum of weights constrained to 1
                for r=1:rr
                    if any(cc(r,d)=='msb')
                        if ~all(cc(:,d) == cc(r,d))
                            error('spectral dimension %d cannot have ''%s'' constraint for one order and other constraint for other orders',d, cc(r,d));
                        end
                        B = sum(S(d).B_fft,2); % ( 1*(Bfft*U) = const -> (1*Bfft)*U = const
                        if cc(d)=='m' % rescale
                            B = B/obj.nRegressor(d);
                        end
                        invB = diag(1./B);
                        S(d).B_fft = invB * S(d).B_fft; % change projection matrix so that constraint is sum of weight equal to one
                        
                        for r2=1:rr
                            if cc(r,d)~='b'
                                obj.constraint(r2,d) = 's';
                            end
                            obj.covfun{r2,d} = @(x) covfun_transfo(x,  diag(B) , obj.covfun{r2,d} );
                        end
                        break; % do not do it for other orders
                    end
                    
                end
                
                %% if initial value of weights are provided, compute in new basis
                if ~isempty(obj.U)
                    for r=1:rr
                        % if some hyperparameters associated with other
                        % rank
                        % need to work this case!
                        % if ~isempty(idx{m}{r,d}) % if some hyperparameters associated with other rank
                        %     obj.covfun{r,d} =  obj.covfun{1,d};
                        % end
                        if ~isempty( obj.U{r,d})
                            obj.U{r,d} = obj.U{r,d}/ S(d).B_fft; % obj.U{r,d}*Bfft{d}';
                        end
                    end
                end
                
                % change basis to spectral for all components within main design matrix
                obj.val = tensorprod(obj.val, [{{}},S(d).B_fft2]);
                obj.nRegressor = ss; % size of each dimension
            end
            
            %     else % if not spectral
            %         if ~isempty(UU) % define best fitting parameters so far as initial values for next round, to speed up convergence (somehow diverges for spectral, shoould check why)
            %             obj.U = UU{m};
            %         end
            
            %   end
            obj.spectral = S;
            
        end
        
        %% PROJECT FROM SPECTRAL SPACE BATCH TO ORIGINAL SPACE
        function obj = project_from_spectral(obj)
            
            for m=1:length(obj) % for each module
                S = obj(m).spectral;
                if ~isempty(S) % if any dimension in spectral domain
                    
                    for d=1:obj(m).nDim
                        if length(S)>=d && ~isempty(S(d).fun)
                            S(d).U =  obj(m).U{d}; % save weight in fourier domain
                            S(d).se =  obj(m).se{d};
                            
                            
                            for r=1:obj(m).rank
                                obj(m).U{r,d} = obj(m).U{r,d} * S(d).Bfft; % compute coefficient back in original domain
                                obj(m).V{r,d} = S(d).Bfft' * obj(m).V{d} * S(d).Bfft; % posterior covariance in original domain
                                obj(m).se{r,d} = sqrt(diag(obj(m).V{r,d}))'; % standard error in original domain
                            end
                        end
                    end
                    obj(m).spectral = S;
                end
            end
        end
        
        
        
        %% COMPUTE PRIOR COVARIANCE (and gradient)
        function [obj,GHP] = compute_prior_covariance(obj)
            
            nMod = length(obj);
            grad_sf  = cell(1,nMod);
            with_grad = (nargout>1); % compute gradient
            
            for m=1:length(obj)
                
                %% first project to spectral domain
                obj(m) = project_to_spectral(obj(m));
                
                with_grad = (nargout>1); % compute gradient
                DD = obj(m).nDim; % number of dimensions for this module
                rr = obj(m).rank; % rank
                ss = obj(m).nRegressor; % size of each dimension
                
                %% evaluate prior covariance
                covf = obj(m).covfun;
                if isempty(obj(m).sigma)
                    sig = cell(rr,DD);
                else
                    sig = obj(m).sigma;
                end
                
                gsf = cell(rr,DD);
                for d=1:DD
                    if size(covf,1)==1 || isempty(covf{2,d})
                        rrr = 1;
                    else
                        rrr = rr;
                    end
                    % r=1;
                    for r=1:rrr
                        if ~isempty(sig{r,d})
                            % do nothing we already have it
                        elseif isa(covf{r,d}, 'function_handle') % function handle
                            
                            
                            % hyperparameter values for this component
                            this_HP = obj(m).HP(r,d).HP; %hyperparameter values
                            this_nHP = length(this_HP); % number of hyperparameters
                            
                            [sig{r,d}, gg]= covf{r,d}(this_HP); % compute associated covariance matrix (and gradient)
                            if isstruct(gg)
                                gg = gg.grad;
                            end
                            if size(gg,3) ~= this_nHP
                                error('For component %d and rank %d, size of covariance matrix gradient along dimension 3 (%d) does not match corresponding number of hyperparameters (%d)',...
                                    d,r, size(gg,3),this_nHP);
                            end
                            
                            %% compute gradient now
                            if with_grad
                                gsf{r,d} = zeros(size(gg)); % gradient of covariance w.r.t hyperparameters
                                for l=1:this_nHP
                                    sig = sig{r,d};
                                    if obj(m).constraint(r,d)=='1' % if first weight fixed to 1, invert only the rest
                                        % if rcond(sig(2:end,2:end))<1e-16
                                        %     fprintf('Covariance matrix for component %d and rank %d close to singular for hyperparameters',d,r);
                                        %     fprintf('%f,',P(cc));
                                        %     fprintf('\n');
                                        % end
                                        gsf{r,d}(:,:,l) = - blkdiag(0, (sig(2:end,2:end) \ gg(2:end,2:end,l)) / sig(2:end,2:end));% gradient of precision matrix
                                    else
                                        % if rcond(sig)<1e-16
                                        %     fprintf('Covariance matrix for component %d and rank %d close to singular for hyperparameters',d,r);
                                        %     fprintf('%f,',P(cc));
                                        %     fprintf('\n');
                                        % end
                                        % invsig = inv(sigma{r,d});
                                        % gsf{r,d}(:,:,l) = - invsig * gg(:,:,l) * invsig;% gradient of precision matrix
                                        gsf{r,d}(:,:,l) = - (sig \ gg(:,:,l)) / sig;% gradient of precision matrix
                                    end
                                end
                                
                                % select gradient only for fittable HPs
                                gsf{r,d} = gsf{r,d}(:,:,HP_fittable);
                            end
                        elseif isempty(covf{r,d}) % default (fix covariance)
                            sig{r,d} = [];
                            gsf{r,d} = zeros(ss(d),ss(d),0);
                        else % fixed custom covariance
                            sig{r,d} = covf{r,d};
                            gsf{r,d} = zeros(ss(r,d),ss(d),0);
                        end
                        if ((size(sig{r,d},1)~=obj(m).nRegressor(d)) || (size(sig{r,d},2)~=obj(m).nRegressor(d))) && ~isempty(sig{r,d})>0
                            error('covariance prior, dimension %d and rank %d should be square of size %d',d,r,obj(m).nRegressor(d));
                        end
                    end
                    
                    %end
                    
                    if rr>1 && (rrr==1) % rank>1 and same covariance function and HPs for each rank
                        % replicate covariance matrices
                        sig(:,d) = repmat(sig(1,d),rr,1);
                        
                        % extend gradient
                        if with_grad
                            % for d=1:DD
                            % this_nHP = length(obj(m).HP(d).HP);
                            gsf_new = zeros(ss(d)*rr,ss(d)*rr,this_nHP);
                            for l=1:this_nHP
                                gg = repmat({gsf{1,d}(:,:,l)},rr,1);
                                gsf_new(:,:,l) = blkdiag( gg{:});
                            end
                            gsf{1,d} = gsf_new;
                            % end
                        end
                    end
                    
                end
                
                
                obj(m).sigma = sig;
                
                grad_sf{m} = gsf(:)';
            end
            
            % gradient over module
            if with_grad
                grad_sf = [grad_sf{:}]; % concatenate over modules
                GHP = blkdiagn(grad_sf{:}); % overall gradient of covariance matrices w.r.t hyperparameters
            end
            
        end
        
        %% COMPUTE PRIOR MEAN
        function obj = compute_prior_mean(obj)
            dd = obj.nDim; % dimension in this module
            ss = obj.nRegressor; % size along each dimension in this module
            cc = obj.constraint; % constraint for this module
            rr = obj.rank;
            
            if  ~isempty(obj.mu)
                Mu = obj.mu;
                assert( size(Mu,2) == dd, 'incorrect number of columns for mu');
                if isvector(Mu) && rr>1
                    Mu = repmat(Mu,rr,1);
                end
            else
                Mu = cell(rr,dd);
                for r=1:rr
                    for d=1:dd
                        switch cc(r,d)
                            case {'f','n','b'} % mean 0
                                Mu{r,d} = zeros(1,ss(d));
                            case '1' % first weight 1, then 0
                                Mu{r,d} = [1 zeros(1,ss(d)-1)];
                            case 's' %
                                Mu{r,d} = ones(1,ss(d))/ss(d);
                            case 'm'
                                Mu{r,d} = ones(1,ss(d));
                        end
                    end
                end
            end
            obj.mu = Mu;
            
        end
        
        %% GLOBAL PRIOR COVARIANCE
        function sigma = global_prior_covariance(obj)
            sigma = cellfun(@(x) x(:)', {obj.sigma}, 'unif',0);
            sigma = [sigma{:}];
            sigma = blkdiag(sigma{:}); % block-diagonal matrix
        end
        
        %% INITIALIZE WEIGHTS
        function obj = initialize_weights(obj)
            ss = obj.nRegressor; % size along each dimension in this module
            cc = obj.constraint; % constraint for this module
            for r=1:obj.rank
                for d=1: obj.nDim
                    if isempty(obj.U{r,d})
                        if first_update_dimension(obj)==d && cc(r,d)~='n'
                            % if first dimension to update, leave as nan to
                            % initialize by predictor and not by weights (for
                            % stability of IRLS)
                            UU = nan(1,ss(d));
                        else
                            switch cc(r,d)
                                case 'f' %free basis
                                    if any(cc(r,1:d-1)=='f')
                                        UU = mvnrnd(obj.mu{r,d},obj.sigma{r,d});
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
                        obj.U{r,d} = UU;
                    elseif length(obj.U{r,d}) ~= ss(d)
                        error('number of weights for component %d and rank %d (%d) does not match number of corresponding covariates (%d)',d,r, length(obj.U{r,d}),ss(d));
                    end
                end
            end
            
        end
        
        %% sample weights from gaussian prior
        function obj = sample_weights_from_prior(obj)
            for i=1:numel(obj)
                PP = ProjectionMatrix(obj(i));
                
                for r=1:obj(i).rank
                    for d=1:obj(i).nDim
                        if obj(i).constraint(r,d) ~= 'n'
                            if d==first_update_dimension(obj(i)) % first dimension to update should be zero (if free, otherwise just standard)
                                obj(i).U{r,d} = obj(i).mu{r,d};
                            else % other dimensions are sampled for mvn distribution with linear constraint
                                pp = PP{r,d};
                                obj(i).U{r,d} =  obj(i).mu{r,d} + mvnrnd(zeros(1,size(pp,1)), pp*obj(i).sigma{r,d}*pp')*pp;
                            end
                            
                            switch obj(i).constraint(r,d)
                                case 'b' %mean zero
                                    obj(i).U{r,d} = obj(i).U{r,d} - mean(obj(i).U{r,d});
                                case 'm' % mean equal to one
                                    obj(i).U{r,d} = obj(i).U{r,d}/mean(obj(i).U{r,d}); % all weight set to one
                                case 's' % sun of weights equal to one
                                    obj(i).U{r,d} = obj(i).U{r,d}/sum(obj(i).U{r,d}); % all weights equal summing to one
                                case '1' % first weight set to one
                                    obj(i).U{r,d}(1) = 1;
                                    
                            end
                        else
                            obj(i).U{r,d} = ones(1,obj(i).nRegressor);
                        end
                    end
                end
            end
            
        end
        
        %% compute log-prior (unnormalized)
        function LP = LogPrior(obj,dims) % d is the dimension
            if nargin<2
                dims =1:obj.nDim;
            end
            
            LP = zeros(obj.rank,length(dims)); % for each rank
            for d=1:length(dims)
                d2 = dims(d);
                for r=1:obj.rank % assign new set of weight to each component
                    if obj.constraint(r,d2)~='n' % unless fixed weights
                        dif =  obj.U{r,d2} - obj.mu{r,d2}; % distance between weights from prior mean
                        LP(r,d) = - dif / obj.sigma{r,d2} * dif'/2; % log-prior for this weight
                    end
                    
                end
            end
            
        end
        
        %% first dimension to be updated in IRLS
        function d = first_update_dimension(obj)
            d = find(any(obj.constraint=='f',1),1); % start update with first free dimension
            if isempty(d)
                d = 1;
            end
        end
        
        %% compute predictor rho
        function  rho = Predictor(obj, rr)
            % compute predictor from regressor
            % rho = Predictor(R)
            % rho = Predictor(R,r) to compute for specific rank
            if isempty(obj.val)
                error('cannot compute predictor, data has been cleared');
            end
            rho = zeros(obj.nObs,1);
            if all(obj.nRegressor>0)
                if nargin<2
                    rr = 1:obj.rank;
                else
                    assert(all(ismember(rr, 1:obj.rank)));
                end
                for r=rr
                    rho = rho + ProjectDimension(obj,r,zeros(1,0)); % add activation due to this component
                end
            end
        end
        
        %% project regressor dimensions
        function P = ProjectDimension(obj,r,d, do_squeeze, Uobs, to_double)
            %  P = projdim(obj.val, obj.U(r,:), obj.sparse, d, varargin{:});  % add activation for component r
            
            if nargin<6 % by default, always convert output to double if sparse array
                to_double = true;
            end
            
            X = obj.val;
            VV = obj.U(r,:);
            for dd=d % dimensions over which we project (no tensor product over this one)
                VV{dd} = [];
            end
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
                dd=1;
                while dd<=length(d) && obj.nRegressor(d(dd))==1
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
                obj(i).val = [];
            end
        end
        
        
        %% extract design matrix for subset of observations
        function obj = extract_observations(obj,subset) %,dim)
                for i=1:numel(obj)
            S = size(obj(i).val);
            obj(i).val = reshape(obj(i).val, S(1), prod(S(2:end))); % reshape as matrix to make subindex easier to call
            obj(i).val = obj(i).val(subset,:);
            n_Obs = size(obj(i).val,1); % update number of observations
            if issparse(obj(i).val) && length(S)>2
                obj(i).val = sparsearray(obj(i).val);
            end
            obj(i).val = reshape(obj(i).val, [n_Obs, S(2:end)]); % back to nd array
            obj(i).nObs = n_Obs;
                end
        end
        
        %% PRODUCT OF REGRESSORS
        function obj = times(obj1,obj2)
            %M = regressor_multiplier( M1, M2,... )
            %   multiplies regressors for GUM
            %
            % See also gum, regressor
            
            if isnumeric(obj1)
                %if ndims(obj1)>=ndims(obj2.val) % sum over new dimensions
                %    obj1 = regressor2(obj1,'constant');
                %else
                obj2.val = obj1 .* obj2.val;
                obj = obj2;
                return;
            elseif isnumeric(obj2)
                % obj2 = regressor2(obj2,'constant');
                obj1.val = obj2 .* obj1.val;
                obj = obj1;
                return;
            end
            
            if obj1.nObs ~= obj2.nObs
                error('regressors should have the same number of observations');
            end
            obj = regressor();
            obj.nObs = obj1.nObs;
            
            nDim_all = [obj1.nDim obj2.nDim];
            
            obj.nDim =sum(nDim_all);
            
            %% size  properties
            
            % obj.sparse = [];
            obj.nRegressor = [];
            % if ~all( obj2.sparse)
            %  dimdim = 1:
            %     error('The second regressor cannot have non-sparse dimensions (not coded yet)');
            % end
            
            
            %    obj.sparse = [obj1.sparse obj2.sparse];
            obj.nRegressor = [obj1.nRegressor obj2.nRegressor];
            
            
            %% build design matrix
            if isnumeric(obj2.val) && issparse(obj2.val)
                obj2.val = sparsearray(obj2.val); % conver to sparse array class to allow for larger than 2 arrays
            end
            
            % permute second object to use different dimensions for each
            % object (except first dim = observations, shared)
            P = [1 obj2.nDim+1+(1:obj1.nDim) 2:obj2.nDim+1];
            obj2.val = permute(obj2.val, P);
            
            obj.val = obj1.val .* obj2.val;
            %obj.val = tocell(obj1.val); % start with first regressor
            %if ~obj1.sparse
            %    obj.val(2:nDim_all(1)+1) = {[]};
            %end
            %obj.val{1} =  obj.val{1}.*obj2.val{1}; % multiply by fixed term
            %obj.val(end+(1:obj2.nDim)) =obj2.val(2:end); % concatenate sparse dimensions
            
            
            %% other properties (if present in at least one regressor)
            obj.U = [obj1.U obj2.U];
            obj.HP = [obj1.HP obj2.HP];
            obj.scale = [obj1.scale obj2.scale];
            if ~isempty(obj2.spectral)
                spk = obj2.spectral(1); % create empty structure
                spk.fun = [];
                obj2.spectral = [repmat(spk,1, obj1.nDim) obj2.spectral];
                if isempty(obj1.spectral)
                    obj.spectral = obj2.spectral;
                else
                    obj.spectral = [obj1.spectral obj2.spectral];
                end
            else
                obj.spectral = obj1.spectral;
            end
            obj.spectral = [obj1.spectral obj2.spectral];
            obj.label = [obj1.label obj2.label];
            obj.formula = [obj1.formula ' * ' obj2.formula];
            obj.covfun = [obj1.covfun obj2.covfun];
            
            freereg = [obj1.constraint=='f', obj2.constraint=='f'];
            if sum(freereg)>1
               % fprintf('multiplying free regressors ... now only first will be free\n');
                % turning free regressors after first one to one-mean regressor
                
                obj2.constraint = strrep(obj2.constraint,'f','m');
            end
            obj.constraint = [obj1.constraint obj2.constraint];
            
            obj.plot = [obj1.plot obj2.plot];
            
            
        end
        
        % R1 * R2 is the same as R1 .* R2
        function obj = mtimes(varargin)
            obj = times(varargin{:});
        end
        
        %% SUM OF REGRESSORS
        function obj= plus(obj1,obj2)
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
            
            obj = [obj1 obj2];
        end
        
        %% SPLIT/CONDITION ON REGRESSORS
        function obj = split(obj,X, scale, label)
            % obj = split(obj,X, do_sparse, scale)
            
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
                if ~size(X,d+1)==1 && ~size(X,d+1)==obj.nRegressor(d)
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
            
            nVal = length(scale);
            if nVal==1 % if just one value,let's keep it identical
                return;
            end
            
            % add as new dummy dimension
            d = obj.nDim+1;
            %             obj.nDim =d;
            %           %  obj.sparse(d) = subindex_coding;
            %             obj.scale{d} = scale;
            %             obj.size(d) = nVal;
            X = replace(X, scale); % replace each value in the vector by its index
            
            %             %% values along new dimension
            %           %  obj.val = tocell(obj.val);
            %             if subindex_coding
            %
            %                 obj.val = sparsearray(obj.val);
            %                 obj.val = define_subindices(obj.val,d+1, X, nVal);
            %                % obj.val{d+1} = X;
            %
            %
            %             else % non-sparse coding
            %                                 val1 = obj.val;
            %
            %                % val1 = obj.val{1};
            %                 % size of diesgn matrix
            %                 % if iscolumn(val1), siz = length(val1);
            %                 % else
            %                 siz = size(val1);
            %                 % end
            %                 siz(d+1) = nVal;
            %
            %                 % index for matrix
            %                 idx = cell(1,length(siz));
            %                 for i=1:length(siz)
            %                     idx{i} = 1:siz(i);
            %                 end
            %
            %                 if issparse(val1)
            %                     obj.val = sparsearray('empty',siz,nnz(val1)); % empty n-dimensional sparse array
            %                 else
            %                 obj.val = zeros(siz);
            %                 end
            %                 for u=1:nVal % one column for each value
            %                     idx{d+1} = u;
            %                     obj.val(idx{:}) = val1 .* (X==u); % true if corresponding observation value and corresponding column
            %                 end
            %                % obj.val{d+1} = [];
            %             end
            
            %% replicate design matrix along other dimensions
            for dd = 1:obj.nDim
                
                if isa(obj.val,'sparsearray') && subcoding(obj.val,dd+1) %obj.sparse(dd)
                    %  shift_value = obj.val{dd+1} * (0:nVal-1); % shift in each dimension (make sure we use a different set of indices for each value along dimension d
                    %  shift_size = ones(1,1+obj.nDim);
                    %  shift_size(d) = nVal;
                    %  shift = reshape(shift_value,shift_size);
                    shift = (X-1)*obj.nRegressor(dd);  % shift in each dimension (make sure we use a different set of indices for each value along dimension d
                    
                    obj.val.sub{dd+1} = obj.val.sub{dd+1} +  shift;
                    obj.val.siz(dd+1) = obj.val.siz(dd+1)*nVal;
                else
                    VV = obj.val;
                    new_size = size(VV);
                    new_size(dd+1) = new_size(dd+1)*nVal; % each regressor is duplicated for each value along dimension d
                    obj.val = zeros(new_size);
                    
                    idx = cell(1,ndims(VV));
                    for ee = 1:ndims(VV)
                        idx{ee} = 1:size(VV,ee);
                    end
                    
                    for r=1:nVal
                        idx{1} = find(X==r); % focus on r-th value
                        idx2 = idx;
                        idx2{dd+1} = (r-1)*size(VV,dd+1) + idx{dd+1};
                        obj.val(idx2{:}) = VV(idx{:}); % copy content
                    end
                end
                
                
                %% build covariance function as block diagonal
                if ~isempty(obj.covfun{dd})
                    obj.covfun{dd} = @(P)  replicate_covariance(obj.covfun{dd}, P, nVal);
                end
                
                obj.nRegressor(dd) = obj.nRegressor(dd)*nVal;
            end
            
            
            %             % set this dimension as constant
            %             %  obj.nRegressor(d) = 1;
            %             obj.HP(d) = HPstruct;
            %             obj.covfun{d} = {};
            %             obj.constraint(d) = 'n';
            
            % add label if required
            if nargin>=4
                if length(label) ~= length(scale)
                    error('length of label does not match number of values');
                end
                obj.scale{d} = label;
                
            end
        end
        
        
    end
end




%% creates a design matrix with one regressor for each value of xx
function val = indicator_regressors(xx, allval,SubindexCoding, nD, summing)
nVal = length(allval);
siz = size(xx);
%if any(strcmp(summing,{'split','separate'})) % different function for each column
%    f_ind = repmat(0:prod(siz(2:end))-1,siz(1),1); % indicator function
%    f_ind = reshape(f_ind, siz);
%    nFun = prod(siz(2:end)); % total number of weights
%else
f_ind = zeros(siz);
nFun = 1;
%end


xx = replace(xx,allval); % replace each value in the vector by its index
sub = xx + nVal*f_ind;
nValTot = nVal*nFun;

if SubindexCoding
    % dimension for regressor
    %  val = cell(1,nD+1);
    %  val{1} = ones(size(xx)); % neutral value on first dim
    %  val{nD+1} = sub;
    
    % if 0
    val = sparsearray(ones(size(xx)));
    val = define_subindices(val,nD+1, sub, nValTot);
    
    % end
    %  val{nD+1} = zeros(size(xx));
    %  for u=1:nval % replace each value in the vector by its index
    %      idx = xx==u;
    %      val{nD+1}(idx) = u + nval*f_ind(idx); % mark observation with given value of input
    %  end
    
else % non-sparse
    % size of diesgn matrix
    if iscolumn(xx), siz = length(xx);
    else
        siz = size(xx);
    end
    siz(end+1) = nVal*nFun;
    
    % index for matrix
    idx = cell(1,length(siz));
    for i=1:length(siz)-1
        idx{i} = 1:siz(i);
    end
    
    val = zeros(siz);
    for u=1:nVal % one column for each value
        for w=1:nFun
            idx{end} = u + nVal*(w-1);
            val(idx{:}) = (xx==u) & (f_ind==w-1); % true if corresponding observation value and corresponding column
        end
    end
end
end


%% L2-regression covariance function
%function [K, grad] = L2_covfun(nreg, loglambda,HPdefault, which_par)
function [K, grad] = L2_covfun(nreg, loglambda)
% if ~which_par
%     loglambda = HPdefault; % default value
% end
lambda2 = exp(2*loglambda);
if isinf(lambda2)
    K = diag(lambda2 * ones(1,nreg));
else
    K = lambda2 * eye(nreg); % diagonal covariance
end

%if which_par
grad.grad = 2*K; %eye(nreg);
grad.EM = @(m,V) log(mean(m(:).^2 + diag(V)))/2; % M-step of EM to optimize L2-parameter given posterior on weights
%else
%    grad = zeros(nreg,nreg,0);
%end
end

%% Squared Exponential covariance
%function [K, grad] = covSquaredExp(X, HP, HPdefault, which_par, nrep)
function [K, grad] = covSquaredExp(X, HP, nrep)
% radial basis function covariance matrix:
%  K = covSquaredExp(X, [log(tau), log(rho)], incl)
% K(i,j) = rho^2*exp(- [(x(i,1)-x(j,1))^2 + ...
% (x(i,d)-x(j,d))^2]/(2*tau^2))
%
% tau can be scalar (same scale for each dimension) or a vector of length D
%
% [K, grad] = covSquaredExp(X, tau, rho, incl) get gradient over
% hyperparameters, incl is a vector of two booleans indicating which
% hyperparameter to optimize
%

%HPfull = HPdefault; % default values
%HPfull(which_par) = HP; % active hyperparameters

X = X';

tau = exp(HP(1:end-1));
rho2 = exp(2*HP(end));

n_tau = length(tau);
if n_tau ~=1 && n_tau ~= size(X,2)
    error('tau should be scalar or a vector whose length matches the number of column in X');
end
tau = tau(:);
if length(tau)==1
    tau = repmat(tau,size(X,2),1);
end
n = size(X,1); % number of data points
Xnorm = X'./tau;
nulltau = tau==0;

xx = Xnorm(~nulltau,:);
D = zeros(size(xx,2));  % cartesian distance matrix between each column vectors of X
for i=1:size(xx,2)
    for j=1:i
        D(i,j) = sqrt(sum((xx(:,i)-xx(:,j)).^2));
        D(j,i) = D(i,j);
    end
end
if all(nulltau)
    D = zeros(n);
end

% treat separately for null scale
for tt=1:find(nulltau)
    Dinf = inf(n); % distance is infinite for all pairs
    Dinf( dist(X(:,tt)')==0) = 0; % unless values coincide
    D = D + Dinf;
end

K = rho2* exp(-D.^2/2);

% if separate functions for each column, full covariance matrix is block
% diagonal
if nrep>1
    K = repmat({K}, 1, nrep);
    K = blkdiag(K{:});
end

% compute gradient
if nargout>1
    grad = zeros(n,n,n_tau+1); % pre-allocate
    if n_tau == 1
        grad(:,:,1) = rho2* D.^2 .* exp(-D.^2/2); % derivative w.r.t GP scale
        
        
    else
        for t=1:n_tau % gradient w.r.t scale for each dimension
            
            dd = bsxfun(@minus, X(:,t),X(:,t)').^2; % distance along dimension
            grad(:,:,t) = log(rho2) * dd .* exp(-D.^2/2)/tau(t)^2; % derivative w.r.t GP scale
        end
    end
    grad(:,:,n_tau+1) = 2*K; % derivative w.r.t GP log-variance (log-rho)
    % grad(:,:,n_tau+1) = exp(-D.^2/2); % derivative w.r.t GP variance (rho)
    % grad(:,:,n_tau+2) = eye(n); % derivative w.r.t. innovation noise
    % grad = grad(:,:,which_par);
    
    if nrep>1
        GG = grad;
        for hp=1:size(GG,3)
            gc = repmat({grad(:,:,hp)}, 1, nrep);
            GG(:,:,hp) = blkdiag(gc{:});
        end
        grad = GG;
    end
end

end


%% Periodic covariance function
%function [K, grad] = cov_periodic(x, HP, which_par, period, nrep)
function [K, grad] = cov_periodic(x, HP, period) %, nrep)
%HPfull = [period 1]; % default values
%HPfull(which_par) = HP; % active hyperparameters

ell = exp(HP(1));
sf = exp(2*HP(2));

% use function from GPML
%[cov, dK] = covPeriodic([log(ell) log(period) log(sf)], x);

%adapted from GPML
T = pi/period*bsxfun(@plus,x,-x');
S2 = (sin(T)/ell).^2;

K = sf*exp( -2*S2 );

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

%% REPLICATE COVARIANCE MATRIX (AND GRADIENT) WHEN SPLITTING REGRESSOR
function [cov, grad] = replicate_covariance(covfun, P, nRep)
% [cov, grad] = replicate_covariance(covfun, nRep)

C = cell(1,nargout);
[C{:}] = covfun(P);

% full covariance matrix is block
% diagonal
cov = C{1};
cov = repmat({cov}, 1, nRep);
cov = blkdiag(cov{:});

% compute gradient
if nargout>1
    gg = C{2};
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

%% combine default and specified HP values
function   HP = HPwithdefault(HPspecified, HPdefault)
HP = HPdefault;
HP(~isnan(HPspecified)) = HPspecified(~isnan(HPspecified)); % use specified values
end

%% void HP structure
function S = HPstruct()
S.HP = []; % vlaue of hyperparameter
S.label = {}; % labels
S.fit = []; % which ones are fittable
S.LB = []; % lower bound
S.UB = []; % upper bound
end

%% HP strucutre for L2 regularization
%% void HP structure
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

%% maximum log variance (to avoid non-invertible covariance matrices)
function mlv = max_log_var()  %
mlv = 20;
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
function [cov, grad] = covfun_transfo(x, M,covfun)
[cov,grad] = covfun(x);
cov = M*cov*M';
for i=1:size(grad,3)
    grad(:,:,i) = M*grad(:,:,i)*M';
end
end


%% replace value by index
function I = replace(X, scale) % replace X values by indices

I = zeros(size(X));

for u=1:length(scale) % replace each value in the vector by its index
    I( X==scale(u)) = u;
end
end