function R = timeregressor(EventTime, dt, Tbnd, varargin)

% timeregressor creates regressors for time series regression (using gum)
% from a vector of event times
%
%R = timeregressor(EventTime, dt, [Tstart Tend])
% - EventTime is the vector of event times
% - dt is the time step for the created regressor
% - Tstart and Tend are the initial and final time for the regressor
%
% - R = timeregressor(... 'duration', D)
% with D scalar creates regressor with a certain duration D. If D is a
% vector, then duration D(i) is applied to event at time EventTime(i).
%
% - R = timeregressor(... 'duration', 'untilnext') so that duration
% corresponds to the time until the next event (this is typically used in
% conjunction with 'modulation' option to create a continuous regressor
% with step-like behavior
%
% R = timeregressor(... 'duration', D, 'ramp')
% constructs a regressor with ramping value of duration D (from 0 to 1)
%
% R = timeregressor(..., 'kernel', [kini kend])
% creates time-shifted regressors with all time-shifts from kini to kend
%
% R = timeregressor(..., 'kernel', [kini kend], 'kerneltype', type)
% 'independent' [default],'exponentialN' (e.g. 'exponential4', etc.), 'raisedcosineN','gaussian' (default)
%
% R = timeregressor(...,'timescale',T) to define time scale hyperparameter
% of kernel in second (default: 0.05 s)
%
% R = timeregressor(...,'variance',V) to define variance hyperparameter
% (L2 regularization for independent kernel). Use Inf for no variance
% (default:1)
%
% R = timeregressor(..., 'modulation', M)
%
% R = timeregressor(..., 'split', S)
% R = timeregressor(..., 'split', S, 'split_label',{'val1','val2',...})
% R = timeregressor(..., 'split', S, 'split_value',[S1 S2...Sn])
%
% R = timeregressor(....,'label',str)
%
% R = timeregressor('constant',dt, [Tstart Tend]) to capture baseline
% log-firing rate
%
% R = timeregressor('dt',dt, [Tstart Tend])
%
% See also regressor, gum

% Todo: work on labels

if ~isscalar(dt) || dt<=0
    error('dt should be a positive scalar');
end
if length(Tbnd)~=2 || Tbnd(1)>=Tbnd(2)
    error('Tbnd should a vector of two elements and the first element should be the smaller');
end

%n = ceil(diff(Tbnd)/dt); % number of time bins
n = 1+floor(diff(Tbnd)/dt); % number of time bins

if ischar(EventTime)
    switch lower(EventTime)
        case 'constant'
            R = regressor(ones(n,1),'linear','label','log-baseline rate');
        case 'dt'
            x = log(dt)*ones(n,1);
            R = regressor(x,'constant','label','log-dt');
        otherwise
            error('incorrect first argument');
    end
    return;

elseif ~isvector(EventTime)
    error('T should be a vector');
end


%% process arguments

% default parameters values
duration = [];
split = [];
split_value = [];
split_label = {};
ramp = false;
kernel = [];
M = [];
timescale = .05;
kerneltype = 'independent';
variance = 1;
nEvent = length(EventTime);
label = '';
use_reg = false;

v = 1;
while v<length(varargin)
    switch lower(varargin{v})
        case 'duration'
            v = v+1;
            duration = varargin{v};
        case 'ramp'
            ramp = true;
        case 'kernel'
            v = v+1;
            kernel = varargin{v};
            if length(kernel)~=2 || kernel(1)>=kernel(2)
                error('kernel must be a vector of two sorted elements');
            end
        case 'kerneltype'
            v = v+1;
            kerneltype = varargin{v};

        case 'timescale'
            v = v+1;
            timescale = varargin{v};
        case 'variance'
            v = v+1;
            variance = varargin{v};
        case 'modulation'
            v = v+1;
            M = varargin{v};
            use_reg = isa(M, 'regressor');
            if ~use_reg
                if isscalar(M)
                    M = M*ones(nEvent,1);
                elseif ~isempty(M) && (~isvector(M) || length(M) ~=nEvent)
                    error('M should be a vector of the same length as EventTime');
                end
                M = M(:); % make sure it's a column vector
            end
        case 'split'
            v = v+1;
            split = varargin{v};
            if ~isempty(split) && (~isvector(split) || length(split) ~=nEvent)
                error('S should be a vector of the same length as EventTime');
            end
        case 'split_value'
            v = v+1;
            split_value = varargin{v};
        case 'split_label'
            v = v +1;
            split_label = varargin{v};
            if ~iscell(split_label)
                error('split_label should be cell array of labels or values');
            end
        case 'label'
            v = v+1;
            label = varargin{v};
        otherwise
            error('incorrect argument: %s', varargin{v});
    end
    v = v+1;
end

if use_reg && ~isempty(split)
    error('regressor modulation and split variables are not compatible at the moment');
end

%% compute basic regressor (no time shift)

%% compute basic regressor (before unrolling in time)
modulation = ~isempty(M);


if isequal(duration, 'untilnext')
    % compute time until next event (for last event, time until final
    % time step)
    duration = diff([EventTime;Tbnd(2)]);
end

if any(EventTime<Tbnd(1)) || any(EventTime>Tbnd(2))
    nRemove = sum(EventTime<Tbnd(1) | EventTime>Tbnd(2));
    if ~isempty(label)
        fprintf('%s regressor: removing %d events outside of bounds\n', label, nRemove);
    else
        fprintf('removing %d events outside of bounds\n', label, nRemove);
    end
    if modulation
        M(EventTime<Tbnd(1),:) = [];
        M(EventTime>Tbnd(2),:) = [];
    end
    EventTime(EventTime<Tbnd(1)) = [];
    EventTime(EventTime>Tbnd(2)) = [];
    nEvent = length(EventTime);
end

% time from starting time and express as time bin
EventTimeBin =  (EventTime - Tbnd(1))/dt;
%ceil((EventTime-Tbnd(1))/dt);
%EventTimeBin(EventTimeBin==0) = 1; % just in case an event coincided with lower time bound

% shape of regressor (delta, uniform with interval, duration)
if ramp && isempty(duration)
    error('for ramping signal, you must provide the duration');
end

if ~isempty(duration)
    if length(duration)==1
        duration = duration*ones(nEvent,1); % same duration for all events
    elseif length(duration)~=nEvent
        error('duration must be a scalar or a vector of the same length as EventTime');
    end
    if any(duration<=0)
        error('duration must be a vector of positive values');
    end
    duration = duration/dt; % convert to time bin
    %duration = round(duration/dt)+1; % convert to time bins

    %!!!! warning add padded time to avoid trimming duration with negative
    %value kernel
end

modulation = ~isempty(M);


doSplit = ~isempty(split);
if doSplit
    if isempty(split_value)
        % if not provided, use all levels of value
        split_value = unique(split);
    else
        % check events with values not in list
        excl_event = true(nEvent,1);
        for v=1:length(split_value)
            excl_event(select_from_val(split, split_value, v)) = false;
        end

        % exclude corresponding events
        if any(excl_event)
            EventTimeBin(excl_event) = [];
            nEvent = length(EventTimeBin);
            split(excl_event) = [];
            if modulation
                M(excl_event) = [];
            end
            if ~isempty(duration)
                duration(excl_event) = [];
            end
        end
    end
    nSplit = length(split_value);

    if isempty(split_label)
        split_label = split_value;
    end
else
    nSplit = 1;
    %s = 1;
end

%% CONVERT TO VECTOR OF EVENT COUNTS PER BIN

if isempty(duration) % if discrete events
    % nz = 2*nEvent;  % number of non-zero values

    % we split the regressor between the two time steps closest to event
    % time
    deci = mod(EventTimeBin,1); % decimal part (fraction of time bin)
    %EventTimeBin = floor(EventTimeBin);
    EventTimeBin = floor(EventTimeBin)+1;

    % row indices (time)
    I = [EventTimeBin EventTimeBin+1];

    % column indices
    if doSplit % use split value as column
        J = zeros(nEvent,1);
        for u=1:length(split_value)
            J(select_from_val(split,split_value,u)) = u;
        end
        J = repmat(J,1,2);
    elseif use_reg % use event as column, as we will later project regressor from event to time
        J = [1:nEvent 1:nEvent];
    else
        J = ones(2*nEvent,1);
    end

    % values
    V = [1-deci deci]; % relative part assigned to each time bin
    if modulation && ~use_reg
        V = V .* M;
        %      V = V .* [M;M]';
    end

    if any(EventTimeBin==n) % in the unfortunate case where there is an event right at the end, avoid issues with index end+1
        I(EventTimeBin==n,2) = nan;
        I = I(:);
        J = J(:);
        V = V(:);
        J(isnan(I)) = [];
        V(isnan(I)) = [];
        I(isnan(I)) = [];
    end

else % if modelling response with certain duration (uniform or ramp)
    nz = sum(ceil(duration+1));
    I = zeros(nz,1);
    J = ones(nz,1);
    V = zeros(nz,1);

    ii = 1;
    %end
    % EventCount = spalloc(n,nSplit,nz);
    for i=1:nEvent
        etb = EventTimeBin(i);
        deci = mod(etb,1); % decimal part (fraction of time bin)
        etb = floor(etb)+1;

        % which vector to add to regressor for each event
        % if isempty(duration)
        %     K = [1-deci deci]; % simple delta

        % else
        duration_bin = deci + duration(i);
        deci2 = mod(duration_bin,1); % decimal part in last bin
        db = floor(duration_bin)-1; % number of time bins fully covered by interval

        % constant signal, proportion with overlap of interval with bins
        K = [1-deci ones(1,db) deci2];

        if ramp % portion of integral of signal in the time step ( = h * (mean t in the interval))
            K = K .* [(1-deci)/2 1-deci+.5+(0:db-1) duration(i)-deci2/2];
            K = K/duration(i);
        end

        %end
        if modulation && ~use_reg
            K = K*M(i); % modulation
        end

        %     % update vector
        %     idx = etb + (0:length(K)-1);
        %     if idx(end)>n % trim if it goes out of bounds
        %         K(idx>n) = [];
        %         idx(idx>n) = [];
        %     end
        %
        %     % for split, to which column we should assign it
        %     if doSplit
        %         s = find(split(i)==unq);
        %     end
        %
        %     EventCount(idx,s) = EventCount(idx,s) + K';

        % update vector
        l = length(K);
        if etb+l-1>n % trim if it goes out of bounds
            K(etb+(0:l-1)>n) = [];
            l = length(K);
        end

        idx = ii + (0:l-1);
        ii = ii + l;

        % row indices
        I(idx) = etb+(0:l-1);

        if doSplit % for split, to which column we should assign it
            J(idx) = find(select_from_val(split_value, split,i));
        elseif use_reg % for regressor input, columns represent event index
            J(idx) = i;
        end

        % values
        V(idx) = K;

        % EventCount(idx,s) = EventCount(idx,s) + K';
    end
    I(ii:end) = [];
    J(ii:end) = [];
    V(ii:end) = [];

end

if use_reg % number of columns
    nCol = nEvent;
else
    nCol = nSplit;
end

% create sparse matrix (note: if there's an overlap in indices of I and J
% across events, values are summed - which is great!)
EventCount=sparse(I(:), J(:),V(:),n, nCol);

% basis functions for temporal kernel
basis = [];


%% ADD KERNEL (TIME SHIFT)
if ~isempty(kernel)

    % kernel = round(kernel/dt); % convert to time bin
    % nK = diff(kernel)+1; % number of time bins
    kernel = ceil(kernel/dt); % convert to time bin
    nK = diff(kernel); % number of time bins
    nz = nnz(EventCount)*nK; % number of non-zero values

    II = zeros(nz,1); % row indices (time)
    JJ = zeros(nz,1); % column indices (regressor)
    KK = zeros(nz,1); % 3rd dim indices (split value)
    VV = zeros(nz,1); % values
    cnt = 0;

    %     II = []; % row indices (time)
    %     JJ = []; % column indices (regressor)
    %     KK = []; % 3rd dim indices (split value)
    %     VV = []; % values

    for s=1:nCol
        % get indices and values
        [iE,~,vE] = find(EventCount(:,s));
        nE = length(iE);

        for k=1:nK % for each time shift
            binshift = kernel(1)+k-1; %number of bin shift

            %             II = [II; iE+binshift]; % vector of shifted event times
            %             JJ = [JJ; k*ones(nE,1)]; %regressor index
            %             KK = [KK; s*ones(nE,1)]; % split value
            %             VV = [VV; vE]; % regressor value
            idx = cnt + (1:nE); % indices
            II(idx) = iE+binshift; % vector of shifted event times
            JJ(idx) = k; %regressor index
            KK(idx) =s; % split value
            VV(idx) = vE; % regressor value
            cnt = cnt+nE;
        end
    end

    % remove if time index is out of range due to shifting
    VV(II<1 | II>n) = [];
    KK(II<1 | II>n) = [];
    JJ(II<1 | II>n) = [];
    II(II<1 | II>n) = [];

    if doSplit % 3d sparse array
        DM = sparsearray([II JJ KK],VV, [n,nK,nSplit]); % time step x regressor x split value array
    elseif use_reg
        DM = sparsearray([II KK JJ],VV, [n,nEvent,nK]); % time step x event x regressor array

        DM_dummy = sparse(II,JJ,VV, n, nK); % dummy design matrix (time step x regressor array)
    else
        DM = sparse(II,JJ,VV, n, nK); % time step x regressor array
    end

    %             if doSplit
    %
    %     DM = sparsearray('empty',[n,nK,nSplit],nz); % design matrix (one column per time shift)
    %     for s=1:nSplit
    %         for k=1:nK % for each time shift
    %             binshift = kernel(1)+k-1; %number of bin shift
    %             if binshift>=0
    %                 DM(:,k,s) = [sparse(binshift,1); EventCount(1:end-binshift,s)];
    %             else
    %                 DM(:,k,s) = [EventCount(1-binshift:end,s); sparse(-binshift,1)];
    %             end
    %         end
    %     end
    %     else
    %         DM = spalloc(n,nK, nz); % design matrix (one column per time shift)
    %         for k=1:nK % for each time shift
    %             binshift = kernel(1)+k-1; %number of bin shift
    %             if binshift>=0
    %                 DM(:,k) = [sparse(binshift,1); EventCount(1:end-binshift,s)];
    %             else
    %                 DM(:,k) = [EventCount(1-binshift:end,s); sparse(-binshift,1)];
    %             end
    %
    %         end
    %     end

    %     if doSplit
    %         DM = reshape(DM,n,nK*nSplit);
    %     end

    switch kerneltype
        case 'independent'
            summing = 'linear';
            HP = log(variance);
        case 'gaussian'
            summing = 'continuous';
            HP = [log(timescale) log(variance)]; % values of hyperparameters (log-time scale and log-variance)
        otherwise
            summing = 'continuous';
            basis = kerneltype;
            if length(kerneltype)>3 && strcmp(kerneltype(1:3), 'exp')
                nExp = str2double(kerneltype(4:end)); % number of exponentials
                % default hyperparameter values
                HP = linspace(log(dt), log(diff(kernel)),nExp); % log-time scale are uniformally spaced between dt and the kernel width
                HP(end+1) = 0; % variance hyperparameter
            elseif length(kerneltype)>12 && strcmp(kerneltype(1:12), 'raisedcosine')
                HP = [];
            else
                error('incorrect kernel type: %s', kerneltype);
            end
    end

    if doSplit
        summing = {summing, 'split'}; % split kernels along 2nd dimension
    end

    scale = dt*(kernel(1)+(0:nK-1));

else % no convolution
    HP = log(variance);
    DM = EventCount;
    summing = 'linear';
    scale = label;
end


if use_reg
    % if uses regressor object as modulation
    if isempty(kernel)
        % project event to time in regressor array
        R = M.project_observations(EventCount);
    else

        M = M.add_dummy_dimension(1); % add dummy dimension in regressor at 1 as a placeholder for timeshift

        objval = cell(1,nK);
        for k=1:nK % for each time shift
            % project event to time in regressor array for this time shift
            P = matrix(DM(:,:,k)); % projection matrix
            R = M.project_observations(P);
            objval{k} = R.Data; % data array for this time shift
        end

        % now concatenate arrays from all time shifts
        R.Data = cat(2,objval{:});

        % create other regressor object without modulation just to get
        % weights properties
        Rdummy = regressor(DM_dummy, 'linear','scale',scale,'sum',summing, 'basis', basis,'hyperparameter',HP,'label',label);

        R.Weights(1) = Rdummy.Weights;
        R.Prior(1) = Rdummy.Prior;
        R.HP(1) = Rdummy.HP;

        %  W = R.Weights;
        %  W(1).nWeight = nK;
        %  W(1).scale = scale;
        %          R.Weights = W;

        if any([R.Weights.constraint]=='f') % if any other dimension is without constraint
            R.Weights(1).constraint = 'm'; % we constraint the dynamics to be 1 on average to make sure our model is identifiable
        end
    end
else
    % create regressor object
    R = regressor(DM, 'linear','scale',scale,'sum',summing, 'basis', basis,'hyperparameter',HP,'label',label);

    if doSplit && ~isempty(split_label)
        if length(split_label)~=nSplit
            error('number of split values (%d) does not match number of split labels (%d)',  nSplit, length(split_label));
        end
        R.Weights(1).plot = {{{}, split_label}};
    end


end
end

% select events corresponding to value
function bool = select_from_val(E, F, val)
if iscell(E)
    bool = strcmp(E, F{val});
else
    bool = E==F(val);
end

end
