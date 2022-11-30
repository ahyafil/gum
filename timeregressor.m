function R = timeregressor(EventTime, dt, Tbnd, varargin)

%R = timeregressor(EventTime, dt, [Tstart Tend])
%
% R = timeregressor(... 'duration', D)
% D scalar or vector
%
% R = timeregressor(... 'duration', D, 'ramp')
%
%R = timeregressor(..., 'kernel', [kini kend])
%
% R = timeregressor(..., 'kernel', [kini kend], 'kerneltype', type)
% 'gaussian' (default), 'independent', 'shiftedcosine'
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
%
% R = timeregressor(....,'label',str)
%
% R = timeregressor('constant',dt, [Tstart Tend]) to capture baseline
% log-firing rate
%
% R = timeregressor('dt',dt, [Tstart Tend])


if ~isscalar(dt) || dt<=0
    error('dt should be a positive scalar');
end
if length(Tbnd)~=2 || Tbnd(1)>=Tbnd(2)
    error('Tbnd should a vector of two elements and the first element should be the smaller');
end

n = ceil(diff(Tbnd)/dt); % number of time bins

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
            if isscalar(M)
                M = M*ones(nEvent,1);
            elseif ~isempty(M) && (~isvector(M) || length(M) ~=nEvent)
                error('M should be a vector of the same length as EventTime');
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



%% compute basic regressor (before unrolling in time)
modulation = ~isempty(M);


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

doSplit = ~isempty(split);
if doSplit
    if isempty(split_value) % if not provided, use all levels of value
    split_value = unique(split);
    else
        % check events with values not in list
        excl_event = true(nEvent,1);
        for v=1:length(split_value)
           excl_event(split==split_value(v)) = false; 
        end
        
        % exclude thos events
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
else
    nSplit = 1;
    %s = 1;
end

%% convert to vector of event counts per bin

if isempty(duration) % number of non-zero values
    % nz = 2*nEvent;
    
    deci = mod(EventTimeBin,1); % decimal part (fraction of time bin)
    EventTimeBin = floor(EventTimeBin)+1;
    
    % row indices
    I = [EventTimeBin EventTimeBin+1];
    
    % column indices
    if doSplit
        J = zeros(nEvent,1);
        for u=1:length(split_value)
            J(split==split_value(u)) = u;
        end
        J = repmat(J,1,2);
    else
        J = ones(nEvent,2);
    end
    
    % values
    V = [1-deci deci];
    if modulation
        V = V .*M;
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

else
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
        if modulation
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
            K(etb+l-1>n) = [];
            l = length(K);
        end
        
        idx = ii + (0:l-1);
        ii = ii + l;
        
        % row indices
        I(idx) = etb+(0:l-1);
        
        
        % for split, to which column we should assign it
        if doSplit
            J(idx) = find(split(i)==split_value);
        end
        
        % values
        V(idx) = K;
        
        % EventCount(idx,s) = EventCount(idx,s) + K';
    end
    I(ii:end) = [];
    J(ii:end) = [];
    V(ii:end) = [];
    
end

% create sparse matrix
EventCount = sparse(I(:), J(:),V(:),n, nSplit);


%% add kernel
if ~isempty(kernel)
    
    kernel = round(kernel/dt); % convert to time bin
    nK = diff(kernel)+1; % number of time bins
    nz = nnz(EventCount)*nK; % number of non-zero values
    
    
    
    II = []; % row indices
    JJ = []; % column indices
    KK = []; %·rd dim indices
    VV = []; % values
    
    for s=1:nSplit
        % get indices and values
        [iE,~,vE] = find(EventCount(:,s));
        nE = length(iE);
        
        for k=1:nK % for each time shift
            binshift = kernel(1)+k-1; %number of bin shift
            
            II = [II; iE+binshift];
            JJ = [JJ; k*ones(nE,1)];
            KK = [KK; s*ones(nE,1)];
            VV = [VV; vE];
        end
    end
    
    % remove if time index is out of range due to shifting
    VV(II<1 | II>n) = [];
    KK(II<1 | II>n) = [];
    JJ(II<1 | II>n) = [];
    II(II<1 | II>n) = [];
    
    if doSplit % 3d sparse array
        DM = sparsearray([II JJ KK],VV, [n,nK,nSplit]);
    else
        DM = sparse(II,JJ,VV, n, nK);
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
    
    if ~strcmp(kerneltype, 'independent')
        HP = [log(timescale) log(variance)]; % values of hyperparameters (log-time scale and log-variance)
        
        %  error('not coded yet');
        summing = 'continuous';
    else
        summing = 'linear';
        HP = log(variance);
    end
    
    if doSplit
        summing = {summing, 'split'}; % split kernels along 2nd dimension
    end
    
    scale = {dt*(kernel(1)+(0:nK-1))};
else
    HP = log(variance);
    DM = EventCount;
    summing = 'linear';
    scale = [];
end


R = regressor(DM, 'linear','scale',scale,'sum',summing, 'hyperparameter',HP,'label',label);

if doSplit && ~isempty(split_label)
    if length(split_label)~=nSplit
        error('number of split values (%d) does not match number of split labels (%d)',  nSplit, length(split_label));
    end
    R.plot{1} = {{{}, split_label}};
end
