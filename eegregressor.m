function EEG = eegregressor(EEG, cfg)
% R = eegregressor(EEG, cfg)
% cfg is a structure with fields:
% -'eventtypes' (mandatory): cell array for different events
% -'formula': a formula string or cell array of formulas. Formulas are
% similar to unfold formulas (i.e R-type) with extra capabilities.
%  Typical formula is 'y~1+var1+var2+...' for linear encoding of variables
%  'var1','var2', etc. (the names of the variables should correspond to
%  fied names in the event structure of EEG). Use cat(varX) for separate
%  regressors depending on value of variable varX. (Note: 'y~' at the
%  beginning of formula is option)
%  Use brackets as in 'y~1+[var1+var2]+var3+... to enforce similar temporal
%  response to all variable in brackets (bilinear model, more generally a
%  GUM).
%
% % - 'window' (mandatory)
% - 'code': 'gaussian', 'independent' (default), 'shiftedcosine'
% - 'variance': hyperparameter for L2 regularization (default: Inf, i.e. no
% regularization)

assert(isstruct(cfg), 'cfg must be a structure');
assert(isfield(cfg, 'eventtypes'), 'cfg must have field ''eventtypes''');
assert(isfield(cfg, 'window'), 'cfg must have field ''window''');
if ~iscell(cfg.eventtypes)
    cfg.eventtypes = {cfg.eventtypes};
end
if ~iscell(cfg.window)
    cfg.window = {cfg.window};
end
nEvent = length(cfg.eventtypes); % number of events

if ~isfield(cfg, 'formula')
    cfg.formula = cell(1,nEvent);
end
if ~isfield(cfg, 'code')
    cfg.code = cell(1,nEvent);
elseif ~iscell(cfg.code)
    cfg.code = repmat({cfg.code}, 1, nEvent);
end

if ~isfield(cfg, 'variance')
    cfg.variance = inf(1,nEvent);
elseif isscalar(cfg.variance)
    cfg.variance = cfg.variance*ones(1,nEvent);
end
%if ~isfield(cfg, 'split')
%    cfg.split = cell(1,nEvent);
%end
dt = 1/EEG.srate; % time step

Tbnd = [EEG.times(1) EEG.times(end)]; % time boundaries

dt_error = diff(EEG.times)*EEG.srate - 1;
assert(all(abs(dt_error)<1e-3), 'irregular time steps');

event_types = {EEG.event.type}; % types for each event

r = 1;
for e=1:nEvent % for each event

    this_event =  [EEG.event(strcmp(event_types, cfg.eventtypes{e}))]; % corresponding events


    EventTime = [this_event.sample]; % corresponding timing
    kernel = cfg.window{e}; % corresponding window
    code = cfg.code{e};
    if isempty(code)
        code = 'independent';
    end

    % parse formula
    [M,splt] = parse_fmla(this_event, cfg.formula{e});
    nReg = max(size(M,2),1); % number of regressors

    % define regressor for each modulator
    for rr=1:nReg
        EEG.regressor(r) = timeregressor2(EventTime/EEG.srate, dt, Tbnd,'variance', cfg.variance(e),...
            'split',splt{rr}, 'kernel', kernel, 'kerneltype', code, 'modulation', M{rr});
        r = r+1; % update regressor counter
    end
end
end

%% parse formula for regressors
function [M, splt] = parse_fmla(event, fmla)

if isempty(fmla)
    M = {};
    return;
end

assert(ischar(fmla), 'formula should be a character array or be empty');

% parse formula into regressor names and splitting variable
fmla = strrep(fmla,' ',''); % remove empty space
if strcmp(fmla(1:2), 'y~')
    fmla(1:2) = [];
end

% look for brackets signalling similar dynamicss (GUM) or + signs
cnt = 1;
subfmla = {}; % either independent regressors or GUM formulas
is_gum = [];
while ~isempty(fmla)
    if fmla(1) == '[' % opening bracket: GUM formula
        next_bk = find(fmla==']'); % next bracket sign
        
        subfmla{cnt} = fmla(2:next_bk-1); % corresponding part
        is_gum(cnt) = true;
        fmla(1:next_bk) = []; % trim left part of formula
    else % independent variable
        next_plus = find(fmla=='+'); % next plus sign
        if isempty(next_plus) % end of equation
            next_plus = length(fmla)+1;
        end

        subfmla{cnt} = fmla(1:next_plus-1);
        is_gum(cnt) = false;
        fmla(1:next_plus-1) = []; % trim left part of formula

    end

    if ~isempty(fmla)
        assert(fmla(1)=='+', 'expected a plus sign');
        fmla(1)= [];
    end

    cnt = cnt +1;

end

nReg = cnt-1; % total number of regressors


% if any(fmla=='|')
%     splt_idx = find(fmla=='|',1);
%     split_var = fmla(split_idx+1:end); % splitting variable
%     if ~isfield(event, split_var)
%         error('unrecognized splitting variable (not correspond field in event structure):%s', split_var);
%     end
%     splt = [event.(split_var)]; % extract value
%     fmla(splt_idx:end) = []; % trim end of formula to keep only regressors
% else
%     splt = [];
% end
%
% plus_signs = [0 find(fmla=='+') length(fmla)+1]; % index of plus signs
% nReg = length(plus_signs)-1;
% regnames = cell(1,nReg);
% for r=1:nReg
%     regnames{r} = fmla(plus_signs(r)+1:plus_signs(r+1)-1); % regressor name corresponds to the string between the + signs
% end


% deal with offset
no_offset = any(strcmp(subfmla, '-1'));
if no_offset
    subfmla(strcmp(subfmla, '-1')) = []; % remove from list
    nReg = nReg-1;
elseif ~any(strcmp(subfmla, '1')) % add offset
    subfmla{end+1} = '1';
    is_gum(end+1) = false;
    nReg = nReg+1;
end

%splt = repmat({splt}, 1, nReg); % one splitting variable per regressor
splt = cell(1,nReg); % one splitting variable per regressor

% get value for each regressor
M = cell(1,nReg); %nan(length(event), nReg);
for r=1:nReg
    regn = subfmla{r};
    if is_gum(r) % GUM formula

        % convert event structure to table
        Tbl = struct2table(event);
        Tbl.dmyvar = ones(length(event),1); % create dummy dependent variable
        Tbl.dmyvar(1) = 0; % just to avoid a warning signal later
        dummyM = gum(Tbl,['dmyvar~' regn]); % create GUM with regressors defined by formula (and dummy dep variable)
        M{r} = dummyM.regressor; % extract regressor

    elseif strcmp(regn,'1') % offset
        M{r} = ones(length(event),1);
    elseif length(regn)>4 && strcmp(regn(1:4),'cat(') % cat(x) is the same as 1|x
        split_var = regn(5:end-1);
        if ~isfield(event, split_var)
            error('unrecognized categorical variable (not correspond field in event structure):%s', split_var);
        end
        if ischar(event(1).(split_var))
            splt{r} = {event.(split_var)};
        else
            splt{r} = [event.(split_var)];
        end
        M{r}= ones(length(event),1);

    else % regressor name

        if regn(1) == '(' && regn(end)==')' % ignore parenthesis
            regn = regn(2:end-1);
        end
        if ~isfield(event, regn)
            error('unrecognized regressor name (not correspond field in event structure):%s', regn);
        end
        M{r} = [event.(regn)]'; % extract value from corresponding firled

    end
end

end




