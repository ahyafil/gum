% EXAMPLES OF HOW TO USE LAGGED REGRESSOR IN GUM

%% first simulate a simple Reinforcement Learning model in a very simple environement (no stimulus, two possible actions)
clear; close all;
SessionLength = 1000; % length of each session
nSession = 10; % number of sessions

p_reversal = .05; % probability of reversal of the rewarded action
beta = .2; % learning rate
eta = .3; % decision noise

T = [];
for i=1:nSession % for each session

    rew = sign(randn); % side of rewarded action (-1/1)
    V = .5*[1 1]; % value for actions

    Response = zeros(SessionLength,1); % response on each trial
    Outcome = zeros(SessionLength,1); % outcome on each trial
    for t=1:SessionLength

        % selection action
        DeltaV = diff(V); % difference of value
        Response(t) = sign(DeltaV + eta*randn); % response

        % outcome
        Outcome(t) = (Response(t) ==rew); % whether it's rewarded

        % update option values (Rescorla-Wagner rule)
        i_resp = 1 +(Response(t)>0); % map -1/1 onto 1/2
        V(i_resp) = V(i_resp) + beta*(Outcome(t)-V(i_resp));

        if rand<p_reversal
            rew = -rew; % reversal of rewarded side
        end


    end

    % code response as 0/1 (required for binomial model)
    ResponseBinary = Response==1;

    % session index
    Session = i*ones(SessionLength,1);

    % create session table
    Ts = table(Response, ResponseBinary,Outcome, Session);

    % append table
    T = cat(1, T, Ts);
end


%% NOW ESTIMATE CHOICE KERNELS

% option: binomial GUM, do not include intercept
options = struct('observations','binomial','constant','off');

% start with regression where we only capture the impact of the
% previous choice
M(1) = gum(T, 'ResponseBinary ~ lag(Response)', options);
M(1) = M(1).infer;
figure;
M(1).plot_weights;

% now instead of looking only at the last trial, we look back further in
% time at last 10 trials
M(2) = gum(T, 'ResponseBinary ~ lag(Response;Lags=1:10)', options);
M(2) = M(2).infer;
figure;
M(2).plot_weights;

% because the updating rule depends on the outcome, we actually want to look
% at the impact of last trials CONDITIONED on the outcome (i.e. capture separately the impact of previous
% choices if rewarded and of previous choices if not rewarded)
M(3) = gum(T, 'ResponseBinary ~ lag(Response|Outcome;Lags=1:10)', options);
M(3) = M(3).infer;
figure;
M(3).plot_weights;

% display design matrix
figure;
M(3).plot_design_matrix(1:200);


% now we'll use a set of two exponential functions as basis functions for
% the kernels
M(4) = gum(T, 'ResponseBinary ~ lag(Response|Outcome; Lags=1:10; basis=exp2)', options);
M(4) = M(4).fit;
figure;
M(4).plot_weights;

%% actually we are collapsing data over different sessions. Because the side
% values are reset at each session, the last trials in a session will not
% influence the first trials of the next session. We add the 'group' option
% so that the regressors do not 'spill over' one session to the next;
M(5) = gum(T, 'ResponseBinary ~ lag(Response|Outcome; Lags=1:10; group=Session)', options);
M(5) = M(5).infer;
figure;
M(5).plot_weights;

% let's check that in the Design matrix zooming at the end of session 1 -
% start of session 2
figure;
M(5).plot_design_matrix(SessionLength + (-50:50));
hold on; 
plot(xlim, (50+1.5)*[1 1],'r'); % the red line marks the session change

%% add split
M(6) = gum(T, 'ResponseBinary ~ lag(Response|Outcome; Lags=1:10; split=false)', options);
M(6) = M(6).infer;
figure;
M(6).plot_weights;


