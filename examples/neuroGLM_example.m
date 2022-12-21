clear; close all;

baseline_rate = 20; % baseline firing rate (Hz)
visual_kernel_left = @(t) 2*exp(-t/.3).*(1-exp(-t/.03));
visual_kernel_right = @(t) 4*exp(-t/.1).*(1-exp(-t/.03));
auditory_kernel = @(t, s) 1.5*s.*t.*(t<.5); % ramp until one second, parametrized by stimulus
self_kernel = @(t) -1*exp(-t/.02); % self inhibition with 20 ms time scale

figure;
subplot(131); hold on;
tt = 0:.001:.6;
plot(tt, visual_kernel_left(tt));
plot(tt, visual_kernel_right(tt));
legend({'left stim','right stim'});
xlabel('time after visual stim')
title('visual kernel');

subplot(132); 
tt = (0:.001:1.2)';
plot(tt, [auditory_kernel(tt,-1) auditory_kernel(tt,1) auditory_kernel(tt,2)]);
legend({'s=-1','s=1','s=2'});
xlabel('time after acoustic stim')
title('acoustic kernel');

subplot(133); 
tt = 0:.001:.1;
plot(tt, self_kernel(tt))
xlabel('time after spike')
title('self kernel');


%% simulate

nTrial = 100; % number of trials
ITI_range = [.8 1.2]; % range of interstimulus interval

% generate times and value of stimuli
ITI = ITI_range(1)+rand(1,nTrial-1)*diff(ITI_range); % inter-trial interval
tTrial = cumsum([0 ITI]); % timing of trials
%isVisual = boolean(randi([0 1],1,nTrial)); % whether a visual stim is shown for given stimulus
isVisual = true(1,nTrial); % whether a visual stim is shown for given stimulus

VisualSide = 2*randi([0 1],1,nTrial)-1; % whether stimulus is shown to the left (-1) or right (1)
%isAuditory = boolean(randi([0 1],1,nTrial)); % acoustim stim always shown
isAuditory = true(1,nTrial); % acoustim stim always shown

sAuditory = randn(1,nTrial); % value of acoustisc stim (i.e. frequency w.r.t. ref tone)
tVisual = tTrial(isVisual);
tVisualLeft = tTrial(isVisual & VisualSide==-1);
tVisualRight = tTrial(isVisual & VisualSide==1);
VisualSideShown = VisualSide(isVisual); % side of stim actually shown

tAuditory = tTrial(isAuditory);

T = tTrial(end)+1; % overall duration
dt = .001; % time step for simulation
TT = round(T/dt);


tSpk = [];
for i=1:TT
    t=i*dt; % current time
    tFromVisualLeft = t-tVisualLeft(tVisualLeft<t); % time elapsed from previous visual stim to the left
    xVisLeft = sum(visual_kernel_left(tFromVisualLeft)); % 
    
        tFromVisualRight = t-tVisualRight(tVisualRight<t); % time elapsed from previous visual stim to the left
    xVisRight = sum(visual_kernel_right(tFromVisualRight)); % 
       
        tFromAuditory= t-tAuditory(tAuditory<t); % time elapsed from previous auditory stim
        xAuditory = sAuditory(tAuditory<t);
        xAud = sum(auditory_kernel(tFromAuditory,xAuditory));
        
        tFromSpike = t-tSpk;
        xSpk = sum(self_kernel(tFromSpike));
        
        lambda = baseline_rate*exp(xAud + xVisLeft + xVisRight + xSpk); % Poisson rate
        nSpk = poissrnd(dt*lambda); % generate spike accordingly
        
        if nSpk>0
           tSpk(end+1:end+nSpk) = t*ones(1,nSpk); % append spike times 
        end
end


%% plot rasterplot
xl = [0 T]; [0 15];
figure; subplot(311); 
rasterplot({tSpk});
text( mean(xl), 1.2, 'spikes');
xlim(xl); axis off;

subplot(312);
VisVec = zeros(1,TT);
VisVec(1+round(tVisualLeft/dt)) = 1;
plot(dt*(0:TT-1),VisVec);
xlim(xl); axis off;
text(mean(xl), 1.2, 'visual stim');

subplot(313); 
AudVec = zeros(1,TT);
AudVec(1+round(tAuditory/dt)) = sAuditory;
plot(dt*(0:TT-1),AudVec);
xlim(xl); axis off;
text(mean(xl), 1.2, 'auditory stim');

%% build regressors
dt = .01; % 10 ms bins
kerneltype ='gaussian'; 'independent';
TTm = ceil(T/dt);

boundVisual = [0 .5]; % we fit visual response in first 500 ms
timescaleVisual = .05; % time scale of visual kernel (hyperparameter)
durationAuditory = .5; % duration of ramp we fit
boundSpike = [dt .1]; % we fit spiking history effect in first 200 ms ( DO NOT INCLUDE 0,otherwise the spike is predicting itself)
timescaleSpike = .01; % time scale of spike kernel (hyperparameter)

%Rvis = timeregressor(tVisual, dt, [0 T], 'kernel',boundVisual,'timescale',timescaleVisual);
%Rvis = timeregressor(tVisual, dt, [0 T], 'kernel',boundVisual,'kerneltype',kerneltype,...
%    'name','visual kernel');
Rvis = timeregressor(tVisual, dt, [0 T], 'kernel',boundVisual,'kerneltype',kerneltype,'split',VisualSideShown,...
    'split_label', {'left','right'},'label','visual kernel');

Raud = timeregressor(tAuditory, dt, [0 T], 'modulation', sAuditory,'duration',durationAuditory, 'ramp','label','auditory kernel');
%Raud.plot{1} = {'color',[.4 .3 .9]}; % define color options (wu arguments)

Rspk = timeregressor(tSpk, dt, [0 T], 'kernel',boundSpike, 'kerneltype',kerneltype, 'label', 'history kernel');
Roffset = timeregressor('constant',dt,[0 T]);
Rdt = timeregressor('dt',dt,[0 T]);


% generate sum of regressors (will actually create an array of regressors)
R = [Rvis Raud Rspk Roffset Rdt];

% turn into binary vector
binSpk = zeros(TTm,1);
for i=1:TTm
    binSpk(i) = sum( (tSpk>=(i-1)*dt) & (tSpk<i*dt));
end

%% build GUM
param = struct;
param.observations = 'count';
param.constant = 'off'; % no offset parameter (we already have coded it)
M = gum(R, binSpk, param);

%% plot design matrix for first 10 seconds
figure;
M.plot_design_matrix();

%% fit GLM
param = struct;
M = infer(M,param);

%% plot GLM results
subfigure;
M.plot_weights();

%% plot spike count vs predicted spike count
subfigure;
M.plot_data_vs_predictor;
ylabel('spike count');

%% simulate model
[M, samp] = Sample(M);


M = M.clear_data;
