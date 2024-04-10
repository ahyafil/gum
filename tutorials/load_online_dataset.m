function T = load_online_dataset()
% load the dataset of monkey choices in motion integration perceptual task
% (Yates et al., Nat Neuro 2017; Hyafil et al., eLife 2023)

fprintf('loading csv file from url...')
csv_path = 'https://raw.githubusercontent.com/ahyafil/TemporalIntegration/main/data/monkey.csv';
T = readtable(csv_path);

% identify how many samples in stimuli (i.e. how many columns named
% 'stimulus_i')
nSample = 1;
while ismember("stimulus_"+nSample,T.Properties.VariableNames)
        T.("Stimulus"+nSample) = T.("stimulus_"+nSample); % we copy it because we want to show how to use both as splitted and merged variables
    nSample = nSample+1;
end
nSample = nSample-1;

% merge columns for stimulus into single variable (one row vector for each
% trial)
T = mergevars(T,"stimulus_"+(1:nSample),'NewVariableName',"Stimulus");

% only monkey P
T= T(strcmp(T.subject,'N'),:);
T  = removevars(T,"subject");

% add accuracy
accuracy = T.resp == T.target;
accuracy_label = ["error";"correct"];
T.accuracy = accuracy_label(accuracy+1);

% add signed response (for lagged regressor)
T.response = sign(T.resp-0.5); % -1 or +1

fprintf('done\n');
end