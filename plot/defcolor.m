function colaz = defcolor(clas)
% default colours
% adapted from https://colorbrewer2.org
if nargin==1 && strcmp(clas, 'char')
    colaz={'b' 'r' 'g' 'k' 'y' 'c' 'm'};
else

    colaz = [ 70, 70, 70; % dark gray
        55,126,184; % blue
        228,26,28; % red
        77,175,74; % green
        152,78,163; % purple
        255,127,0; % orange
        166,86,40; % brown
        247,129,191;
        255,255,51]/255;

    %          [   .3     .3     .3  ; ...   %black
    %         1      0     0  ; ... %red
    %         0     0     1  ; ...  % blue
    %         0    .5     0  ; ...
    %         0.75   .75     0  ; ...
    %         0.75     0   .75  ; ...
    %         0   .75   .75  ; ...
    %         0.5   .5   .5  ; ... % grey
    %      ];

    colaz = num2cell(colaz,2);
end