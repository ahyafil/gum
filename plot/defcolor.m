function colaz=defcolor(clas)
%colaz = defcolor; set colaz to {'b' 'r' 'g' 'k' 'y' 'c' 'm'};

if nargin==1 && strcmp(clas, 'char')
    colaz={'b' 'r' 'g' 'k' 'y' 'c' 'm'};
else
    
    colaz = [ ...
        0     0     0  ; ...   %black
        1      0     0  ; ... %red
        0     0     1  ; ...  % blue
        0    .5     0  ; ...
        0.75   .75     0  ; ...
        0.75     0   .75  ; ...
        0   .75   .75  ; ...
        0.25   .25   .25  ; ... % grey
        1     0     0  ; ...
        0    .5     0  ; ...
        0     0     1 ];
    
    colaz = num2cell(colaz,2);
    
    
end