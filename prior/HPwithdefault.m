function   HP = HPwithdefault(HPspecified, HPdefault)
% HP = HPwithdefault(HPspecified, HPdefault)
% combine default and specified HP values

HP = HPdefault;
HP(~isnan(HPspecified)) = HPspecified(~isnan(HPspecified)); % use specified values
end