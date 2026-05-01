clc, clear , close all

% load('PersonEvent.mat')
load('Wooden_8prsn_feat_db1_denoise.mat')

%%

footstep_feat = []
for i = 1:size(person_feat,2)
% person_feat{i} = person(i).feat;

footstep_feat =  [footstep_feat ; person_feat{i}];
% footstep_feat =  [footstep_feat ; person{i}feat2];
end



label = [];
for i = 1:size(person_feat,2)
    
    label = [label ; i*ones(size(person_feat{i},1),1)];

    
end

footstep_feat = [footstep_feat , label];


%%

save('1_footsteps_feat_wood_db1.mat', 'footstep_feat');

%%

no_of_footstep = [2 3 5 7 10];


for x = 1:length(no_of_footstep)

load('1_footsteps_feat_SF_8prsn_500_10%.mat')

    
new_feat = [];
m =no_of_footstep(x);

fprintf('\n================= Creating features for %d footsteps =================\n',m )

for k = 1:8; %size(person_feat,2)
feat = [];
ind  = find(footstep_feat(:,end)==k);
id = inf;
j=ind(1);
i=1;

while (id >= m)
    
    start = j;
    stop = start + m-1;
    
    feat(i,:) = mean(footstep_feat(start:stop,:));

    j = stop +1;
    i=i+1;
    id = ind(end) - stop;
    
end

new_feat = [new_feat ; feat];

end

one_footstep_feat = footstep_feat;
footstep_feat = [];
footstep_feat = new_feat;
no_of_feat = size(footstep_feat,2)-1;
filename = sprintf('%d_footsteps_feat_SF_8prsn_500_10%.mat', m);


fprintf('------ Saving file %s ------\n' , filename)
save(filename,'footstep_feat');

end





