clc, clear all, close all;
%%
% Load the person_feat variable saved .mat file after running the USLEET code to concatenate
% each person footstep into 1 single variable

load(['P1_P5_STD_KUR_MSE_Q1_person_feat_2.mat'])



footstep_feat = [];
labels=[];

for i = 1:5
    i
    footstep_feat = [footstep_feat ; person_feat{i}];
    labels = [labels; i*ones(size(person_feat{i},1),1)];
    
    
end

footstep_feat = [footstep_feat, labels];

%%
% Now save the footstep_feat variable as .mat file
