clc; clear all; close all;

% Ask user to select the root folder that contains P1–P100, or P1-P30, P1-P40 subfolders
rootFolder = uigetdir(pwd, 'Select the root folder containing P1–P5 subfolders');
if rootFolder == 0
    error('No folder selected. Exiting.');
end

% List of person folders
persons = {'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10', 
           'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20', 
           'P21','P22','P23','P24','P25','P26','P27','P28','P29','P30', 
           'P31','P32','P33','P34','P35','P36','P37','P38','P39','P40', 
           'P41','P42','P43','P44','P45','P46','P47','P48','P49','P50', 
           'P51','P52','P53','P54','P55','P56','P57','P58','P59','P60', 
           'P61','P62','P63','P64','P65','P66','P67','P68','P69','P70', 
           'P71','P72','P73','P74','P75','P76','P77','P78','P79','P80', 
           'P81','P82','P83','P84','P85','P86','P87','P88','P89','P90', 
           'P91','P92','P93','P94','P95','P96','P97','P98','P99','P100'};

dataset_original = [];

for j = 1:length(persons)
    personName = persons{j};
    personFolder = fullfile(rootFolder, personName);
    
    if ~isfolder(personFolder)
        warning('Folder not found: %s', personFolder);
        continue;
    end

    data = [];

    for i = 1:4
        filename = sprintf('%s_%i.mat', personName, i);
        filepath = fullfile(personFolder, filename);

        if exist(filepath, 'file')
            load(filepath);  % loads geo_data
            tempdata = geo_data;
            data = [data; tempdata];
        else
            warning('File not found: %s', filepath);
        end
    end

    dataset_original = [dataset_original, data];

    % Save combined data in root folder, not person folder
    geo_data = data;
    dataname = sprintf('%s_all.mat', personName);
    save(fullfile(rootFolder, dataname), 'geo_data');
end
