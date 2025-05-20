% Convert the cell array to a table for better handling and export
clear all;close all;clc

allUCBCasesResults = table();
sleepStatus ='Sleep';
phaseOfStudy = 'Pre';
subjectStatus = 'case';


load('Sleep1_Pre_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

load('Sleep2_Pre_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

phaseOfStudy = 'Post';
load('Sleep1_Post_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

load('Sleep2_Post_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames


sleepStatus ='Awake';
phaseOfStudy = 'Pre';
load('Wake1_Pre_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

load('Wake2_Pre_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

phaseOfStudy = 'Post';
load('Wake1_Post_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames

load('Wake2_Post_results.mat')
[allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,fileNames,sleepStatus,phaseOfStudy,subjectStatus);
clear eegComputationalMetrics fileNames




% Save the table to a CSV file
writetable(allUCBCasesResults, 'allPatinetsMetrics_Cz.csv');






function [allUCBCasesResults ] = Get_CSV_table(allUCBCasesResults,eegComputationalMetrics,subjectFileNames,sleepStatus,phaseOfStudy,subjectStatus)

numElectrods = size(eegComputationalMetrics{1, 1}.electrodes,2);

for i  = 1: length(eegComputationalMetrics)

    dataLength = size(eegComputationalMetrics{i,1}.amplitude,1);

    fileName = subjectFileNames{i};
    patient_ID = Get_Patient_ID (fileName);
    patient_ID  = repmat({patient_ID},dataLength,1);
    file_name = repmat({fileName},dataLength,1);
    sleep_or_awake = repmat({sleepStatus},dataLength,1);
    pre_or_post = repmat({phaseOfStudy},dataLength,1);
    case_or_control = repmat({subjectStatus},dataLength,1);
    
    

    for j = 1:numElectrods
        electrod = repmat(eegComputationalMetrics{1, 1}.electrodes(j),dataLength,1);
    
        currentTable = table(patient_ID,file_name,sleep_or_awake,pre_or_post,case_or_control,electrod, ...
        eegComputationalMetrics{i, 1}.epochTimes(:,1),...
        eegComputationalMetrics{i, 1}.epochTimes(:,2),...
        eegComputationalMetrics{i, 1}.shanEntDelta(:,j), ...
        eegComputationalMetrics{i, 1}.permEntBeta(:,j),...
        eegComputationalMetrics{i, 1}.permEntTheta(:,j), ...
        eegComputationalMetrics{i, 1}.shanEntTheta(:,j),...
        eegComputationalMetrics{i, 1}.permEntAlpha(:,j), ...
        eegComputationalMetrics{i, 1}.shanEntAlpha(:,j),...
        eegComputationalMetrics{i, 1}.permEntBeta(:,j), ...
        eegComputationalMetrics{i, 1}.shanEntBeta(:,j),...
        eegComputationalMetrics{i, 1}.deltaPSD(:,j), ...
        eegComputationalMetrics{i, 1}.thetaPSD(:,j),...
        eegComputationalMetrics{i, 1}.alphaPSD(:,j), ...
        eegComputationalMetrics{i, 1}.betaPSD(:,j),...
        eegComputationalMetrics{i, 1}.broadPSD(:,j), ...
        eegComputationalMetrics{i, 1}.amplitude(:,j), ...
        eegComputationalMetrics{i, 1}.SEF(:,j),...,
        'VariableNames',{'patient_ID','file_name','sleep_or_awake','pre_or_post','case_or_control','electrod_channel' ...
        'epoch_start_index','epoch_stop_index'...
        'permutation_entropy_delta', ...
        'shannon_entropy_delta',...
        'permutation_entropy_theta', ...
        'shannon_entropy_theta',...
        'permutation_entropy_alpha', ...
        'shannon_entropy_alpha',...
        'permutation_entropy_beta', ...
        'shannon_entropy_beta',...
        'power_spectral_density_delta', ...
        'power_spectral_density_theta',...
        'power_spectral_density_alpha', ...
        'power_spectral_density_beta',...
        'power_spectral_density_broadband', ...
        'amplitude_range', ...
        'spectral_edge_frequency'} );


    allUCBCasesResults = [allUCBCasesResults ; currentTable];
    end
end


end

function  PatientID = Get_Patient_ID (file_names)

% ucbID = 'M145.g' - '0';
% ucbID(ucbID < 0 | ucbID > 9) = [];

ucbID = file_names(1:3); % getting the only first 3 charachters, because we onlyhave up to 3 digits
patientNum = num2str(sscanf(ucbID,'%i')); %extracting the patient Id number
numZeros = 3- length(patientNum);
myzeros = num2str(zeros(1,numZeros));
myzeros = myzeros(~isspace(myzeros)); %removing spaces between zeros
PatientID = ['UCB' myzeros patientNum];


end