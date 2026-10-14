clc;
clear all;


% Select the nine raw pullout exports; generated result tables are excluded.
files = dir('*TW_t*_*.csv');

% Initialize a structure to store data
dataStruct = struct();

for k = 1:length(files)
    filename = files(k).name;
    
    % Open the file to check where the data starts
    fid = fopen(filename, 'r');
    dataStartLine = 0;
    line = fgetl(fid);
    
    % Read file line by line to find the data start point
    while ischar(line)
        dataStartLine = dataStartLine + 1;
        if contains(string(line), "Reading") % Checking for the first keyword in the header
            break;
        end
        line = fgetl(fid); % Read next line
    end
    fclose(fid);
    
    % Read the data starting from the found header line
    opts = detectImportOptions(filename, 'NumHeaderLines', dataStartLine);
    
    % Ensure the column names are set correctly
    opts.VariableNames = ["Reading", "Load_N", "Distance_mm", "Time_s"];
    
    % Read table and store it in the structure
    dataTable = readtable(filename, opts);
    
    % Store data with filename (without extension) as field name
    [~, name, ~] = fileparts(filename);
    dataStruct.(matlab.lang.makeValidName(name)) = dataTable;
    
    fprintf('Loaded: %s\n', filename);
end

disp('All files loaded successfully.');


m=50*10^-6;   %Reference tangle mass: 50 milligrams.
g=9.8;      %acceleration due to gravity in m/s
%%

% Get all field names (which correspond to file names)
fileNames = fieldnames(dataStruct);

% Plot one representative force-displacement trace per condition.
figure('Name','Representative pullout force-displacement','Color','w','Position',[100 100 1100 440]);
for k = [1 7 4]%[1 4 7]%1:length(fileNames)
    % Extract the current dataset
    fileData = dataStruct.(fileNames{k});
    x = fileData.Distance_mm;
y = abs(smooth(fileData.Load_N,11));   %Force magnitude in newtons.


    %hold on;
    % Create a new figure for each file
    hold on;
    if (k<=3)
        subplot(1,3,1)
        plotCondition = 0;
    plot(x(1:end-1), y(1:end-1), 'b-', 'LineWidth', 1.5);
    ylim([0 35]*m*g)
    else 
        if (k<=6)
            subplot(1,3,3)
            plotCondition = 10;
        plot(x(1:end-1), y(1:end-1), 'k-', 'LineWidth', 1.5);
    ylim([0 35]*m*g)
        else 
            subplot(1,3,2)
            plotCondition = 2;
            plot(x(1:end-1), y(1:end-1), 'r-', 'LineWidth', 1.5);
        ylim([0 35]*m*g)
        end
    end
    
    
    % Add labels and title
    xlabel('Distance (mm)', 'FontSize', 12);
    ylabel('Load (N)', 'FontSize', 12);
    title(sprintf('%d TW',plotCondition), 'FontSize', 14);
    
    % Grid for better visibility
    grid on;
    
    % Show the plot
    fprintf('Plotted: %s\n', fileNames{k});
end

disp('All plots generated successfully.');


%%



% Get all field names (which correspond to file names)
fileNames = fieldnames(dataStruct);

% Define Sampling Parameters
fs = 100;  % Sampling frequency in Hz
nyquist = fs / 2; % Nyquist frequency
f_low = 1 / nyquist; % Normalize the low cutoff frequency
%f_high = 49.99 / nyquist; % Normalize the high cutoff frequency

% Fourth-order low-pass filter with a 1 Hz cutoff.
[b, a] = butter(4, f_low, 'low');

stps=[34417 22642 25528 26584 26189 29393 27580 2737 25818 ]

pks0tw=[];
pks10tw=[];
pks2tw=[];
mx0tw=[];
mx10tw=[];
mx2tw=[];
% Loop through each dataset, apply filter, and plot
for k = 1:length(fileNames)
    % Extract the current dataset
    fileData = dataStruct.(fileNames{k});
    
    % Apply the low-pass filter to the load data
    filteredLoad = filtfilt(b, a, fileData.Load_N/(m*g));
    distance = filtfilt(b, a, fileData.Distance_mm);
    filteredLoad = filteredLoad(2:stps(k));
    distance = distance(2:stps(k));
    [mx1,loc1] = findpeaks(abs(filteredLoad),'MinPeakDistance',round(0.1*fs),'MinPeakProminence',0.0001);

    pks{k} = mx1;
    if (k<=3)
        pks0tw=[pks0tw; mx1];
        mx0tw=[mx0tw; max(abs(filteredLoad))];
    else 
        if (k<=6)
            pks10tw=[pks10tw; mx1];
            mx10tw=[mx10tw; max(abs(filteredLoad))];
        else 
            pks2tw=[pks2tw; mx1];
            mx2tw=[mx2tw; max(abs(filteredLoad))];
        end
    end

    % Create a new figure for each file
%     figure(k);
%     hold on
%     plot(distance,abs(filteredLoad), 'b-', 'LineWidth', 1.5);
%     plot(distance(loc1),abs(filteredLoad(loc1)),'ro');
%     % Add labels and title
%     xlabel('Distance (mm)', 'FontSize', 12);
%     ylabel('Load (N) [Filtered]', 'FontSize', 12);
%     title(sprintf('%d TW',plotCondition), 'FontSize', 14);
%     
%     % Grid for better visibility
%     grid on;
    
    % Show the plot
    fprintf('Filtered and plotted: %s\n', fileNames{k});
end

disp('All filtered plots generated successfully.');


%%
figure('Name','All loading peaks (diagnostic)');
violinplot([pks0tw; pks2tw; pks10tw],...
    [ones(size(pks0tw)); 2*ones(size(pks2tw)); 3*ones(size(pks10tw))]);
figure('Name','Maximum force per tangle (normalized)');
violinplot([mx0tw; mx2tw; mx10tw],...
    [ones(size(mx0tw)); 2*ones(size(mx2tw)); 3*ones(size(mx10tw))]);


 numplt=3;
 colrs=lines(numplt);
 figure;
 axes('NextPlot','replacechildren', 'ColorOrder',colrs);
 boxchart([ones(size(pks0tw)); 2*ones(size(pks2tw)); 3*ones(size(pks10tw))],...
    [pks0tw; pks2tw; pks10tw],'WhiskerLineColor','k');
xticks([1 2 3]); xticklabels({'0','2','10'});
xlabel('N_{TW}'); ylabel('Loading peak force / (mg)');
title('Within-trial loading peaks (diagnostic)');



 numplt=3;
 colrs=lines(numplt);
 figure;
 axes('NextPlot','replacechildren', 'ColorOrder',colrs);
 boxchart([ones(size(mx0tw)); 2*ones(size(mx2tw)); 3*ones(size(mx10tw))],...
    [mx0tw; mx2tw; mx10tw],'WhiskerLineColor','k');
xticks([1 2 3]); xticklabels({'0','2','10'});
xlabel('N_{TW}'); ylabel('Maximum force / (mg)');
title('Maximum force per independently prepared tangle');


%%
clc
% Create grouping variables
condition = categorical([repmat("0TW", 3, 1); repmat("2TW", 3, 1); repmat("10TW", 3, 1)]);
repeat = categorical([1;2;3; 1;2;3; 1;2;3]);  % Each experimental repeat

% Sample mean peak values per repeat (instead of pooling peaks)
%meanPeaks = [mean(pks{1}), mean(pks{2}), mean(pks{3}), ...
 %            mean(pks{4}), mean(pks{5}), mean(pks{6}), ...
  %           mean(pks{7}), mean(pks{8}), mean(pks{9})]';

maxPeaks = [mx0tw(1), mx0tw(2), mx0tw(3), ...
             mx2tw(1), mx2tw(2), mx2tw(3), ...
             mx10tw(1), mx10tw(2), mx10tw(3)]';


% Fit a mixed-effects model
lme = fitlme(table(condition, repeat, maxPeaks), ...
    'maxPeaks ~ condition + (1|repeat)');

% Display results
disp(lme);


%%

clc;

% Define data for each condition (3 repeats per condition)
data_0TW = [mx0tw(1), mx0tw(2), mx0tw(3)];
data_2TW = [mx2tw(1), mx2tw(2), mx2tw(3)];
data_10TW = [mx10tw(1), mx10tw(2), mx10tw(3)];

% Perform Mann-Whitney U test (Wilcoxon Rank-Sum test)
[p12, ~] = ranksum(data_0TW, data_2TW); % 0TW vs 2TW
[p13, ~] = ranksum(data_0TW, data_10TW); % 0TW vs 10TW
[p23, ~] = ranksum(data_2TW, data_10TW); % 2TW vs 10TW

% Store p-values
p_values = [p12, p13, p23];

% Number of comparisons
num_tests = length(p_values);

% Bonferroni Correction (Divide alpha by number of tests)
alpha = 0.05;
bonferroni_threshold = alpha / num_tests;

% Holm-Bonferroni Correction (Stronger than Bonferroni)
[sorted_p, sorted_idx] = sort(p_values);  % Sort p-values
holm_thresholds = alpha ./ (num_tests:-1:1); % Compute Holm-Bonferroni thresholds

% Display results
disp('Pairwise Mann-Whitney U test results:');
disp(['p-value for 0TW vs 2TW: ', num2str(p12)]);
disp(['p-value for 0TW vs 10TW: ', num2str(p13)]);
disp(['p-value for 2TW vs 10TW: ', num2str(p23)]);

% Applying Bonferroni Correction
significant_bonferroni = p_values < bonferroni_threshold;
disp(['Bonferroni significant results (alpha = ', num2str(bonferroni_threshold), '):']);
disp(significant_bonferroni);

% Applying Holm-Bonferroni Correction
significant_holm = false(size(p_values));
for i = 1:num_tests
    if sorted_p(i) < holm_thresholds(i)
        significant_holm(sorted_idx(i)) = true;
    else
        break; % If one test is not significant, all larger p-values are also not significant
    end
end

disp('Holm-Bonferroni significant results:');
disp(significant_holm);



%%



% Combine the data into one array
allData = [pks0tw; pks2tw; pks10tw];

% Create group labels
groupLabels = [ones(size(pks0tw)); 2*ones(size(pks2tw)); 3*ones(size(pks10tw))];

% Perform Kruskal-Wallis test
[p, ~, stats] = kruskalwallis(allData, groupLabels, 'off');

% Display p-value
disp(['Kruskal-Wallis test p-value: ', num2str(p)]);

% Post-hoc test (Dunn's test for multiple comparisons)
if p < 0.05
    c = multcompare(stats, 'CType', 'dunn-sidak', 'Display', 'off'); % Dunn-Sidak correction
    disp('Significant differences found. Multiple comparison results:');
    disp(c);
else
    disp('No significant differences found between groups.');
end


%%



% Sample non-Gaussian datasets
data1 = pks0tw;  % Example skewed data
data2 = pks2tw;
data3 = pks10tw;

% Perform pairwise Mann-Whitney U (Wilcoxon rank-sum) tests
[p12, ~] = ranksum(data1, data2); % Data1 vs Data2
[p13, ~] = ranksum(data1, data3); % Data1 vs Data3
[p23, ~] = ranksum(data2, data3); % Data2 vs Data3

% Store p-values
p_values = [p12, p13, p23];

% Number of comparisons
num_tests = length(p_values);

% Bonferroni Correction (Divide alpha by number of tests)
alpha = 0.05;
bonferroni_threshold = alpha / num_tests;

% Holm-Bonferroni Correction (Stronger than Bonferroni)
[sorted_p, sorted_idx] = sort(p_values);  % Sort p-values
holm_thresholds = alpha ./ (num_tests:-1:1); % Compute Holm-Bonferroni thresholds

% Display results
disp('Pairwise Mann-Whitney U test results:');
disp(['p-value for Data1 vs Data2: ', num2str(p12)]);
disp(['p-value for Data1 vs Data3: ', num2str(p13)]);
disp(['p-value for Data2 vs Data3: ', num2str(p23)]);

% Applying Bonferroni Correction
significant_bonferroni = p_values < bonferroni_threshold;
disp(['Bonferroni significant results (alpha = ', num2str(bonferroni_threshold), '):']);
disp(significant_bonferroni);

% Applying Holm-Bonferroni Correction
significant_holm = false(size(p_values));
for i = 1:num_tests
    if sorted_p(i) < holm_thresholds(i)
        significant_holm(sorted_idx(i)) = true;
    else
        break; % If one test is not significant, all larger p-values are also not significant
    end
end

disp('Holm-Bonferroni significant results:');
disp(significant_holm);


%% Figure 1J: one maximum tensile force per independent tangle, in mN
conditionsTW = [0 2 10];
conditionColors = [0.72 0.29 0.45; 0.40 0.65 0.34; 0.45 0.36 0.72];
peakForce_mN = 1e3*m*g*[mx0tw; mx2tw; mx10tw];
conditionCode = [zeros(numel(mx0tw),1); 2*ones(numel(mx2tw),1); 10*ones(numel(mx10tw),1)];
peakForceTable = table(conditionCode, peakForce_mN, ...
    'VariableNames',{'Condition_TW','PeakForce_mN'});
outputFolder = fullfile(pwd,'results');
if ~isfolder(outputFolder), mkdir(outputFolder); end
writetable(peakForceTable,fullfile(outputFolder,'figure1J_peak_force_per_tangle.csv'));
figure('Name','Figure 1J: peak tensile force','Color','w','Position',[100 100 560 460]);
hold on;
for c = 1:numel(conditionsTW)
    selected = conditionCode == conditionsTW(c);
    boxchart(c*ones(nnz(selected),1),peakForce_mN(selected), ...
        'BoxFaceColor',conditionColors(c,:),'BoxFaceAlpha',0.75,'MarkerStyle','o');
end
xticks(1:3); xticklabels({'0','2','10'}); xlim([0.5 3.5]);
xlabel('N_{TW}'); ylabel('Maximum tensile force (mN)');
box on; set(gca,'FontSize',12,'TickDir','out');
exportgraphics(gcf,fullfile(outputFolder,'figure1J_peak_force.png'),'Resolution',300);
savefig(gcf,fullfile(outputFolder,'figure1J_peak_force.fig'));
