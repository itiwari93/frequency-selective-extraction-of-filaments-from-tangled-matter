%% PCA MAJOR AND MINOR AXIS BOXPLOTS
%
% Loads:
%   plot_colrs.mat
%   tangle_2D_compaction_individual.csv
%
% Produces:
%   Figure 1 = Major principal-axis RMS extent
%   Figure 2 = Minor principal-axis RMS extent
%
% Conditions 0, 2, 10 are equally spaced on the x-axis.
% Individual data points are NOT plotted.

clear;
clc;
close all;


%% ============================================================
% LOAD DATA AND COLORS
% ============================================================

load('plot_colrs.mat');

data = readtable('tangle_2D_compaction_individual.csv');


% Rows of the stored plotting palette correspond to:
%
% plot_colrs(1,:) = 0 TW
% plot_colrs(2,:) = 2 TW
% plot_colrs(3,:) = 10 TW

color10 = plot_colrs(3,:);
color2  = plot_colrs(2,:);
color0  = plot_colrs(1,:);


conditionColors = [...
    color0; ...
    color2; ...
    color10];


conditions = [0 2 10];

% Equally spaced plotting positions.
xPositions = [1 2 3];


%% ============================================================
% COMMON PLOT SETTINGS
% ============================================================

boxWidth = 0.55;

fontSize = 12;

axisLineWidth = 1;


%% ============================================================
% FIGURE 1
% MAJOR PRINCIPAL AXIS
% ============================================================

figure(...
    'Color','w', ...
    'Position',[150 150 500 350]);

hold on;


for conditionIndex = 1:numel(conditions)

    currentCondition = ...
        conditions(conditionIndex);


    rows = ...
        data.Condition == currentCondition;


    y = ...
        data.PCA_MajorRMS_mm(rows);


    boxchart(...
        repmat(xPositions(conditionIndex),size(y)), ...
        y, ...
        'BoxFaceColor',conditionColors(conditionIndex,:), ...
        'BoxFaceAlpha',0.75, ...
        'BoxWidth',boxWidth, ...
        'MarkerStyle','none');

end


%% Axis formatting

xlim([0.5 3.5]);

xticks(xPositions);

xticklabels({'0','2','10'});


xlabel('N_{TW}');

ylabel('Major-axis RMS extent (mm)');


box on;


set(gca, ...
    'FontSize',fontSize, ...
    'LineWidth',axisLineWidth, ...
    'TickDir','out');


%% ============================================================
% FIGURE 2
% MINOR PRINCIPAL AXIS
% ============================================================

figure(...
    'Color','w', ...
    'Position',[700 150 500 350]);

hold on;


for conditionIndex = 1:numel(conditions)

    currentCondition = ...
        conditions(conditionIndex);


    rows = ...
        data.Condition == currentCondition;


    y = ...
        data.PCA_MinorRMS_mm(rows);


    boxchart(...
        repmat(xPositions(conditionIndex),size(y)), ...
        y, ...
        'BoxFaceColor',conditionColors(conditionIndex,:), ...
        'BoxFaceAlpha',0.75, ...
        'BoxWidth',boxWidth, ...
        'MarkerStyle','none');

end


%% Axis formatting

xlim([0.5 3.5]);

xticks(xPositions);

xticklabels({'0','2','10'});


xlabel('N_{TW}');

ylabel('Minor-axis RMS extent (mm)');


box on;


set(gca, ...
    'FontSize',fontSize, ...
    'LineWidth',axisLineWidth, ...
    'TickDir','out');
%% Save the two manuscript principal-axis panels.
outputFolder = fullfile(pwd,'results');
if ~isfolder(outputFolder), mkdir(outputFolder); end
figures = findall(groot,'Type','figure');
for figureIndex = 1:numel(figures)
    axisHandle = findobj(figures(figureIndex),'Type','axes');
    if contains(axisHandle.YLabel.String,'Major')
        stem = 'figure1E_major_axis';
    else
        stem = 'figure1F_minor_axis';
    end
    exportgraphics(figures(figureIndex),fullfile(outputFolder,[stem '.png']),'Resolution',300);
    savefig(figures(figureIndex),fullfile(outputFolder,[stem '.fig']));
end
