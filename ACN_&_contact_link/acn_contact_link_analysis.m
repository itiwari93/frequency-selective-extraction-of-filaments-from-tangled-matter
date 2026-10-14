%% ACN_CONTACT_LINK_ANALYSIS
% 
% Primary metrics
%   1. Pairwise average crossing number (ACN), reported as the mean
%      ACN strength per filament.
%   2. Contact link per filament, using definition:
%
%          cLink = (1/N) * sum_i sum_j A_ij * |L_ij|
%
%      where A_ij indicates physical contact and L_ij is the signed open
%      Gauss link between filaments i and j.
%
% Numerical choices
%   - Fiber diameter: 0.25 mm.
%   - Contacts: exact segment-to-segment distance on the original CT
%     polylines, rather than sampled vertex-to-vertex distance.
%   - Gauss integrals: exact polygonal-segment solid angles. 
%   - Resolution check: calculations are repeated after arc-length
%     resampling to 250 and 400 points per filament.
%
% Interpretation
%   - The signed open Gauss link is not generally an integer because the
%     filaments are open.
%   - ACN and |L_ij| are invariant to reversing a filament's digitization
%     direction.

clearvars;
close all;
clc;

%% User-adjustable settings
fiberDiameterMm = 0.25;
primarySamples = 250;
checkSamples = 400;
solidAngleBlockSize = 48;
warnRelativeChangePercent = 5;

datasetFiles = {
    'corrected_0wts_all_threads_s15.mat'
    'abs_corrected_2wts_all_connected_threads_s7.mat'
    'abs_corrected_10wts_all_connected_threads_s7.mat'
};
conditionNames = ["0 TW"; "2 TW"; "10 TW"];
conditionCodes = [0; 2; 10];

outputDirectory = fullfile(pwd, 'acn_contact_link_results');
figureDirectory = fullfile(outputDirectory, 'figures');
if ~exist(outputDirectory, 'dir'), mkdir(outputDirectory); end
if ~exist(figureDirectory, 'dir'), mkdir(figureDirectory); end

conditionColors = [
    0.72 0.29 0.45
    0.40 0.65 0.34
    0.45 0.36 0.72
];


%% Calculate ACN and contact link
summaryTable = table();
filamentTable = table();
pairTable = table();
analysis = struct([]);

for datasetIndex = 1:numel(datasetFiles)
    fprintf('\nAnalyzing %s (%s)\n', ...
        conditionNames(datasetIndex), datasetFiles{datasetIndex});

    source = load(datasetFiles{datasetIndex});
    validateInputDataset(source, datasetFiles{datasetIndex});

    numberFilaments = size(source.x, 2);
    rawCurves = cell(numberFilaments, 1);
    primaryCurves = cell(numberFilaments, 1);
    checkCurves = cell(numberFilaments, 1);

    for filamentIndex = 1:numberFilaments
        rawCurves{filamentIndex} = [
            source.x(:,filamentIndex), ...
            source.y(:,filamentIndex), ...
            source.z(:,filamentIndex)
        ];
        primaryCurves{filamentIndex} = resampleByArcLength( ...
            rawCurves{filamentIndex}, primarySamples);
        checkCurves{filamentIndex} = resampleByArcLength( ...
            rawCurves{filamentIndex}, checkSamples);
    end

    numberPairs = nchoosek(numberFilaments, 2);
    pairI = zeros(numberPairs, 1);
    pairJ = zeros(numberPairs, 1);
    isContact = false(numberPairs, 1);
    pairAcn = zeros(numberPairs, 1);
    pairAcnCheck = zeros(numberPairs, 1);
    signedOpenGaussLink = zeros(numberPairs, 1);
    signedOpenGaussLinkCheck = zeros(numberPairs, 1);

    pairCounter = 0;
    for filamentI = 1:numberFilaments-1
        for filamentJ = filamentI+1:numberFilaments
            pairCounter = pairCounter + 1;
            pairI(pairCounter) = filamentI;
            pairJ(pairCounter) = filamentJ;

            isContact(pairCounter) = polylinesAreInContact( ...
                rawCurves{filamentI}, rawCurves{filamentJ}, ...
                fiberDiameterMm);

            [pairAcn(pairCounter), signedOpenGaussLink(pairCounter)] = ...
                exactPolygonalPairMetrics( ...
                    primaryCurves{filamentI}, ...
                    primaryCurves{filamentJ}, ...
                    solidAngleBlockSize);

            [pairAcnCheck(pairCounter), ...
                    signedOpenGaussLinkCheck(pairCounter)] = ...
                exactPolygonalPairMetrics( ...
                    checkCurves{filamentI}, ...
                    checkCurves{filamentJ}, ...
                    solidAngleBlockSize);
        end
        fprintf('  completed filament %d of %d\n', ...
            filamentI, numberFilaments-1);
    end

    acnMatrix = symmetricPairMatrix( ...
        numberFilaments, pairI, pairJ, pairAcn);
    acnMatrixCheck = symmetricPairMatrix( ...
        numberFilaments, pairI, pairJ, pairAcnCheck);
    openGaussLinkMatrix = symmetricPairMatrix( ...
        numberFilaments, pairI, pairJ, signedOpenGaussLink);
    openGaussLinkMatrixCheck = symmetricPairMatrix( ...
        numberFilaments, pairI, pairJ, signedOpenGaussLinkCheck);
    contactMatrix = logical(symmetricPairMatrix( ...
        numberFilaments, pairI, pairJ, double(isContact)));

    % Each node strength sums its pair metric over all other filaments.
    % Taking the mean over nodes counts every unordered pair twice
    filamentAcnStrength = sum(acnMatrix, 2);
    filamentAcnStrengthCheck = sum(acnMatrixCheck, 2);
    contactLinkMatrix = contactMatrix .* openGaussLinkMatrix;
    contactLinkMatrixCheck = contactMatrix .* openGaussLinkMatrixCheck;
    filamentContactLink = sum(abs(contactLinkMatrix), 2);
    filamentContactLinkCheck = sum(abs(contactLinkMatrixCheck), 2);

    meanAcnPerFilament = mean(filamentAcnStrength);
    meanAcnPerFilamentCheck = mean(filamentAcnStrengthCheck);
    contactLinkPerFilament = mean(filamentContactLink);
    contactLinkPerFilamentCheck = mean(filamentContactLinkCheck);

    acnRelativeChangePercent = 100 * abs( ...
        meanAcnPerFilamentCheck - meanAcnPerFilament) / ...
        max(abs(meanAcnPerFilamentCheck), eps);
    contactLinkRelativeChangePercent = 100 * abs( ...
        contactLinkPerFilamentCheck - contactLinkPerFilament) / ...
        max(abs(contactLinkPerFilamentCheck), eps);

    if acnRelativeChangePercent > warnRelativeChangePercent
        warning('ACN resolution change for %s is %.2f%%.', ...
            conditionNames(datasetIndex), acnRelativeChangePercent);
    end
    if contactLinkRelativeChangePercent > warnRelativeChangePercent
        warning('Contact-link resolution change for %s is %.2f%%.', ...
            conditionNames(datasetIndex), ...
            contactLinkRelativeChangePercent);
    end

    summaryRow = table( ...
        conditionCodes(datasetIndex), conditionNames(datasetIndex), ...
        string(datasetFiles{datasetIndex}), numberFilaments, ...
        meanAcnPerFilament, meanAcnPerFilamentCheck, ...
        acnRelativeChangePercent, ...
        contactLinkPerFilament, contactLinkPerFilamentCheck, ...
        contactLinkRelativeChangePercent, sum(isContact), ...
        'VariableNames', {
            'condition_tw', 'condition', 'source_file', ...
            'filament_count', ...
            'mean_acn_per_filament_250', ...
            'mean_acn_per_filament_400', ...
            'acn_resolution_change_percent', ...
            'contact_link_per_filament_250', ...
            'contact_link_per_filament_400', ...
            'contact_link_resolution_change_percent', ...
            'physical_contact_edge_count'
        });
    summaryTable = [summaryTable; summaryRow]; %#ok<AGROW>

    filamentRows = table( ...
        repmat(conditionCodes(datasetIndex), numberFilaments, 1), ...
        repmat(conditionNames(datasetIndex), numberFilaments, 1), ...
        (1:numberFilaments)', filamentAcnStrength, ...
        filamentAcnStrengthCheck, filamentContactLink, ...
        filamentContactLinkCheck, sum(contactMatrix,2), ...
        'VariableNames', {
            'condition_tw', 'condition', 'filament_id', ...
            'acn_strength_250', 'acn_strength_400', ...
            'contact_link_strength_250', ...
            'contact_link_strength_400', 'contact_degree'
        });
    filamentTable = [filamentTable; filamentRows]; %#ok<AGROW>

    pairRows = table( ...
        repmat(conditionCodes(datasetIndex), numberPairs, 1), ...
        repmat(conditionNames(datasetIndex), numberPairs, 1), ...
        pairI, pairJ, isContact, pairAcn, pairAcnCheck, ...
        signedOpenGaussLink, signedOpenGaussLinkCheck, ...
        abs(signedOpenGaussLink).*isContact, ...
        'VariableNames', {
            'condition_tw', 'condition', 'filament_i', 'filament_j', ...
            'is_physical_contact', 'pair_acn_250', 'pair_acn_400', ...
            'signed_open_gauss_link_250', ...
            'signed_open_gauss_link_400', ...
            'contact_link_abs_250'
        });
    pairTable = [pairTable; pairRows]; %#ok<AGROW>

    analysis(datasetIndex).condition = conditionNames(datasetIndex);
    analysis(datasetIndex).sourceFile = datasetFiles{datasetIndex};
    analysis(datasetIndex).contactMatrix = contactMatrix;
    analysis(datasetIndex).acnMatrix = acnMatrix;
    analysis(datasetIndex).acnMatrixCheck = acnMatrixCheck;
    analysis(datasetIndex).openGaussLinkMatrix = openGaussLinkMatrix;
    analysis(datasetIndex).openGaussLinkMatrixCheck = ...
        openGaussLinkMatrixCheck;
    analysis(datasetIndex).contactLinkMatrix = contactLinkMatrix;
    analysis(datasetIndex).contactLinkMatrixCheck = ...
        contactLinkMatrixCheck;
    analysis(datasetIndex).filamentAcnStrength = filamentAcnStrength;
    analysis(datasetIndex).filamentContactLink = filamentContactLink;
end

%% Save numerical outputs before plotting
writetable(summaryTable, fullfile(outputDirectory, 'metric_summary.csv'));
writetable(filamentTable, fullfile(outputDirectory, ...
    'filament_metrics.csv'));
writetable(pairTable, fullfile(outputDirectory, 'pair_metrics.csv'));
save(fullfile(outputDirectory, 'acn_contact_link_analysis.mat'), ...
    'analysis', 'summaryTable', 'filamentTable', 'pairTable', ...
    'fiberDiameterMm', 'primarySamples', 'checkSamples');

disp(summaryTable);

%% Figure 1: ACN per filament box chart
acnFigure = figure( ...
    'Color', 'w', 'Units', 'pixels', 'Position', [80 80 620 520], ...
    'Name', 'ACN per filament');
plotFilamentBoxChart(gca, filamentTable, conditionNames, ...
    conditionColors, 'acn_strength_250', ...
    'ACN strength per filament', ...
    'Sum of pairwise ACN over the other 19 filaments');
exportFigureSet(acnFigure, figureDirectory, ...
    'acn_per_filament_boxchart');

%% Figure 2: contact link per filament box chart
contactLinkFigure = figure( ...
    'Color', 'w', 'Units', 'pixels', 'Position', [100 100 620 520], ...
    'Name', 'Contact link per filament');
plotFilamentBoxChart(gca, filamentTable, conditionNames, ...
    conditionColors, 'contact_link_strength_250', ...
    'Contact link per filament', ...
    'Sum of |open Gauss link| over physically contacting filaments');
exportFigureSet(contactLinkFigure, figureDirectory, ...
    'contact_link_per_filament_boxchart');

%% Figures 3-5: separate weighted contact-link networks
maximumContactLink = 0;
for datasetIndex = 1:numel(analysis)
    maximumContactLink = max(maximumContactLink, ...
        max(abs(analysis(datasetIndex).contactLinkMatrix), [], 'all'));
end

for datasetIndex = 1:numel(analysis)
    networkFigure = figure( ...
        'Color', 'w', 'Units', 'pixels', 'Position', [120 80 560 560], ...
        'Name', sprintf('%s contact-link network', ...
            conditionNames(datasetIndex)));
    axisHandle = axes(networkFigure, 'Position', [0.02 0.02 0.96 0.96]);
    plotContactLinkNetwork(axisHandle, ...
        analysis(datasetIndex).contactMatrix, ...
        analysis(datasetIndex).contactLinkMatrix, ...
        conditionColors(datasetIndex,:), maximumContactLink);
    networkBaseName = sprintf('contact_link_network_%dTW', ...
        conditionCodes(datasetIndex));
    exportFigureSet(networkFigure, figureDirectory, networkBaseName);
end
fprintf('\nSaved results to:\n  %s\n', outputDirectory);

%% Local functions
function validateInputDataset(source, filename)
    requiredFields = {'x','y','z'};
    assert(all(isfield(source, requiredFields)), ...
        'Expected x, y, and z variables in %s.', filename);
    assert(isequal(size(source.x), size(source.y), size(source.z)), ...
        'x, y, and z have different sizes in %s.', filename);
    assert(size(source.x,2) == 20, ...
        'Expected 20 filaments in %s.', filename);
    assert(all(isfinite(source.x), 'all') && ...
        all(isfinite(source.y), 'all') && ...
        all(isfinite(source.z), 'all'), ...
        'Non-finite coordinates found in %s.', filename);
end

function resampled = resampleByArcLength(curve, numberSamples)
    segmentLength = vecnorm(diff(curve,1,1), 2, 2);
    cumulativeArc = [0; cumsum(segmentLength)];
    [uniqueArc, uniqueIndex] = unique(cumulativeArc, 'stable');
    assert(uniqueArc(end) > 0, 'A filament has zero total arc length.');
    queryArc = linspace(0, uniqueArc(end), numberSamples)';
    resampled = interp1(uniqueArc, curve(uniqueIndex,:), ...
        queryArc, 'linear');
end

function [pairAcn, signedOpenGaussLink] = ...
        exactPolygonalPairMetrics(curveA, curveB, blockSize)
    a0 = curveA(1:end-1,:);
    a1 = curveA(2:end,:);
    b0 = curveB(1:end-1,:);
    b1 = curveB(2:end,:);
    dA = a1 - a0;
    dB = b1 - b0;

    signedOpenGaussLink = 0;
    halfPairAcn = 0;

    for firstRow = 1:blockSize:size(a0,1)
        rows = firstRow:min(firstRow+blockSize-1, size(a0,1));

        n0 = reshape(a0(rows,:), [], 1, 3) - ...
            reshape(b0, 1, [], 3);
        ns = reshape(a1(rows,:), [], 1, 3) - ...
            reshape(b0, 1, [], 3);
        nst = reshape(a1(rows,:), [], 1, 3) - ...
            reshape(b1, 1, [], 3);
        nt = reshape(a0(rows,:), [], 1, 3) - ...
            reshape(b1, 1, [], 3);

        % Exact solid angle of the spherical quadrilateral. 
        solidAngle = -(orientedTriangleSolidAngle(n0, ns, nst) + ...
            orientedTriangleSolidAngle(n0, nst, nt));
        solidAngle = mod(solidAngle + 2*pi, 4*pi) - 2*pi;

        segmentCross = cross3( ...
            reshape(dA(rows,:), [], 1, 3), ...
            reshape(dB, 1, [], 3));
        orientation = sign(dot3(n0, segmentCross));
        solidAngle = orientation .* abs(solidAngle);

        contribution = solidAngle / (4*pi);
        contribution(~isfinite(contribution)) = 0;
        signedOpenGaussLink = signedOpenGaussLink + ...
            sum(contribution, 'all');
        halfPairAcn = halfPairAcn + sum(abs(contribution), 'all');
    end

    % For two distinct curves, the average unsigned crossing count has
    % twice the 1/(4*pi) normalization of the signed Gauss link.
    pairAcn = 2 * halfPairAcn;
end

function angle = orientedTriangleSolidAngle(a, b, c)
    normA = norm3(a);
    normB = norm3(b);
    normC = norm3(c);
    numerator = dot3(a, cross3(b,c));
    denominator = normA.*normB.*normC + ...
        dot3(a,b).*normC + dot3(b,c).*normA + ...
        dot3(c,a).*normB;
    angle = 2 * atan2(numerator, denominator);
    angle(~isfinite(angle)) = 0;
end

function value = dot3(a, b)
    value = sum(a.*b, 3);
end

function value = norm3(a)
    value = sqrt(sum(a.^2, 3));
end

function value = cross3(a, b)
    component1 = a(:,:,2).*b(:,:,3) - a(:,:,3).*b(:,:,2);
    component2 = a(:,:,3).*b(:,:,1) - a(:,:,1).*b(:,:,3);
    component3 = a(:,:,1).*b(:,:,2) - a(:,:,2).*b(:,:,1);
    value = cat(3, component1, component2, component3);
end

function matrix = symmetricPairMatrix(numberFilaments, pairI, pairJ, values)
    matrix = zeros(numberFilaments);
    for pairIndex = 1:numel(values)
        matrix(pairI(pairIndex), pairJ(pairIndex)) = values(pairIndex);
        matrix(pairJ(pairIndex), pairI(pairIndex)) = values(pairIndex);
    end
end

function isContact = polylinesAreInContact(curveA, curveB, threshold)
    % A vertex contact is automatically a segment contact and provides a
    % fast exit for most contacting pairs.
    if exist('knnsearch', 'file') == 2
        [~, vertexDistance] = knnsearch(curveB, curveA, 'K', 1);
        if min(vertexDistance) < threshold
            isContact = true;
            return;
        end
    end

    a0 = curveA(1:end-1,:);
    a1 = curveA(2:end,:);
    b0 = curveB(1:end-1,:);
    b1 = curveB(2:end,:);
    midpointA = 0.5*(a0+a1);
    midpointB = 0.5*(b0+b1);
    lengthA = vecnorm(a1-a0, 2, 2);
    lengthB = vecnorm(b1-b0, 2, 2);

    % If two segments approach within threshold, their midpoints must be
    % within threshold + half the two segment lengths. The broader global
    % radius below is therefore a safe candidate search, not an
    % approximation to contact.
    globalRadius = threshold + 0.5*(max(lengthA)+max(lengthB));
    if exist('rangesearch', 'file') == 2
        candidateLists = rangesearch(midpointB, midpointA, globalRadius);
    else
        candidateLists = cell(size(midpointA,1),1);
        for segmentI = 1:size(midpointA,1)
            centerDistance = vecnorm( ...
                midpointB-midpointA(segmentI,:), 2, 2);
            candidateLists{segmentI} = find(centerDistance <= globalRadius);
        end
    end

    isContact = false;
    for segmentI = 1:numel(candidateLists)
        candidates = candidateLists{segmentI};
        if isempty(candidates), continue; end

        centerDistance = vecnorm( ...
            midpointB(candidates,:)-midpointA(segmentI,:), 2, 2);
        safeBound = threshold + ...
            0.5*(lengthA(segmentI)+lengthB(candidates));
        candidates = candidates(centerDistance <= safeBound);

        for candidateIndex = 1:numel(candidates)
            segmentJ = candidates(candidateIndex);
            if segmentSegmentDistance( ...
                    a0(segmentI,:), a1(segmentI,:), ...
                    b0(segmentJ,:), b1(segmentJ,:)) < threshold
                isContact = true;
                return;
            end
        end
    end
end

function distance = segmentSegmentDistance(p0, p1, q0, q1)
    % Minimum Euclidean distance between two finite 3-D line segments.
    u = p1-p0;
    v = q1-q0;
    w = p0-q0;
    a = dot(u,u);
    b = dot(u,v);
    c = dot(v,v);
    d = dot(u,w);
    e = dot(v,w);
    denominator = a*c-b*b;
    small = 1e-14 * max([a*c, 1]);

    if a <= eps && c <= eps
        distance = norm(p0-q0);
        return;
    elseif a <= eps
        t = min(max(e/c, 0), 1);
        distance = norm(p0-(q0+t*v));
        return;
    elseif c <= eps
        s = min(max(-d/a, 0), 1);
        distance = norm((p0+s*u)-q0);
        return;
    end

    if denominator < small
        sNumerator = 0;
        sDenominator = 1;
        tNumerator = e;
        tDenominator = c;
    else
        sNumerator = b*e-c*d;
        tNumerator = a*e-b*d;
        sDenominator = denominator;
        tDenominator = denominator;
        if sNumerator < 0
            sNumerator = 0;
            tNumerator = e;
            tDenominator = c;
        elseif sNumerator > sDenominator
            sNumerator = sDenominator;
            tNumerator = e+b;
            tDenominator = c;
        end
    end

    if tNumerator < 0
        tNumerator = 0;
        if -d < 0
            sNumerator = 0;
            sDenominator = 1;
        elseif -d > a
            sNumerator = sDenominator;
        else
            sNumerator = -d;
            sDenominator = a;
        end
    elseif tNumerator > tDenominator
        tNumerator = tDenominator;
        if -d+b < 0
            sNumerator = 0;
            sDenominator = 1;
        elseif -d+b > a
            sNumerator = sDenominator;
        else
            sNumerator = -d+b;
            sDenominator = a;
        end
    end

    if abs(sNumerator) < eps
        s = 0;
    else
        s = sNumerator/sDenominator;
    end
    if abs(tNumerator) < eps
        t = 0;
    else
        t = tNumerator/tDenominator;
    end
    distance = norm(w+s*u-t*v);
end

function plotFilamentBoxChart(axisHandle, filamentTable, conditionNames, ...
        conditionColors, variableName, yLabelText, subtitleText)
    hold(axisHandle, 'on');
    numberConditions = numel(conditionNames);

    for conditionIndex = 1:numberConditions
        rows = filamentTable.condition == conditionNames(conditionIndex);
        values = filamentTable.(variableName)(rows);
        darkColor = 0.65 * conditionColors(conditionIndex,:);
        boxchart(axisHandle, ...
            repmat(conditionIndex, numel(values), 1), values, ...
            'BoxFaceColor', conditionColors(conditionIndex,:), ...
            'BoxFaceAlpha', 0.72, ...
            'WhiskerLineColor', darkColor, ...
            'MarkerColor', darkColor, ...
            'LineWidth', 1.35);
    end

    xlim(axisHandle, [0.55 numberConditions+0.45]);
    ylim(axisHandle, [0 inf]);
    xticks(axisHandle, 1:numberConditions);
    xticklabels(axisHandle, conditionNames);
    ylabel(axisHandle, yLabelText);
    title(axisHandle, yLabelText, 'FontWeight', 'bold');
    subtitle(axisHandle, subtitleText);
    grid(axisHandle, 'on');
    axisHandle.GridColor = [0.86 0.86 0.86];
    axisHandle.GridAlpha = 0.65;
    axisHandle.XGrid = 'off';
    axisHandle.Box = 'off';
    axisHandle.FontName = 'Arial';
    axisHandle.FontSize = 11;
    axisHandle.LineWidth = 1;
    hold(axisHandle, 'off');
end
function plotContactLinkNetwork(axisHandle, contactMatrix, ...
        contactLinkMatrix, nodeColor, globalMaximum)
    numberFilaments = size(contactMatrix,1);
    theta = linspace(pi/2, pi/2-2*pi, numberFilaments+1);
    theta(end) = [];
    xNode = cos(theta);
    yNode = sin(theta);

    [edgeI, edgeJ] = find(triu(contactMatrix,1));
    edgeWeight = abs(contactLinkMatrix(sub2ind( ...
        size(contactLinkMatrix), edgeI, edgeJ)));
    [~, edgeOrder] = sort(edgeWeight, 'ascend');

    hold(axisHandle, 'on');
    lightEdge = [0.82 0.82 0.82];
    darkEdge = [0.42 0.16 0.30];
    denominator = max(globalMaximum, eps);

    for orderIndex = 1:numel(edgeOrder)
        edgeIndex = edgeOrder(orderIndex);
        normalizedWeight = sqrt(edgeWeight(edgeIndex)/denominator);
        edgeColor = (1-normalizedWeight)*lightEdge + ...
            normalizedWeight*darkEdge;
        edgeWidth = 0.45 + 4.0*normalizedWeight;
        plot(axisHandle, ...
            xNode([edgeI(edgeIndex),edgeJ(edgeIndex)]), ...
            yNode([edgeI(edgeIndex),edgeJ(edgeIndex)]), '-', ...
            'Color', edgeColor, 'LineWidth', edgeWidth);
    end

    nodeColors = repmat(nodeColor, numberFilaments, 1);
    scatter(axisHandle, xNode, yNode, 145, nodeColors, 'filled', ...
        'MarkerEdgeColor', [0.12 0.12 0.12], 'LineWidth', 0.9);
    axis(axisHandle, 'equal');
    xlim(axisHandle, [-1.12 1.12]);
    ylim(axisHandle, [-1.12 1.12]);
    axis(axisHandle, 'off');
    hold(axisHandle, 'off');
end

function exportFigureSet(figureHandle, figureDirectory, baseName)
    baseName = string(baseName);
    exportgraphics(figureHandle, fullfile(figureDirectory, ...
        baseName + ".png"), 'Resolution', 300);
    print(figureHandle, fullfile(figureDirectory, ...
        baseName + ".svg"), '-dsvg');
    savefig(figureHandle, fullfile(figureDirectory, ...
        baseName + ".fig"));
end
