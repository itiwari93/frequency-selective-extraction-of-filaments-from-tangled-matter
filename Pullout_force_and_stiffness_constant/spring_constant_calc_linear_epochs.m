clc; clear;

% ---------------- User settings ----------------
v_mm_min = 15;       % pull speed (mm/min) -- for reporting only
fc_Hz    = 1;        % low-pass cutoff on TIME signal (Hz)
butterN  = 4;        % filter order

skipStart_s = 0.2;   % skip first 0.2 s AFTER engagement
skipEnd_s   = 0.2;   % skip last 0.2 s of the file


% Engagement detection
tNoise_s = 0.5;      % seconds from start used to estimate force noise
thrSigma = 5;        % threshold = thrSigma*std(noise segment)
thrMin_N = 0.0001;     % absolute minimum threshold (N)

% Backbone smoothing in DISPLACEMENT domain
sgolayOrder = 3;     % Savitzky-Golay order
winFrac     = 0.01;  % window as fraction of engaged samples (0.01-0.05 typical)


% Stick–slip segmentation settings (in z-domain)
minPeakPromFrac   = 0.01;   % prominence as fraction of force range (0.05–0.2 typical)
minPeakDist_mm    = 0.1;    % minimum spacing between slip peaks in mm (tune)
minRise_N         = 0.0005;   % minimum valley→peak force rise to accept an epoch
minEpochPts       = 20;     % minimum number of samples in an epoch
robustFitOn       = true;   % use robustfit (recommended)


files = dir('*TW_t*_*.csv');
fileNames = {files.name};

% Outputs
k_fit_Npm   = nan(numel(fileNames),1);
k_med_Npm   = nan(numel(fileNames),1);
z0_mm       = nan(numel(fileNames),1);
F_peak_N    = nan(numel(fileNames),1);
F_mean_N    = nan(numel(fileNames),1);
W_J         = nan(numel(fileNames),1);
fs_est_Hz   = nan(numel(fileNames),1);
stickSlipRMS_N = nan(numel(fileNames),1);
k_all = cell(numel(fileNames),1);
Iv2 = nan(numel(fileNames),1);
c_eq_pull_NsPm = nan(numel(fileNames),1);
v_mean = nan(numel(fileNames),1);

for k = 1:numel(fileNames)
    filename = fileNames{k};

    % --- Read load, displacement, and time columns ---
    T = readPulloutFile(filename);

    t = T.Time_s(:);         % s
    x = T.Distance_mm(:);    % mm (stage displacement or extension measure)
    F = -T.Load_N(:);         % N (negative sign because force is directed downwards as the thread is pulled out)

    % --- Uniformize the time base ---
    [tU, xU, FU, fs] = uniformizeTimeBase(t, x, F);
    fs_est_Hz(k) = fs;

    % --- Low-pass in time (removes sensor noise) ---
    Wn = fc_Hz / (fs/2);
    if Wn >= 1
        xF = xU;
        FF = FU;
    else
        [b,a] = butter(butterN, Wn, 'low');
        xF = filtfilt(b,a,xU);
        FF = filtfilt(b,a,FU);
    end

    % --- Baseline correction ---
    % Zero force based on early segment median (helps thresholding)
    N0 = max(5, round(tNoise_s * fs));
    N0 = min(N0, numel(FF));
    F0 = median(FF(1:N0));
    FF = FF - F0;

    % Zero displacement at start
    xF = xF - xF(1);

    % --- Engagement detection (slack length z0) ---
    FnoiseStd = std(FF(1:N0));
    thr = max(thrMin_N, thrSigma * FnoiseStd);
    iEng = find(FF > thr, 1, 'first');
    if isempty(iEng) || iEng >= numel(FF)-10
        warning('No clear engagement found in %s', filename);
        continue;
    end
    z0_mm(k) = xF(iEng);

    % --- Build F(z) in displacement domain after engagement ---
    %idx = iEng:numel(xF);
    iEnd = numel(xF) - round(skipEnd_s * fs);
    iEnd = max(iEnd, iEng+10);

    iStart = iEng + round(skipStart_s * fs);
    iStart = min(iStart, iEnd-10);

    idx = iStart:iEnd;



    z_mm = xF(idx) - xF(iEng);   % redefine z=0 at engagement
    Fz   = FF(idx);

    % Ensure monotonic z (pullout should be monotonic; enforce if noisy)
    % Use cummax to remove tiny reversals
    z_mm = cummax(z_mm);

    % Remove duplicates for interpolation
    [z_mm_u, ia] = unique(z_mm, 'stable');
    Fz_u = Fz(ia);

    if numel(z_mm_u) < 50
        warning('Too few engaged samples in %s', filename);
        continue;
    end

    % Resample onto uniform z grid (better derivative)
    dz_mm = median(diff(z_mm_u));
    if ~isfinite(dz_mm) || dz_mm <= 0
        dz_mm = (z_mm_u(end)-z_mm_u(1)) / (numel(z_mm_u)-1);
    end
    zq_mm = (z_mm_u(1):dz_mm:z_mm_u(end))';
    Fq    = interp1(z_mm_u, Fz_u, zq_mm, 'linear', 'extrap');

    % Smooth backbone with Savitzky-Golay in z-domain
    win = max(11, 2*floor(0.5*winFrac*numel(zq_mm))+1); % odd
    win = min(win, numel(zq_mm) - mod(numel(zq_mm)+1,2)); % keep odd <= N
    if win < (sgolayOrder+2)
        win = sgolayOrder + 2 + mod(sgolayOrder+2+1,2);
    end
    F_back = smoothdata(Fq, 'sgolay', win);

    % Tangent stiffness k(z) = dF/dz (convert mm->m)
    z_m  = zq_mm * 1e-3;
    k_t  = gradient(F_back, median(diff(z_m)));  % N/m

    % Summary stiffness stats (ignore negative spikes from slip)
    k_pos = k_t(isfinite(k_t) & k_t > 0);
    if ~isempty(k_pos)
        k_med_Npm(k) = median(k_pos);
    end

    seg = fitStickSlipEpochs(z_m, Fq, dz_mm, ...
        minPeakPromFrac, minPeakDist_mm, minRise_N, minEpochPts, robustFitOn);

    % seg.k_Npm is a vector of per-epoch stiffnesses
    if ~isempty(seg.k_Npm)
        k_fit_Npm(k) = median(seg.k_Npm);   % robust summary stiffness
        k_all{k} = seg.k_Npm;
    end

    % ---- Visualization: per-epoch k vs average displacement ----
    if ~isempty(seg.k_Npm)

        nE = numel(seg.k_Npm);
        zAvg_m = nan(nE,1);

        for i = 1:nE
            ii = seg.epochStartIdx(i):seg.epochEndIdx(i);
            %zAvg_m(i) = mean(z_m(ii));     % average extension for this epoch (m)
            zAvg_m(i) = 0.5*(z_m(seg.epochStartIdx(i)) + z_m(seg.epochEndIdx(i)));
        end

        % Plot: k points along z
        figure('Name', [filename ' | k vs z']); clf
        plot(z_m*1e3, Fq, 'k-'); hold on
        yyaxis right
        plot(zAvg_m*1e3, seg.k_Npm, 'o', 'MarkerFaceColor','auto'); % k in N/m
        ylabel('Epoch stiffness k (N/m)');

        yyaxis left
        ylabel('Force F (N)');
        xlabel('Engaged extension z (mm)');
        title('Per-epoch stiffness estimates along pullout');
        grid on

        % Optional: overlay a running median trend of k(z)
        % (Bin by z, take median k in each bin)
        nbins = 20;
        edges = linspace(min(zAvg_m), max(zAvg_m), nbins+1);
        kMedBin = nan(nbins,1);
        zCtr = 0.5*(edges(1:end-1)+edges(2:end));
        for b = 1:nbins
            inb = zAvg_m >= edges(b) & zAvg_m < edges(b+1);
            if any(inb)
                kMedBin(b) = median(seg.k_Npm(inb));
            end
        end

        yyaxis right
        plot(zCtr*1e3, kMedBin, '-', 'LineWidth', 1.5);
        legend('F(z)', 'k per epoch', 'k running median (binned)', 'Location','best');
    end


    % optional: keep an additional summary (e.g., 25–75% spread)
    %k_med_Npm(k) = k_fit_Npm(k);


    % Friction/stick-slip proxies
    F_peak_N(k) = max(Fq);
    F_mean_N(k) = mean(F_back);

    % Stick-slip intensity as RMS of residual (raw - backbone)
    res = Fq - F_back;
    stickSlipRMS_N(k) = rms(res(isfinite(res)));

    % Total work (energy input) during extraction: W = ∫ F dz
    W_J(k) = trapz(z_m, max(Fq,0)); % J (clamp negatives if any)

    % ---- Equivalent viscous damping from pullout work (time-domain) ----
t_use = tU(idx);                         % s
z_use = (xF(idx) - xF(iEng)) * 1e-3;     % m  (engaged extension vs time)
% Estimate velocity using finite differences.
dt = diff(t_use);
dz = diff(z_use);
v  = dz ./ dt;                           % m/s
t_mid = t_use(1:end-1) + 0.5*dt;         % s (midpoints)

% Integral ∫ v^2 dt over the pullout (m^2/s)
Iv2(k) = trapz(t_mid, v.^2);

% Equivalent viscous damping (N*s/m)
if Iv2(k) > 0 && isfinite(Iv2(k))
    c_eq_pull_NsPm(k) = W_J(k) / Iv2(k);
else
    c_eq_pull_NsPm(k) = NaN;
end

% Optional sanity checks / alternative constant-speed formula
L = z_use(end) - z_use(1);               % m extracted in this window
v_mean(k) = L / (t_use(end) - t_use(1));    % m/s (measured mean speed)

% If speed is near-constant, this should be close to c_eq_pull_NsPm
c_eq_constspeed = W_J(k) / max(v_mean(k) * L, eps);


% Grouped summaries are constructed after all trials have been processed.



    fprintf('%s | fs=%.2f Hz | z0=%.3f mm | k_fit=%.3e N/m | k_med=%.3e N/m | Fpeak=%.3f N | W=%.3e J\n | Iv2=%.3e | c_eq=%.3e N*s/m | v_mean=%.3e m/s\n', ...
        filename, fs, z0_mm(k), k_fit_Npm(k), k_med_Npm(k), F_peak_N(k), W_J(k), Iv2(k), c_eq_pull_NsPm(k), v_mean(k));

    % --- Quick plots ---
    figure('Name', filename); clf
    plot(z_m*1e3, Fq, 'k-'); hold on
    plot(z_m(seg.valleyIdx)*1e3, Fq(seg.valleyIdx), 'bo', 'MarkerFaceColor','b');
    plot(z_m(seg.peakIdx)*1e3,   Fq(seg.peakIdx),   'ro', 'MarkerFaceColor','r');

    for i = 1:numel(seg.epochStartIdx)
        ii = seg.epochStartIdx(i):seg.epochEndIdx(i);
        zii = z_m(ii);
        Fhat = seg.b0(i) + seg.k_Npm(i)*zii;
        plot(zii*1e3, Fhat, '-', 'LineWidth', 1.2);
    end

    xlabel('Engaged extension z (mm)');
    ylabel('Force F (N)');
    title(sprintf('Epoch stiffness median k = %.3e N/m', k_fit_Npm(k)));
    grid on
    legend('F(z)','Valleys','Peaks','Epoch fits','Location','best');
    ax=gca;
    ax.FontSize=30;
    %     figure('Name', filename); clf
    %     tiledlayout(2,1);
    %
    %     nexttile
    %     plot(zq_mm, Fq, 'k-'); hold on
    %     plot(zq_mm, F_back, 'r-', 'LineWidth', 1.3);
    %     xlabel('Engaged extension z (mm)');
    %     ylabel('Force F (N)');
    %     legend('Raw (resampled)','Backbone (smoothed)','Location','best');
    %     grid on
    %
    %     nexttile
    %     plot(zq_mm, k_t, 'b-');
    %     xlabel('Engaged extension z (mm)');
    %     ylabel('Tangent stiffness k_t = dF/dz (N/m)');
    %     grid on
end

% Summary table
results = table(fileNames(:), z0_mm, k_fit_Npm, k_med_Npm, F_peak_N, F_mean_N, stickSlipRMS_N, W_J,c_eq_pull_NsPm, fs_est_Hz, ...
    'VariableNames', {'File','z0_mm','k_fit_Npm','k_med_Npm','F_peak_N','F_mean_N','StickSlipRMS_N','Work_J','C_eq_pull','fs_Hz'});

disp(results);

% Per-tangle medians are the independent stiffness measurements.
conditionTW = nan(numel(fileNames),1);
for trial = 1:numel(fileNames)
    conditionTW(trial) = str2double(regexp(fileNames{trial},'^([0-9]+)TW','tokens','once'));
end
results.Condition_TW = conditionTW;
outputFolder = fullfile(pwd,'results');
if ~isfolder(outputFolder), mkdir(outputFolder); end
writetable(results,fullfile(outputFolder,'pullout_stiffness_per_tangle.csv'));
conditionsTW = [0 2 10];
k_grouped = cell(1,3);
conditionColors = [0.72 0.29 0.45; 0.40 0.65 0.34; 0.45 0.36 0.72];
figure('Name','Quasi-static stiffness per tangle','Color','w','Position',[100 100 560 460]);
hold on;
for c = 1:3
    selected = conditionTW == conditionsTW(c);
    k_grouped{c} = vertcat(k_all{selected});
    boxchart(c*ones(nnz(selected),1),k_fit_Npm(selected), ...
        'BoxFaceColor',conditionColors(c,:),'BoxFaceAlpha',0.75,'MarkerStyle','o');
end
xticks(1:3); xticklabels({'0','2','10'}); xlim([0.5 3.5]);
xlabel('N_{TW}'); ylabel('Effective stiffness (N/m)');
box on; set(gca,'FontSize',12,'TickDir','out');
exportgraphics(gcf,fullfile(outputFolder,'pullout_stiffness_per_tangle.png'),'Resolution',300);
savefig(gcf,fullfile(outputFolder,'pullout_stiffness_per_tangle.fig'));

% ===================== Functions =====================

function T = readPulloutFile(filename)
% Read a pullout export with load, displacement, and time columns.

% Find header line that contains "Reading" OR "Time" (fallback)
fid = fopen(filename,'r');
headerLine = [];
lineNum = 0;
line = fgetl(fid);
while ischar(line)
    lineNum = lineNum + 1;
    if contains(lower(line), 'reading') || contains(lower(line), 'time')
        headerLine = lineNum;
        break;
    end
    line = fgetl(fid);
end
fclose(fid);

if isempty(headerLine)
    headerLine = 1; % fallback
end

opts = detectImportOptions(filename, 'Delimiter', {',','\t'});
opts.VariableNamesLine = headerLine;
opts.DataLines = [headerLine+1, inf];

opts.VariableNamingRule = 'preserve';
Traw = readtable(filename, opts);
vn = string(Traw.Properties.VariableNames);
vnl = lower(vn);

iLoad = find(contains(vnl, "load") | contains(vnl, "force"), 1, 'first');
iDist = find(contains(vnl, "distance") | contains(vnl, "position") | contains(vnl, "disp"), 1, 'first');
iTime = find(contains(vnl, "time"), 1, 'first');

if isempty(iLoad) || isempty(iDist) || isempty(iTime)
    error('Could not identify Load/Distance/Time columns in %s', filename);
end

T = Traw(:, [iLoad, iDist, iTime]);
T.Properties.VariableNames = {'Load_N','Distance_mm','Time_s'};
end


function [tU, xU, FU, fs] = uniformizeTimeBase(t, x, F)
t = t(:); x = x(:); F = F(:);

good = isfinite(t) & isfinite(x) & isfinite(F);
t = t(good); x = x(good); F = F(good);

[t, order] = sort(t);
x = x(order); F = F(order);

[t, ia] = unique(t, 'stable');
x = x(ia); F = F(ia);

dt = diff(t);
dt = dt(dt > 0 & isfinite(dt));
if isempty(dt)
    error('Time vector not usable (no positive dt).');
end

dtMed = median(dt);
fs = 1/dtMed;

jitter = median(abs(dt - dtMed)) / dtMed;
if jitter > 0.02
    tU = (t(1):dtMed:t(end))';
    xU = interp1(t, x, tU, 'linear', 'extrap');
    FU = interp1(t, F, tU, 'linear', 'extrap');
else
    tU = t; xU = x; FU = F;
end
end


function seg = fitStickSlipEpochs(z_m, F_N, dz_mm, promFrac, minDist_mm, minRise_N, minEpochPts, doRobust)
% Segment a jagged stick-slip pullout curve into stick epochs (valley->peak)
% and fit F ≈ k z + b in each epoch.
%
% Inputs:
%   z_m        monotonic extension (m)
%   F_N        force (N)
%   dz_mm      approx z grid spacing (mm) (for converting minDist)
%
% Outputs in seg:
%   k_Npm      per-epoch stiffness (N/m)
%   b0         intercept per epoch (N)
%   peakIdx    indices of detected peaks
%   valleyIdx  indices of valleys chosen for each peak
%   epochStartIdx / epochEndIdx indices defining each fitted epoch
%   R2         goodness of fit per epoch

z_m = z_m(:); F_N = F_N(:);
N = numel(z_m);

% Light smoothing JUST for event detection (not for fitting)
% Use movmedian to preserve steps while suppressing noise
winDet = max(5, round(0.01*N)); % ~1% length; tune if needed
Fdet = F_N;%smoothdata(F_N, 'movmedian', winDet);

Frng = prctile(Fdet,99) - prctile(Fdet,1);
prom = promFrac * Frng;

minDistSamp = max(1, round(minDist_mm / dz_mm));

% Peaks = candidate slip points
[pks, locs] = findpeaks(Fdet, ...
    'MinPeakProminence', prom, ...
    'MinPeakDistance', minDistSamp);

seg.k_Npm = [];
seg.b0 = [];
seg.peakIdx = locs;
seg.valleyIdx = [];
seg.epochStartIdx = [];
seg.epochEndIdx = [];
seg.R2 = [];

if isempty(locs)
    return;
end

prevBound = 1;
for i = 1:numel(locs)
    ip = locs(i);

    % Valley = minimum in [prevBound .. ip]
    [vmin, ivRel] = min(Fdet(prevBound:ip));
    iv = prevBound + ivRel - 1;

    % Basic acceptance checks
    if (ip - iv + 1) < minEpochPts
        prevBound = ip;
        continue;
    end
    if (Fdet(ip) - vmin) < minRise_N
        prevBound = ip;
        continue;
    end

    ii = iv:ip;

    % OPTIONAL: remove points with strongly negative slope (micro-slip inside epoch)
    % Keep points where dF/dz isn't very negative
    dFdz = gradient(Fdet(ii), median(diff(z_m(ii))));
    keep = dFdz > -0.2*median(abs(dFdz));  % tune threshold
    if nnz(keep) >= minEpochPts
        ii = ii(keep);
    end

    zii = z_m(ii);
    Fii = F_N(ii);

    % Robust line fit
    if doRobust
        b = robustfit(zii, Fii); % F = b(1) + b(2)*z
        b0 = b(1); kk = b(2);
    else
        p = polyfit(zii, Fii, 1);
        kk = p(1); b0 = p(2);
    end

    % R^2
    Fhat = b0 + kk*zii;
    SSE = sum((Fii - Fhat).^2);
    SST = sum((Fii - mean(Fii)).^2);
    R2  = 1 - SSE/max(SST, eps);

    % Store
    seg.k_Npm(end+1,1) = kk;
    seg.b0(end+1,1)    = b0;
    seg.valleyIdx(end+1,1) = iv;
    seg.epochStartIdx(end+1,1) = min(ii);
    seg.epochEndIdx(end+1,1)   = max(ii);
    seg.R2(end+1,1) = R2;

    prevBound = ip; % next valley search starts after this peak
end
end
