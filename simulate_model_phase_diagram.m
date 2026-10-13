%% Final reduced TW scan: RMS excursion of a unilateral tether model
%%
% Independent scan coordinates:
%   Gamma_acc = A*omega^2/g
%   chi       = k*L0/(m*g)
%
% Elastic-time variables:
%   tau       = t*sqrt(k/m)
%   eps       = A/L0
%   Omega     = omega*sqrt(m/k)
%   Gamma_acc = eps*chi*Omega^2
%
% Noise convention:
%   sigma_a_nd and tau_corr_nd are fixed in elastic-time units. Thus the
%   noise is held fixed relative to the nondimensional model, rather than
%   representing a fixed dimensional shaker/environmental disturbance.
%   
% Time integration:
%   One midpoint OU acceleration is held over each outer
%   step, which is a piecewise-constant colored-forcing approximation.
%
% Unilateral radial element:
%   When D > 1, the net tension is
%       (D-1) + zeta_rad*V_r.
%   The transmitted tension is max(0,net tension), so the combined
%   elastic-plus-damping element can pull but can never push.

clc;
clear;

%% ---------------- User settings ----------------
n_reps = 10;
rng_seed = 1;

% Geometry and forcing amplitude
eps = 0.15;                       % A/L0

% Physical reference values
L0_ref = 0.05;                    % m
m_ref  = 5e-5;                    % kg
g_ref  = 9.8;                     % m/s^2
A_ref  = eps*L0_ref;              % m

% Dissipation
c_air    = 1.8e-5;                % kg/s
zeta_rad = 0.12;                  % radial damping in elastic-time units

% OU acceleration noise in elastic-time units
sigma_a_nd  = 5e-4;
tau_corr_nd = 5e-2;

% Independent scan axes
Gamma_acc_list = logspace(log10(0.2),log10(160),80);
chi_list = logspace(log10(10),log10(1000),40);

% Integration settings
dtau = 1e-3;
time_dim = 10;                    % fixed physical duration, seconds
X0 = [0,-1,0];
U0 = [0,0,0];

% Output controls
save_workspace = true;
make_quicklook = true;
output_file = 'tw_scan_gammaacc_chi_workspace_final_reduced.mat';
quicklook_file = 'tw_scan_gammaacc_chi_quicklook_final_reduced.png';

%% ---------------- Validate settings ----------------
if tau_corr_nd <= 0
    error('tau_corr_nd must be positive.');
end
if dtau <= 0 || time_dim <= 0
    error('dtau and time_dim must be positive.');
end

%% ---------------- Derived grids ----------------
nA = numel(Gamma_acc_list);
nC = numel(chi_list);
nComb = nC*nA;

k_equiv = chi_list*m_ref*g_ref/L0_ref;
frequency_hz_list = sqrt(Gamma_acc_list*g_ref/A_ref)/(2*pi);

Gamma_list = 1./chi_list;
Tstar_list = sqrt(L0_ref./(g_ref*chi_list));  % sqrt(m/k), seconds
zeta_list = c_air./sqrt(m_ref*k_equiv);

[chi_grid,Gamma_acc_grid] = ndgrid(chi_list,Gamma_acc_list);
Omega_grid = sqrt(Gamma_acc_grid./(eps*chi_grid));

fprintf('Final reduced direct (Gamma_acc, chi) scan\n');
fprintf('  Gamma_acc: %.4g to %.4g\n', ...
    min(Gamma_acc_list),max(Gamma_acc_list));
fprintf('  frequency: %.3f to %.3f Hz for A = %.3f cm\n', ...
    min(frequency_hz_list),max(frequency_hz_list),100*A_ref);
fprintf('  chi: %.4g to %.4g\n',min(chi_list),max(chi_list));
fprintf('  equivalent k: %.4g to %.4g N/m\n', ...
    min(k_equiv),max(k_equiv));
fprintf('  Omega*: %.4g to %.4g over the map\n', ...
    min(Omega_grid,[],'all'),max(Omega_grid,[],'all'));

%% ---------------- Storage ----------------
% Array dimensions: [chi, Gamma_acc, repetition]
std_x_nd = nan(nC,nA,n_reps);
std_y_nd = nan(nC,nA,n_reps);
std_z_nd = nan(nC,nA,n_reps);
com_excursion_rms_nd = nan(nC,nA,n_reps);
com_excursion_rms_mm = nan(nC,nA,n_reps);

%% ---------------- Parallel setup ----------------
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool;
end

% Unique substreams make results independent of parallel scheduling.
streams = parallel.pool.Constant( ...
    @() RandStream('CombRecursive','Seed',rng_seed));

%% ---------------- Main sweep ----------------
for rep = 1:n_reps
    fprintf('Starting repetition %d / %d\n',rep,n_reps);

    std_xyz_lin = nan(nComb,3);
    com_excursion_lin = nan(nComb,1);

    parfor comb_idx = 1:nComb
        [c_idx,a_idx] = ind2sub([nC,nA],comb_idx);

        chi = chi_list(c_idx);
        Gamma_acc = Gamma_acc_list(a_idx);

        Gamma = 1/chi;
        Omega = sqrt(Gamma_acc/(eps*chi));
        zeta = zeta_list(c_idx);
        Tstar = Tstar_list(c_idx);

        tau_end = time_dim/Tstar;
        n_steps = ceil(tau_end/dtau);

        noise_state = streams.Value;
        noise_state.Substream = (rep-1)*nComb+comb_idx;

        state = [X0,U0];

        % Stationary OU initial condition: Var[xi(0)] = 1.
        xi_now = randn(noise_state,1,3);

        % Online component-wise variance (Welford algorithm). This avoids
        % retaining multi-million-sample trajectories on every worker.
        n_samples = 1;
        mean_position = state(1:3);
        position_M2 = zeros(1,3);

        for step_idx = 1:n_steps
            tau_now = (step_idx-1)*dtau;
            h = min(dtau,tau_end-tau_now);

            % Exact OU transition over h, then midpoint/step-average
            % colored acceleration held fixed during this mechanical step.
            ou_alpha = exp(-h/tau_corr_nd);
            ou_beta = sqrt(max(0,1-ou_alpha^2));
            xi_new = ou_alpha*xi_now+ ...
                ou_beta*randn(noise_state,1,3);
            a_noise = sigma_a_nd*0.5*(xi_now+xi_new);

            % nonautonomous RK4 stage times for the moving support.
            k1 = tether_rhs(state,tau_now,Omega,eps, ...
                zeta,zeta_rad,Gamma,a_noise);
            k2 = tether_rhs(state+0.5*h*k1,tau_now+0.5*h, ...
                Omega,eps,zeta,zeta_rad,Gamma,a_noise);
            k3 = tether_rhs(state+0.5*h*k2,tau_now+0.5*h, ...
                Omega,eps,zeta,zeta_rad,Gamma,a_noise);
            k4 = tether_rhs(state+h*k3,tau_now+h,Omega,eps, ...
                zeta,zeta_rad,Gamma,a_noise);

            state = state+(h/6)*(k1+2*k2+2*k3+k4);
            xi_now = xi_new;

            n_samples = n_samples+1;
            delta = state(1:3)-mean_position;
            mean_position = mean_position+delta/n_samples;
            position_M2 = position_M2+ ...
                delta.*(state(1:3)-mean_position);
        end

        component_std = sqrt(position_M2/(n_samples-1));
        std_xyz_lin(comb_idx,:) = component_std;
        com_excursion_lin(comb_idx) = sqrt(sum(component_std.^2));
    end

    std_x_nd(:,:,rep) = reshape(std_xyz_lin(:,1),[nC,nA]);
    std_y_nd(:,:,rep) = reshape(std_xyz_lin(:,2),[nC,nA]);
    std_z_nd(:,:,rep) = reshape(std_xyz_lin(:,3),[nC,nA]);
    com_excursion_rms_nd(:,:,rep) = ...
        reshape(com_excursion_lin,[nC,nA]);
    com_excursion_rms_mm(:,:,rep) = ...
        1e3*L0_ref*com_excursion_rms_nd(:,:,rep);

    fprintf('Finished repetition %d / %d\n',rep,n_reps);
end

%% ---------------- Ensemble averages ----------------
std_x_avg_nd = mean(std_x_nd,3,'omitnan');
std_y_avg_nd = mean(std_y_nd,3,'omitnan');
std_z_avg_nd = mean(std_z_nd,3,'omitnan');
com_excursion_rms_avg_nd = mean(com_excursion_rms_nd,3,'omitnan');
com_excursion_rms_avg_mm = mean(com_excursion_rms_mm,3,'omitnan');

%% ---------------- Save workspace ----------------
if save_workspace
    save(output_file, ...
        'Gamma_acc_list','chi_list','Gamma_acc_grid','chi_grid', ...
        'Omega_grid','Gamma_list','frequency_hz_list','k_equiv', ...
        'Tstar_list','eps','A_ref','L0_ref','m_ref','g_ref', ...
        'zeta_list','zeta_rad','c_air', ...
        'sigma_a_nd','tau_corr_nd','dtau','time_dim', ...
        'X0','U0','n_reps','rng_seed', ...
        'std_x_nd','std_y_nd','std_z_nd', ...
        'com_excursion_rms_nd','com_excursion_rms_mm', ...
        'std_x_avg_nd','std_y_avg_nd','std_z_avg_nd', ...
        'com_excursion_rms_avg_nd','com_excursion_rms_avg_mm', ...
        '-v7.3');
    fprintf('Saved %s\n',output_file);
end

%% ---------------- Quick-look response map ----------------
if make_quicklook
    fig = figure('Color','w','Position',[100 100 900 700]);
    ax = axes(fig);
    surf(ax,Gamma_acc_grid,chi_grid,com_excursion_rms_avg_mm, ...
        'EdgeColor','none');
    view(ax,2);
    set(ax,'XScale','log','YScale','log', ...
        'FontSize',12,'LineWidth',1,'Layer','top');
    colormap(fig,parula(256));
    colorbar_handle = colorbar(ax);
    colorbar_handle.Label.String = 'mean RMS excursion (mm)';
    xlabel(ax,'\Gamma_{acc}=A\omega^2/g');
    ylabel(ax,'\chi=kL_0/(mg)');
    title(ax,'Final reduced unilateral-tether response');
    grid(ax,'on');
    box(ax,'on');
    hold(ax,'on');
    xline(ax,1,'--k','\Gamma_{acc}=1', ...
        'LineWidth',1.2,'LabelVerticalAlignment','bottom');

    exportgraphics(fig,quicklook_file,'Resolution',300);
    fprintf('Saved %s\n',quicklook_file);
end

%% =====================================================================
function derivative = tether_rhs( ...
    state,tau,Omega,eps,zeta,zeta_rad,Gamma,a_noise)

% state = [X Y Z Ux Uy Uz]
position = state(1:3);
velocity = state(4:6);

Ys = eps*sin(Omega*tau);
Ys_d = eps*Omega*cos(Omega*tau);

relative_position = [position(1),position(2)-Ys,position(3)];
D = norm(relative_position);

if D < 1e-14
    rhat = [0,0,0];
else
    rhat = relative_position/D;
end

% A slack connector transmits no force. When geometrically taut, clamp the
% combined elastic-plus-damping tension so it can never become compressive.
tension_nd = 0;
if D > 1
    relative_velocity = [velocity(1),velocity(2)-Ys_d,velocity(3)];
    radial_velocity = dot(relative_velocity,rhat);
    trial_tension_nd = (D-1)+zeta_rad*radial_velocity;
    tension_nd = max(0,trial_tension_nd);
end

acceleration = -tension_nd*rhat-zeta*velocity+ ...
    [0,-Gamma,0]+a_noise;

derivative = [velocity,acceleration];
end
