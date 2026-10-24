import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib as mpl

# %% Line plots

def plot_lines_samex(x, y_list, fig_size=(5, 4), 
               colors_list=None, legend_labels=None, grid=True,
               scilimits=None, margins=None, xlims=None, ylims=None, xscale=None, yscale=None, 
               xlabel=None, ylabel=None, title=None, fig_path=None, transparent=True, **kwargs):
    
    # Set defaults
    if colors_list is None: colors_list = []
    if legend_labels is None: legend_labels = []

    # Extend lists to match y_list length
    colors_list = colors_list + ['k'] * (len(y_list) - len(colors_list))
    legend_labels = legend_labels + [None] * (len(y_list) - len(legend_labels))

    # Plotting
    fig, ax = plt.subplots(figsize=fig_size)

    for i, y in enumerate(y_list):
        ax.plot(x, y, color=colors_list[i], label=legend_labels[i], **kwargs)

    if any(label is not None for label in legend_labels):
        ax.legend(loc='best', fontsize=10)
    if grid: ax.grid(alpha=0.5)
    if scilimits: ax.ticklabel_format(style='sci', axis='y', scilimits=scilimits)
    if margins: ax.margins(*margins)
    if xlims: ax.set_xlim(xlims)
    if ylims: ax.set_ylim(ylims)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent)

    return fig, ax

def plot_lines_vstack(x, y_list, fig_size=(5, 6), sharex=True, 
                       colors_list=None, grid=True, 
                       scilimits=(-3,3), margins=None, xlims=None, ylims_list=None, 
                       xlabel=None, ylabels_list=None, title=None, 
                       fig_path=None, transparent=True, 
                       **kwargs):
    
    if len(y_list) == 0:
        return None, None
    
    # Set defaults
    if colors_list is None: colors_list = ['k']*len(y_list)
    if ylims_list is None: ylims_list = []
    if ylabels_list is None: ylabels_list = []

    # Extend lists to match y_list length
    colors_list = colors_list + ['k'] * (len(y_list) - len(colors_list))
    ylims_list = ylims_list + [None] * (len(y_list) - len(ylims_list))
    ylabels_list = ylabels_list + [None] * (len(y_list) - len(ylabels_list))

    # Plotting
    fig, axes = plt.subplots(len(y_list), 1, figsize=fig_size, sharex=sharex)
    axes = np.atleast_1d(axes)
    for i, y in enumerate(y_list):
        ax = axes[i]
        ax.plot(x, y, color=colors_list[i], **kwargs)

        if grid: ax.grid(alpha=0.5)
        if scilimits: ax.ticklabel_format(style='sci', axis='y', scilimits=scilimits)
        if margins: ax.margins(*margins)
        if xlims: ax.set_xlim(xlims)
        if ylims_list[i] is not None: ax.set_ylim(ylims_list[i])
        if xlabel and not sharex: ax.set_xlabel(xlabel)
        if ylabels_list[i] is not None: ax.set_ylabel(ylabels_list[i])
    if xlabel: axes[-1].set_xlabel(xlabel)
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, axes


def plot_inference_params(
    time_midpoints,
    param_names, param_labels,
    params_fitted, bootstrap_stds,
    bootstrap_ci_lowers, bootstrap_ci_uppers,
    param_colors,
    metrics_plot=None, metric_values=None,
    metric_colors=None, metric_labels=None, metric_limits=None,
    param_limits=None,
    title=None,
    **plot_lines_kwargs,
):
    """
    Plot inferred spring-pendulum parameters over time intervals with bootstrap
    uncertainty, optionally combined with validation metrics.

    Parameters
    ----------
    time_midpoints : ndarray, shape (n_intervals,)
        Mid-point of each time interval (x-axis).
    param_names : list of str
        Ordered parameter names to plot (already filtered).
    param_labels : dict
        Maps parameter name -> LaTeX label string for the y-axis.
    params_fitted : ndarray, shape (n_params, n_intervals)
        Fitted parameter values.
    bootstrap_stds : ndarray, shape (n_params, n_intervals)
        Bootstrap standard deviations used for error bars.
    bootstrap_ci_lowers, bootstrap_ci_uppers : ndarray, shape (n_params, n_intervals)
        95% confidence interval bounds used for shaded fill.
    param_colors : dict
        Maps parameter name -> (point_color, fill_color) tuple.
    metrics_plot : list of str, optional
        Validation metric keys to append below param axes (e.g. ['r2_scores']).
    metric_values : ndarray, shape (n_metrics, n_intervals), optional
        Metric values corresponding to metrics_plot.
    metric_colors : dict, optional
        Maps metric key -> matplotlib color.
    metric_labels : dict, optional
        Maps metric key -> y-axis label string.
    metric_limits : dict, optional
        Maps metric key -> (lo, hi) or None.
    param_limits : dict, optional
        Maps parameter name -> (lo, hi) or None.
    title : str, optional
        Figure suptitle.
    **plot_lines_kwargs
        Forwarded to plot_lines_vstack (e.g. plot_lines_config).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    """
    if metrics_plot is None:
        metrics_plot = []
    if metric_values is None:
        metric_values = np.empty((0, len(time_midpoints)))
    if metric_colors is None:
        metric_colors = {}
    if metric_labels is None:
        metric_labels = {}
    if metric_limits is None:
        metric_limits = {}
    if param_limits is None:
        param_limits = {}

    n_params = len(param_names)

    # Build per-row style lists
    colors_list   = [param_colors[p][0] for p in param_names] + [metric_colors.get(m, 'r') for m in metrics_plot]
    ylabels_list  = [param_labels.get(name, name) for name in param_names] + [metric_labels.get(m, m) for m in metrics_plot]
    ylims_list    = [param_limits.get(name, None) for name in param_names] + [metric_limits.get(m, None) for m in metrics_plot]

    data = np.vstack((params_fitted, metric_values)) if len(metrics_plot) > 0 else params_fitted

    fig, axes = plot_lines_vstack(
        time_midpoints, data,
        colors_list=colors_list,
        ylabels_list=ylabels_list,
        ylims_list=ylims_list,
        title=title,
        **plot_lines_kwargs,
    )

    # Add bootstrap error bars and CI shading for parameter axes
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        if np.isnan(params_fitted[i]).any():
            continue
        ax.errorbar(time_midpoints, params_fitted[i], yerr=bootstrap_stds[i],
                    color=ax.lines[0].get_color(), lw=0, elinewidth=1, capsize=2,
                    zorder=-1, label=r'$\pm$ std')
        ax.fill_between(time_midpoints, bootstrap_ci_lowers[i], bootstrap_ci_uppers[i],
                        color=param_colors[param_name][1], alpha=0.3, zorder=-2, label='95% CI')
        ax.legend(loc='best', fontsize=8)

    # Style metric axes
    for j, metric_name in enumerate(metrics_plot):
        ax = axes[n_params + j]
        if ax.lines:
            ax.lines[0].remove()
        ax.plot(time_midpoints, metric_values[j], linewidth=1, marker='^', markersize=5,
                color=metric_colors.get(metric_name, 'r'))
        if metric_name == 'mse_scores':
            ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        if metric_name == 'r2_scores':
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)

    # Resize figure height based on number of rows
    fig_height = len(axes) * 1.5 + 0.5 if len(axes) < 4 else len(axes) * 1.25 + 0.5
    fig.set_size_inches(4, fig_height)
    fig.tight_layout()

    return fig, axes


def plot_lines_hstack(x_list, y, fig_size=(10, 3), sharey=True, 
                      colors_list=None, grid=True, 
                      scilimits=(-3,3), margins=None, ylims=None, xlims_list=None, 
                      ylabel=None, xlabels_list=None, title=None, 
                      fig_path=None, transparent=True, 
                      **kwargs):
    
    if len(x_list) == 0:
        return None, None

    # Set defaults
    if colors_list is None: colors_list = ['k']*len(x_list)
    if xlims_list is None: xlims_list = []
    if xlabels_list is None: xlabels_list = []

    # Extend lists to match x_list length
    colors_list = colors_list + ['k'] * (len(x_list) - len(colors_list))
    xlims_list = xlims_list + [None] * (len(x_list) - len(xlims_list))
    xlabels_list = xlabels_list + [None] * (len(x_list) - len(xlabels_list))

    # Plotting
    fig, axes = plt.subplots(1, len(x_list), figsize=fig_size, sharey=sharey)
    axes = np.atleast_1d(axes)
    for i, x in enumerate(x_list):
        ax = axes[i]
        ax.plot(x, y, color=colors_list[i], **kwargs)

        if grid: ax.grid(alpha=0.5)
        if scilimits: ax.ticklabel_format(style='sci', axis='x', scilimits=scilimits)
        if margins: ax.margins(*margins)
        if ylims: ax.set_ylim(ylims)
        if xlims_list[i] is not None: ax.set_xlim(xlims_list[i])
        if ylabel and not sharey: ax.set_ylabel(ylabel)
        if xlabels_list[i] is not None: ax.set_xlabel(xlabels_list[i])
    if ylabel: axes[0].set_ylabel(ylabel)
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, axes

#%% Scatter plots

def scatter_hstack(x_list, y, fig_size=(8, 3), sharey=True, 
                    marker_color='k', marker_size=1, marker_alpha=0.5,
                    grid=True, scilimits=(-3, 3), margins=None, 
                    ylims=None, xlims_list=None, 
                    ylabel=None, xlabels_list=None, title=None, 
                    fig_path=None, transparent=True, 
                    cbar_config=None, **kwargs):

    if len(x_list) == 0:
        return None, None
    
    # Set defaults
    if xlims_list is None: xlims_list = []
    if xlabels_list is None: xlabels_list = []
    if cbar_config is None: cbar_config = {}

    # Extend lists
    xlims_list = xlims_list + [None] * (len(x_list) - len(xlims_list))
    xlabels_list = xlabels_list + [None] * (len(x_list) - len(xlabels_list))

    # Plotting
    fig, axes = plt.subplots(1, len(x_list), figsize=fig_size, sharey=sharey)
    axes = np.atleast_1d(axes)

    for i, x in enumerate(x_list):
        ax = axes[i]

        if 'vals' in cbar_config and len(cbar_config['vals']) >= len(x):
            vals = cbar_config['vals']
            cmap = cbar_config.get('cmap', 'viridis')
            label = cbar_config.get('label', 'Colorbar')
            cbar_scilimits = cbar_config.get('scilimits', (-3, 3))

            sc = ax.scatter(x, y, c=vals, cmap=cmap,
                            s=marker_size, alpha=marker_alpha, **kwargs)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(label)
            cbar.formatter = mpl.ticker.ScalarFormatter()
            cbar.formatter.set_powerlimits(cbar_scilimits)
            cbar.update_ticks()
        else:
            ax.scatter(x, y, color=marker_color,
                       s=marker_size, alpha=marker_alpha, **kwargs)

        if grid: ax.grid(alpha=0.5)
        if scilimits: ax.ticklabel_format(style='sci', axis='x', scilimits=scilimits)
        if margins: ax.margins(*margins)
        if ylims: ax.set_ylim(ylims)
        if xlims_list[i] is not None: ax.set_xlim(xlims_list[i])
        if ylabel and not sharey: ax.set_ylabel(ylabel)
        if xlabels_list[i] is not None: ax.set_xlabel(xlabels_list[i])

    if ylabel: axes[0].set_ylabel(ylabel)
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, axes

def scatter_vstack(x, y_list, fig_size=(5, 6), sharex=True, 
                   marker_color='k', marker_size=1, marker_alpha=0.5,
                   grid=True, scilimits=(-3, 3), margins=None, 
                   xlims=None, ylims_list=None, 
                   xlabel=None, ylabels_list=None, title=None, 
                   fig_path=None, transparent=True, 
                   cbar_config=None, **kwargs):
    
    if len(y_list) == 0:
        return None, None
    
    # Set defaults
    if ylims_list is None: ylims_list = []
    if ylabels_list is None: ylabels_list = []
    if cbar_config is None: cbar_config = {}

    # Extend lists
    ylims_list = ylims_list + [None] * (len(y_list) - len(ylims_list))
    ylabels_list = ylabels_list + [None] * (len(y_list) - len(ylabels_list))

    # Plotting
    fig, axes = plt.subplots(len(y_list), 1, figsize=fig_size, sharex=sharex)
    axes = np.atleast_1d(axes)

    for i, y in enumerate(y_list):
        ax = axes[i]

        if 'vals' in cbar_config and len(cbar_config['vals']) >= len(x):
            vals = cbar_config['vals']
            cmap = cbar_config.get('cmap', 'viridis')
            label = cbar_config.get('label', 'Colorbar')
            cbar_scilimits = cbar_config.get('scilimits', (-3, 3))

            sc = ax.scatter(x, y, c=vals, cmap=cmap,
                            s=marker_size, alpha=marker_alpha, **kwargs)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(label)
            cbar.formatter = mpl.ticker.ScalarFormatter()
            cbar.formatter.set_powerlimits(cbar_scilimits)
            cbar.update_ticks()
        else:
            ax.scatter(x, y, color=marker_color,
                       s=marker_size, alpha=marker_alpha, **kwargs)

        if grid: ax.grid(alpha=0.5)
        if scilimits: ax.ticklabel_format(style='sci', axis='y', scilimits=scilimits)
        if margins: ax.margins(*margins)
        if xlims: ax.set_xlim(xlims)
        if ylims_list[i] is not None: ax.set_ylim(ylims_list[i])
        if xlabel and not sharex: ax.set_xlabel(xlabel)
        if ylabels_list[i] is not None: ax.set_ylabel(ylabels_list[i])

    if xlabel: axes[-1].set_xlabel(xlabel)
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, axes


def scatter3D(X, fig_size=(6, 5), marker_color='k', marker_size=1, marker_alpha=0.5, 
              cbar_config=None, view_angles=(30, 45), ax_labels=None, ax_lims=None, 
              title=None, fig_path=None, transparent=True, **kwargs):
    
    # Set defaults
    if cbar_config is None: cbar_config = {}
    if ax_labels is None: ax_labels = []
    if ax_lims is None: ax_lims = []

    # Extend lists to have at least 3 elements
    ax_labels = ax_labels + [''] * (3 - len(ax_labels))
    ax_lims = ax_lims + [None] * (3 - len(ax_lims))
    
    # Plotting
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=marker_color, s=marker_size, alpha=marker_alpha, **kwargs)
    
    # Set colormap if provided
    if 'vals' in cbar_config and len(cbar_config['vals']) >= X.shape[0]:
        vals = cbar_config['vals']
        cmap = cbar_config.get('cmap', 'viridis')
        label = cbar_config.get('label', 'Colorbar')
        scilimits = cbar_config.get('scilimits', (-3, 3))
        
        ax.clear()
        scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=vals[:X.shape[0]], cmap=cmap,
                             s=marker_size, alpha=marker_alpha, **kwargs)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(label)

        # Set scientific formatting for colorbar
        cbar.formatter = mpl.ticker.ScalarFormatter()
        cbar.formatter.set_powerlimits(scilimits)
        cbar.update_ticks()
    
    # Set axis view angle, limits and labels
    if view_angles: ax.view_init(elev=view_angles[0], azim=view_angles[1])
    if ax_labels[0]: ax.set_xlabel(ax_labels[0])
    if ax_labels[1]: ax.set_ylabel(ax_labels[1])
    if ax_labels[2]: ax.set_zlabel(ax_labels[2])
    if ax_lims[0]: ax.set_xlim(ax_lims[0])
    if ax_lims[1]: ax.set_ylim(ax_lims[1])
    if ax_lims[2]: ax.set_zlim(ax_lims[2])

    if title: fig.suptitle(title, y=0.95)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, ax


def plot_scatter_with_surface(
    x_data, y_data, z_data,
    surface_fn,
    x_limits, y_limits,
    z_limits=None,
    meshgrid_size=50,
    piecewise=False,
    breakpoint_val=None,
    piecewise_colors_list=None,
    surface_alpha=0.3,
    ax_labels=None,
    title=None,
    **scatter3D_kwargs
):
    """
    Create a 3D scatter plot with a fitted surface overlay.

    Parameters
    ----------
    x_data, y_data, z_data : array-like (1D)
        Scattered data points for the three axes.
    surface_fn : callable
        ``surface_fn(X_mesh, Y_mesh) -> Z_mesh``.  Receives 2D meshgrid arrays
        and returns the corresponding surface values.  The caller is responsible
        for binding any additional feature means or physics parameters into this
        closure before passing it.
    x_limits, y_limits : tuple (min, max)
        Axis limits used both for the meshgrid extent and for ax.set_xlim/ylim.
    z_limits : tuple (min, max) or None
        If given, applied via ax.set_zlim.
    meshgrid_size : int
        Number of points along each mesh axis (default 50).
    piecewise : bool
        If True, split the surface at ``breakpoint_val`` along the x axis,
        colouring each half with the corresponding entry in ``piecewise_colors_list``.
    breakpoint_val : float or None
        x-value at which to split the surface when ``piecewise=True``.
    piecewise_colors_list : list or None
        Colours for the two surface halves (or the single surface).
        Defaults to ``['C0', 'C1']``.
    surface_alpha : float
        Transparency of the surface(s) (default 0.3).
    ax_labels : list of str or None
        Axis labels forwarded to ``scatter3D``.
    title : str or None
        Figure title forwarded to ``scatter3D``.
    **scatter3D_kwargs
        Any additional keyword arguments forwarded to :func:`scatter3D`
        (e.g. ``fig_size``, ``view_angles``, ``marker_size``, ``cbar_config``).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D
    """
    if piecewise_colors_list is None:
        piecewise_colors_list = ['C0', 'C1']

    # 1. Draw scatter via existing scatter3D helper
    X = np.column_stack([x_data, y_data, z_data])
    fig, ax = scatter3D(X, ax_labels=ax_labels, title=title, **scatter3D_kwargs)

    # 2. Build meshgrid and evaluate surface
    X_mesh, Y_mesh = np.meshgrid(
        np.linspace(x_limits[0], x_limits[1], meshgrid_size),
        np.linspace(y_limits[0], y_limits[1], meshgrid_size),
    )
    Z_mesh = surface_fn(X_mesh, Y_mesh)

    # 3. Draw surface (piecewise split along x, or single)
    if piecewise and breakpoint_val is not None:
        masks = [X_mesh <= breakpoint_val, X_mesh >= breakpoint_val]
        for mask, color in zip(masks, piecewise_colors_list):
            if np.any(mask):
                ax.plot_surface(
                    np.where(mask, X_mesh, np.nan),
                    np.where(mask, Y_mesh, np.nan),
                    np.where(mask, Z_mesh, np.nan),
                    alpha=surface_alpha, color=color,
                )
    else:
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=surface_alpha, color=piecewise_colors_list[0])

    # 4. Apply explicit axis limits (after surface so limits encompass everything)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    if z_limits is not None:
        ax.set_zlim(z_limits)

    return fig, ax


def scatter3D_projections(X, fig_size=(12, 3.6), projections=None, marker_color='k', 
                         marker_size=1, marker_alpha=0.5, cbar_config=None, ax_labels=None, 
                         ax_lims=None, title=None, fig_path=None, transparent=True, **kwargs):
    
    # Set defaults
    if projections is None: projections = [(0, 1), (0, 2), (1, 2)]
    if cbar_config is None: cbar_config = {}
    if ax_labels is None: ax_labels = []
    if ax_lims is None: ax_lims = []
    
    # Extend lists to cover all possible projection indices
    max_idx = max([max(proj) for proj in projections]) if projections else 2
    ax_labels = ax_labels + [''] * (max_idx + 1 - len(ax_labels))
    ax_lims = ax_lims + [None] * (max_idx + 1 - len(ax_lims))
    
    # Plotting
    fig, axes = plt.subplots(1, len(projections), figsize=fig_size)
    axes = np.atleast_1d(axes)

    # Scatter plots for each projection
    for i, (x_idx, y_idx) in enumerate(projections):
        ax = axes[i]
        scatter = ax.scatter(X[:, x_idx], X[:, y_idx], c=marker_color, s=marker_size, alpha=marker_alpha, **kwargs)

        # Set colormap if provided
        if 'vals' in cbar_config and len(cbar_config['vals']) >= X.shape[0]:
            vals = cbar_config['vals']
            cmap = cbar_config.get('cmap', 'viridis')
            label = cbar_config.get('label', 'Colorbar')
            scilimits = cbar_config.get('scilimits', (-3, 3))
            
            ax.clear()
            scatter = ax.scatter(X[:,x_idx], X[:,y_idx], c=vals[:X.shape[0]], 
                               s=marker_size, alpha=marker_alpha, cmap=cmap, **kwargs)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(label)

            # Set scientific formatting for colorbar
            cbar.formatter = mpl.ticker.ScalarFormatter()
            cbar.formatter.set_powerlimits(scilimits)
            cbar.update_ticks()
        
        # Set axis labels and limits
        if ax_labels[x_idx]: ax.set_xlabel(ax_labels[x_idx])
        if ax_labels[y_idx]: ax.set_ylabel(ax_labels[y_idx])
        if ax_lims[x_idx]: ax.set_xlim(ax_lims[x_idx])
        if ax_lims[y_idx]: ax.set_ylim(ax_lims[y_idx])

        ax.grid(alpha=0.5)
        ax.set_aspect('equal')
    
    if title: fig.suptitle(title, y=0.95)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent, bbox_inches='tight')
    return fig, axes

# %% Specialized plots

def plot_validation_curve(cv_results, fig_size=(5, 4), plot_train=True, plot_test=True, plot_std=True, plot_best=True,
                          colors=None, labels=None, markers=None, linewidth=1.5, markersize=5, grid=True, scilimits=None, 
                          margins=None, xlims=None, ylims=None, xscale=None, yscale=None, xlabel=None, ylabel=None, 
                          title=None, fig_path=None, transparent=True):
    
    fig, ax = plt.subplots(figsize=fig_size)

    if colors == None: colors = {'train': 'b', 'test': 'r'}
    if labels == None: labels = {'train': 'Train', 'test': 'Test'}
    if markers == None: markers = {'train': 'o', 'test': 's'}

    if plot_train and len(cv_results.get('mean_train_score', np.array([]))) == len(cv_results.get('param_grid')):
        ax.plot(cv_results['param_grid'], cv_results['mean_train_score'], color=colors['train'], label=labels['train'], linewidth=linewidth, marker=markers['train'], markersize=markersize)
        if plot_std:
            ax.fill_between(cv_results['param_grid'], cv_results['mean_train_score'] - cv_results['std_train_score'], cv_results['mean_train_score'] + cv_results['std_train_score'], color=colors['train'], alpha=0.2)
    if plot_test and len(cv_results.get('mean_test_score', np.array([]))) == len(cv_results.get('param_grid')):
        ax.plot(cv_results['param_grid'], cv_results['mean_test_score'], color=colors['test'], label=labels['test'], linewidth=linewidth, marker=markers['test'], markersize=markersize)
        if plot_std:
            ax.fill_between(cv_results['param_grid'], cv_results['mean_test_score'] - cv_results['std_test_score'], cv_results['mean_test_score'] + cv_results['std_test_score'], color=colors['test'], alpha=0.2)
    if plot_best:
        best_index = np.argmin(cv_results['mean_test_score'])
        best_param = cv_results['param_grid'][best_index]
        best_score = cv_results['mean_test_score'][best_index]
        ax.scatter(best_param, best_score, color=colors['test'], marker='*', s=150, edgecolor='k', zorder=10)
        ax.text(best_param, best_score + np.ptp(ax.get_ylim())*0.05, f'({best_param:.2g}, {best_score:.2g})', fontsize=8, ha='right', va='bottom', color=colors['test'], bbox=dict(facecolor='white', edgecolor=colors['test'], alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax.legend(loc='best', fontsize = 10)

    if grid: ax.grid(alpha=0.5)
    if scilimits: ax.ticklabel_format(style='sci', axis='y', scilimits=scilimits)
    if margins: ax.margins(*margins)
    if xlims: ax.set_xlim(xlims)
    if ylims: ax.set_ylim(ylims)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200, transparent=transparent)

    return fig, ax

def plot_heatmap(heatmap_X, heatmap_Y, heatmap_Z, cmap_config={}, fig_size=(6,4), x_label=None, y_label=None, title=None, fig_path=None, **kwargs):

    if not isinstance(cmap_config, dict): cmap_config = {}
    cmap = cmap_config.get('cmap', 'plasma')
    norm = cmap_config.get('norm', None)
    cbar_label = cmap_config.get('label', None)
    scilimits = cmap_config.get('scilimits', None)

    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.pcolormesh(heatmap_X, heatmap_Y, heatmap_Z, shading='auto', cmap=cmap, norm=norm, **kwargs)
    cbar = fig.colorbar(cax, ax=ax)
    if cbar_label: cbar.set_label(cbar_label, fontsize=10)
    if scilimits is not None and (norm is None or not isinstance(norm, plt.matplotlib.colors.LogNorm)):
        cbar.formatter.set_powerlimits(scilimits)
        cbar.formatter.set_scientific(True)
        cbar.update_ticks()

    if x_label: ax.set_xlabel(x_label, fontsize=10)
    if y_label: ax.set_ylabel(y_label, fontsize=10)
    
    if title: ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if fig_path: fig.savefig(fig_path, dpi=200)
    return fig, ax, cbar


# %% Animations

def animate_turn(fig, ax, fig_path, view_angles = (30,45), n_frames = 180, frame_interval = 50):
    if fig_path is None: return
    def update_frame(frame):
        ax.view_init(elev = view_angles[0], azim = view_angles[1] + frame)  # Change azimuth
        return ax,
    fig.tight_layout()
    ani = FuncAnimation(fig, update_frame, frames=np.linspace(0, 360, n_frames), interval = frame_interval)
    ani.save(str(fig_path).split('.')[0] + '.mp4', writer="ffmpeg", fps = 12, dpi = 200, savefig_kwargs={'transparent': True})
    ani.event_source.stop()


# %% Extra plotting functions

# Helper function to plot a group of related tangle moments
def plot_moment_group(t_series, moment_data, moment_lims, xp_vec_series, 
                      group_config, base_config, exp_key, data_label, 
                      save_path=None, close_plot=False):
    """
    Plot a group of related moments.
    
    Parameters
    ----------
    t_series : array
        Time series data
    moment_data : dict
        Dictionary containing all moment data
    moment_lims : dict
        Dictionary containing y-axis limits for each moment
    xp_vec_series : array
        Piston trajectory data (shape: [n_times, 3])
    group_config : dict
        Configuration for this specific moment group with keys:
        - 'keys': list of moment keys to plot
        - 'labels': list of y-axis labels
        - 'fig_size': tuple of figure dimensions
        - 'add_piston': bool or list of bools (optional, default False)
    base_config : dict
        Base plotting configuration
    exp_key : str
        Experiment key
    data_label : str
        Data label
    save_path : Path, optional
        Path to save figure
    close_plot : bool
        Whether to close the plot after saving
    
    Returns
    -------
    fig, axes : tuple or (None, None)
        Figure and axes objects, or (None, None) if data missing
    """
    keys = group_config['keys']
    
    # Check if all required data exists
    if not all([moment_data.get(key) is not None for key in keys]):
        return None, None
    
    # Prepare data
    x_plot = np.column_stack([moment_data[key] for key in keys])
    
    # Update config
    config = base_config.copy()
    config.update({
        'ylims_list': [moment_lims.get(key) for key in keys],
        'ylabels_list': group_config['labels'],
        'fig_size': group_config.get('fig_size', base_config['fig_size'])
    })
    
    # Create plot
    title = f"{data_label} ({exp_key})"
    fig, axes = plot_lines_vstack(t_series, x_plot.T, **config, title=title)
    
    # Add piston trajectory if requested
    add_piston = group_config.get('add_piston', False)
    if add_piston:
        # Handle both bool and list of bools
        if isinstance(add_piston, bool):
            add_piston_list = [add_piston] * len(keys)
        else:
            add_piston_list = add_piston
        
        # Map position keys to spatial component indices
        spatial_component_map = {'x': 0, 'y': 1, 'z': 2}
        
        for ax, key, add_piston_to_axis in zip(axes, keys, add_piston_list):
            if add_piston_to_axis and key in spatial_component_map:
                component_idx = spatial_component_map[key]
                ax.plot(t_series, xp_vec_series[:, component_idx], color='r', linewidth=1, zorder=10)
                ax.text(t_series[-1], xp_vec_series[:, component_idx].min(), 'piston', 
                       color='r', fontsize=9, ha='right', va='top', zorder=10)
    
    # Save and close
    if save_path:
        fig.savefig(save_path, dpi=200, transparent=True, bbox_inches='tight')
    if close_plot:
        plt.close(fig)
    
    return fig, axes
