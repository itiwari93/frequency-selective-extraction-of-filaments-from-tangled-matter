"""Physics computations for driven spring pendulum system."""

import numpy as np

# Labels

SPRING_PENDULUM_LABELS = {
    # Model parameters
    'm': '$m$',
    'k': '$k$',
    'gamma': '$\\gamma$',
    'c': '$c$',
    'L0': '$L_0$',
    'g': '$g$',
    'A_p': '$A_p$',
    'Omega': '$\\Omega$',

    # Dimensional analysis quantities
    'damping_ratio': '$\\zeta$',
    'natural_frequency': '$\\omega_0$',
    'equilibrium_length': '$L_{eq}$',
    'equivalent_forcing_amplitude': '$F_0$',
    'forcing_phase': '$\\delta$',
    'time_scale': '$T$',
    'length_scale': '$L$',
    'forcing_frequency': '$\\omega$',

    # Resonance frequencies
    'omega_r': '$\\omega_r^*$',
    'omega_theta': '$\\omega_\\theta^*$',

    # Trajectory statistics
    'ma_pos': '$\\langle |\\mathbf{x}-\\mathbf{x}_0| \\rangle$',
    'ma_vel': '$\\langle |\\mathbf{v}| \\rangle$',
    'ma_acc': '$\\langle |\\mathbf{a}| \\rangle$',
    'rms_pos': '$\\sqrt{\\langle (\\mathbf{x}-\\mathbf{x}_0)^2 \\rangle}$',
    'rms_vel': '$\\sqrt{\\langle \\mathbf{v}^2 \\rangle}$',
    'rms_acc': '$\\sqrt{\\langle \\mathbf{a}^2 \\rangle}$',
    'ms_pos': '$\\langle (\\mathbf{x}-\\mathbf{x}_0)^2 \\rangle$',
    'ms_vel': '$\\langle \\mathbf{v}^2 \\rangle$',
    'ms_acc': '$\\langle \\mathbf{a}^2 \\rangle$',

    'ma_vel_rad': '$\\langle |\\mathbf{v}_{\\parallel}| \\rangle$',
    'ma_vel_tan': '$\\langle |\\mathbf{v}_{\\perp}| \\rangle$',
    'ma_acc_rad': '$\\langle |\\mathbf{a}_{\\parallel}| \\rangle$',
    'ma_acc_tan': '$\\langle |\\mathbf{a}_{\\perp}| \\rangle$',
    'rms_vel_rad': '$\\sqrt{\\langle \\mathbf{v}_{\\parallel}^2 \\rangle}$',
    'rms_vel_tan': '$\\sqrt{\\langle \\mathbf{v}_{\\perp}^2 \\rangle}$',
    'rms_acc_rad': '$\\sqrt{\\langle \\mathbf{a}_{\\parallel}^2 \\rangle}$',
    'rms_acc_tan': '$\\sqrt{\\langle \\mathbf{a}_{\\perp}^2 \\rangle}$',
    'ms_vel_rad': '$\\langle \\mathbf{v}_{\\parallel}^2 \\rangle$',
    'ms_vel_tan': '$\\langle \\mathbf{v}_{\\perp}^2 \\rangle$',
    'ms_acc_rad': '$\\langle \\mathbf{a}_{\\parallel}^2 \\rangle$',
    'ms_acc_tan': '$\\langle \\mathbf{a}_{\\perp}^2 \\rangle$',

    'ma_pos_rel': '$\\langle |\\mathbf{x}_{r}-\\mathbf{x}_0| \\rangle$',
    'ma_vel_rel': '$\\langle |\\mathbf{v}_{r}| \\rangle$',
    'ma_acc_rel': '$\\langle |\\mathbf{a}_{r}| \\rangle$',
    'rms_pos_rel': '$\\sqrt{\\langle (\\mathbf{x}_{r}-\\mathbf{x}_0)^2 \\rangle}$',
    'rms_vel_rel': '$\\sqrt{\\langle \\mathbf{v}_{r}^2 \\rangle}$',
    'rms_acc_rel': '$\\sqrt{\\langle \\mathbf{a}_{r}^2 \\rangle}$',
    'ms_pos_rel': '$\\langle (\\mathbf{x}_{r}-\\mathbf{x}_0)^2 \\rangle$',
    'ms_vel_rel': '$\\langle \\mathbf{v}_{r}^2 \\rangle$',
    'ms_acc_rel': '$\\langle \\mathbf{a}_{r}^2 \\rangle$',

    'ma_vel_rel_rad': '$\\langle |\\mathbf{v}_{r,\\parallel}| \\rangle$',
    'ma_vel_rel_tan': '$\\langle |\\mathbf{v}_{r,\\perp}| \\rangle$',
    'rms_vel_rel_rad': '$\\sqrt{\\langle \\mathbf{v}_{r,\\parallel}^2 \\rangle}$',
    'rms_vel_rel_tan': '$\\sqrt{\\langle \\mathbf{v}_{r,\\perp}^2 \\rangle}$',
    'ms_vel_rel_rad': '$\\langle \\mathbf{v}_{r,\\parallel}^2 \\rangle$',
    'ms_vel_rel_tan': '$\\langle \\mathbf{v}_{r,\\perp}^2 \\rangle$',
    'ma_acc_rel_rad': '$\\langle |\\mathbf{a}_{r,\\parallel}| \\rangle$',
    'ma_acc_rel_tan': '$\\langle |\\mathbf{a}_{r,\\perp}| \\rangle$',
    'rms_acc_rel_rad': '$\\sqrt{\\langle \\mathbf{a}_{r,\\parallel}^2 \\rangle}$',
    'rms_acc_rel_tan': '$\\sqrt{\\langle \\mathbf{a}_{r,\\perp}^2 \\rangle}$',
    'ms_acc_rel_rad': '$\\langle \\mathbf{a}_{r,\\parallel}^2 \\rangle$',
    'ms_acc_rel_tan': '$\\langle \\mathbf{a}_{r,\\perp}^2 \\rangle$',
}



# Dimensional analysis computations

def damping_ratio(gamma, c, k, m):
    """
    Compute the damping ratio zeta.
    """
    zeta = (gamma + c) / (2 * np.sqrt(k * m))
    return zeta


def natural_frequency(k, m):
    """
    Compute the natural frequency omega_0.
    """
    omega_0 = np.sqrt(k / m)
    return omega_0


def equilibrium_position(L0, m, g, k):
    """
    Compute the equilibrium position x_eq.
    """
    x_eq = (L0 + m*g/k)*np.array([0, 0, -1])
    return x_eq


def equivalent_forcing_amplitude(k, A_p, c, Omega):
    """
    Compute the equivalent forcing amplitude F_0.
    """
    F_0 = k*A_p * np.sqrt(1 + (c*Omega/k)**2)
    return F_0


def forcing_phase(c, Omega, k):
    """
    Compute the forcing phase phi.
    """
    phi = np.arctan2(c*Omega, k)
    return phi


def spring_pendulum_dimensional_analysis(params):
    """
    Perform dimensional analysis for the spring pendulum system.

    Parameters
    ----------
    params : dict
        Contains parameters 'm', 'k', 'gamma', 'c', 'L0', 'g', 'A_p', 'Omega'

    Returns
    -------
    dict
        Dictionary containing computed dimensional analysis quantities
    """
    m = params['m']
    k = params['k']
    gamma = params['gamma']
    c = params['c']
    L0 = params['L0']
    g = params['g']
    A_p = params['A_p']
    Omega = params['Omega']
    
    dimensional_analysis = {}
    
    dimensional_analysis['damping_ratio'] = damping_ratio(gamma, c, k, m)
    dimensional_analysis['natural_frequency'] = natural_frequency(k, m)
    # dimensional_analysis['equilibrium_position'] = equilibrium_position(L0, m, g, k)
    dimensional_analysis['equilibrium_length'] = np.linalg.norm(equilibrium_position(L0, m, g, k))
    dimensional_analysis['equivalent_forcing_amplitude'] = equivalent_forcing_amplitude(k, A_p, c, Omega)
    dimensional_analysis['forcing_phase'] = forcing_phase(c, Omega, k)

    dimensional_analysis['time_scale'] = 1 / dimensional_analysis['natural_frequency']
    dimensional_analysis['length_scale'] = dimensional_analysis['equivalent_forcing_amplitude']/k
    dimensional_analysis['forcing_frequency'] = Omega/dimensional_analysis['natural_frequency']
    
    return dimensional_analysis


# Resonance Frequencies

def radial_resonance(k, m, gamma, c):
    """
    Compute the radial resonance frequency omega_r^*.
    """
    omega_r = np.sqrt(max(k / m - ((gamma + c)**2) / (2 * m**2), 0))
    return omega_r


def angular_resonance(g, L0, m, k):
    """
    Compute the angular parametric resonance frequency omega_theta^*.
    """
    L0_eff = L0 + m * g / k
    omega_theta = 2 * np.sqrt(g / L0_eff)
    return omega_theta


def spring_pendulum_resonance(params):
    """
    Compute radial and angular resonance frequencies of the spring pendulum.
    """
    k = params['k']
    m = params['m']
    gamma = params['gamma']
    c = params['c']
    g = params['g']
    L0 = params['L0']

    omega_r = radial_resonance(k, m, gamma, c)
    omega_theta = angular_resonance(g, L0, m, k)

    resonance_freqs = {
        'omega_r': omega_r,
        'omega_theta': omega_theta
    }

    return resonance_freqs