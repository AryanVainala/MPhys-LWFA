"""
LWFA Theory Utility Module

This module contains functions to calculate theoretical parameters for 
Laser-Wakefield Acceleration (LWFA) simulations.
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, pi

def get_omega_p(n_e):
    """Calculate plasma angular frequency (rad/s)."""
    return np.sqrt(n_e * e**2 / (m_e * epsilon_0))

def get_lambda_p(n_e):
    """Calculate plasma wavelength (m)."""
    return 2 * pi * c / get_omega_p(n_e)

def get_k_p(n_e):
    """Calculate plasma wave number"""
    return get_omega_p(n_e) / c

def get_skin_depth(n_e):
    """Calculate plasma skin depth (m)."""
    return c / get_omega_p(n_e)

def get_omega_l(lambda0):
    """Calculate laser angular frequency (rad/s)."""
    return 2 * pi * c / lambda0

def get_critical_e_density(lambda0):
    """Calculate the crit. e- density that laser freq. ω cannot propagate"""
    omega_l = get_omega_l(lambda0)
    return epsilon_0*m_e*omega_l**2 / e**2

def get_density_ratio(n_e, lambda0):
    """Calculate the ratio of electron density to critical density."""
    n_c = get_critical_e_density(lambda0)
    return n_e / n_c

def get_critical_power(lambda0, n_e):
    """Calculate critical power for relativistic self-focusing (W)."""
    omega_l = get_omega_l(lambda0)
    omega_p = get_omega_p(n_e)
    return 17e9 * (omega_l / omega_p)**2

def get_dephasing_length(a0, lambda0, n_e):
    """Calculate dephasing length in blowout regime (m)."""
    omega_l = get_omega_l(lambda0)
    omega_p = get_omega_p(n_e)
    return (2/3) * (omega_l/omega_p)**2 * np.sqrt(a0) * (c/omega_p)

def get_pump_depletion_length(a0, lambda0, n_e):
    """Calculate pump depletion length in blowout regime (m)."""
    omega_l = get_omega_l(lambda0)
    omega_p = get_omega_p(n_e)
    return (omega_l / omega_p)**2 * a0 * (c / omega_p)

def get_intensity_from_a0(a0, lambda0):
    """Calculate peak laser intensity (W/m^2) from a0."""
    lambda0_um = lambda0 * 1e6
    # I_0[10^18 W/cm^2] = (a0 / (0.85 * lambda0[um]))^2
    I_0_normalized = (a0 / (0.85 * lambda0_um))**2 
    I_0_cgs = I_0_normalized * 1e18 # W/cm^2
    return I_0_cgs

def get_laser_power(a0, lambda0, w0):
    """Calculate total laser power (W) for a Gaussian pulse."""
    I_0 = get_intensity_from_a0(a0, lambda0)
    return (pi / 2) * I_0 * w0**2

def get_rayleigh_length(w0, lambda0):
    """Calculate Rayleigh length (m)."""
    return pi * w0**2 / lambda0

def get_matched_spot_size(n_e, a0):
    k_p = get_k_p(n_e)
    return 2*np.sqrt(a0)/ k_p

def get_guiding_channel_params(n_e, w0):
    """
    Calculate the relative density perturbation for a matched parabolic channel.
    Returns rel_delta_n_over_w2 (m^-2).
    """
    r_e = e**2 / (4 * pi * epsilon_0 * m_e * c**2)
    rel_delta_n_over_w2 = 1.0 / (pi * r_e * w0**4 * n_e)
    return rel_delta_n_over_w2

def get_laser_pulse_length(tau):
    return c*tau