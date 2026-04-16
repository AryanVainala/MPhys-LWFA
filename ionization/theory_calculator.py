"""
Theory Calculator for LWFA Ionization Experiment

This script calculates the theoretical parameters for laser-wakefield acceleration
based on the simulation parameters. These values serve as benchmarks for comparison
with simulation results.

Formulas implemented:
- Plasma frequency and wavelength
- Critical power for relativistic self-focusing
- Dephasing length (blowout regime)
- Pump depletion length (blowout regime)
"""

import numpy as np
from scipy.constants import c, pi
import lwfa_theory as theory

# =====================================
# PARAMETERS FROM ionization_script.py
# =====================================

# Laser parameters
a0 = 2.5            # Normalized laser amplitude
lambda0 = 0.8e-6    # Laser wavelength (m)
w0 = 6e-6         # Laser waist (m)
tau = 16.e-15      # Laser duration (s)

# Target electron density (after full ionization)
# This is the controlled variable across all three gas simulations
n_e = 2.5e24       # electrons/m3

# =====================================
# THEORETICAL CALCULATIONS
# =====================================

# Laser angular frequency
omega_l = 2 * pi * c / lambda0  # rad/s

# Plasma parameters
omega_p = theory.get_omega_p(n_e)
lambda_p = theory.get_lambda_p(n_e)
k_p = theory.get_k_p(n_e)
c_over_omega_p = theory.get_skin_depth(n_e)
n_c = theory.get_critical_e_density(lambda0)
n_ratio = theory.get_density_ratio(n_e, lambda0)
r_bubble = theory.get_bubble_radius(a0, n_e)
gamma_wkfld = theory.get_wakefield_boost(n_e, lambda0)

# Critical power
P_c = theory.get_critical_power(lambda0, n_e)

# Characteristic lengths
L_d = theory.get_dephasing_length(a0, lambda0, n_e)
L_pd = theory.get_pump_depletion_length(a0, lambda0, n_e)

# Laser intensity and power
I_0_cgs = theory.get_intensity_from_a0(a0, lambda0) 
I_0_normalized = I_0_cgs * 1e-18
P_laser = theory.get_laser_power(a0, lambda0, w0)

# Rayleigh length
Z_R = theory.get_rayleigh_length(w0, lambda0)

# Matched spot size and pulse duration
w_m = theory.get_matched_spot_size(a0, n_e)
tau_m = theory.get_matched_pulse_duration(a0, n_e)

# Laser pulse length
L = theory.get_laser_pulse_length(tau)

# Save to file
output_filename = "theoretical_parameters.txt"
with open(output_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("THEORETICAL LWFA PARAMETERS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("INPUT PARAMETERS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Normalized laser amplitude (a₀):     {a0:.2f}\n")
    f.write(f"  Laser wavelength (λ₀):               {lambda0*1e6:.2f} µm\n")
    f.write(f"  Laser waist (w₀):                    {w0*1e6:.2f} µm\n")
    f.write(f"  Laser duration (τ):                  {tau*1e15:.2f} fs\n")
    f.write(f"  Target electron density (nₑ):        {n_e:.2e} m⁻³\n\n")
    
    f.write("PLASMA PARAMETERS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Plasma frequency (ωₚ):               {omega_p:.4e} rad/s\n")
    f.write(f"  Plasma wavelength (λₚ):              {lambda_p*1e6:.4f} µm\n")
    f.write(f"  Plasma wavenumber (kₚ):              {k_p:.4e} m⁻¹\n")
    f.write(f"  Plasma skin depth (c/ωₚ):            {c_over_omega_p*1e6:.4f} µm\n")
    f.write(f"  Critical electron density (n_c):     {n_c*1e-6:.4e} cm⁻³\n")
    f.write(f"  Bubble radius (rb):                  {r_bubble*1e6:.2f} µm\n")
    f.write(f"  Density ratio (n_e/n_c):             {n_ratio:.4e}\n")
    f.write(f"  Plasma status:                       {'UNDERDENSE' if n_e < n_c else 'OVERDENSE'}\n")
    f.write(f"  Frequency ratio (ωₗ/ωₚ):              {omega_l/omega_p:.4f}\n")
    f.write(f"  Wakefield Lorentz Factor (𝛾ₚ):        {gamma_wkfld:.4f}\n\n")
    
    f.write("LASER PARAMETERS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Peak intensity (I₀):                 {I_0_normalized:.4f} × 10¹⁸ W/cm²\n")
    f.write(f"  Laser pulse length (L):              {L*1e6:.2f} µm\n")
    f.write(f"  Laser power (P):                     {P_laser*1e-12:.4f} TW\n")
    f.write(f"  Critical power (P_c):                {P_c*1e-12:.4f} TW\n")
    f.write(f"  Power ratio (P/P_c):                 {P_laser/P_c:.4f}\n")
    f.write(f"  Rayleigh length (ZR):                {Z_R*1e6:.2f} µm\n")
    f.write(f"  Matched spot size (wm):              {w_m*1e6:.2f} µm\n")
    f.write(f"  Matched pulse duration (tau_m):      {tau_m*1e15:.2f} fs\n\n")
    
    f.write("CHARACTERISTIC LENGTHS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Dephasing length (Ld):               {L_d*1e3:.4f} mm  ({L_d*1e6:.2f} µm)\n")
    f.write(f"  Pump depletion length (Lpd):         {L_pd*1e3:.4f} mm  ({L_pd*1e6:.2f} µm)\n")
    f.write(f"  Limiting length (min):               {min(L_d, L_pd)*1e3:.4f} mm\n\n")

print(f"\nResults saved to: {output_filename}")
