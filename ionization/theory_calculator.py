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
from scipy.constants import c, e, m_e, epsilon_0, pi

# =====================================
# PARAMETERS FROM ionization_script.py
# =====================================

# Laser parameters
a0 = 4.0           # Normalized laser amplitude
lambda0 = 0.8e-6   # Laser wavelength (m)
w0 = 5.e-6         # Laser waist (m)
tau = 16.e-15      # Laser duration (s)

# Target electron density (after full ionization)
# This is the controlled variable across all three gas simulations
n_e = 7.0e24       # electrons/m³

# =====================================
# THEORETICAL CALCULATIONS
# =====================================

# Laser angular frequency
omega_l = 2 * pi * c / lambda0  # rad/s

# Plasma angular frequency
omega_p = np.sqrt(n_e * e**2 / (m_e * epsilon_0))  # rad/s

# Plasma wavelength
lambda_p = 2 * pi * c / omega_p  # m

# Plasma wavenumber
k_p = omega_p / c  # 1/m

# Plasma skin depth
c_over_omega_p = c / omega_p  # m

# Critical power for relativistic self-focusing
P_c = 17e9 * (omega_l / omega_p)**2  # W

# Dephasing length (blowout regime, a0 > 2)
# This is the distance over which electrons slip from accelerating to decelerating phase
L_d = (2/3) * (omega_l/omega_p)**2 * np.sqrt(a0) * (c/omega_p)  # m

# Pump depletion length (blowout regime)
# This is the distance over which the laser loses significant energy
L_pd = (omega_l / omega_p)**2 * a0 * (c / omega_p)  # m

# Laser power (calculated from a0)
# For Gaussian beam: I₀ = (ε₀ c / 2) × (m_e c ω_l / e)² × a0²
# P = (π / 2) × I₀ × w0²
I_0 = (epsilon_0 * c / 2) * (m_e * c * omega_l / e)**2 * a0**2  # W/m²
P_laser = (pi / 2) * I_0 * w0**2  # W

# Rayleigh length
Z_R = pi * w0**2 / lambda0  # m

# =====================================
# OUTPUT RESULTS
# =====================================

print("=" * 70)
print("THEORETICAL LWFA PARAMETERS")
print("=" * 70)
print()

print("INPUT PARAMETERS:")
print("-" * 70)
print(f"  Normalized laser amplitude (a₀):    {a0:.2f}")
print(f"  Laser wavelength (λ₀):               {lambda0*1e6:.2f} µm")
print(f"  Laser waist (w₀):                    {w0*1e6:.2f} µm")
print(f"  Laser duration (τ):                  {tau*1e15:.2f} fs")
print(f"  Target electron density (nₑ):        {n_e:.2e} m⁻³")
print()

print("PLASMA PARAMETERS:")
print("-" * 70)
print(f"  Plasma frequency (ωₚ):               {omega_p:.4e} rad/s")
print(f"  Plasma wavelength (λₚ):              {lambda_p*1e6:.4f} µm")
print(f"  Plasma wavenumber (kₚ):              {k_p:.4e} m⁻¹")
print(f"  Plasma skin depth (c/ωₚ):            {c_over_omega_p*1e6:.4f} µm")
print(f"  Frequency ratio (ωₗ/ωₚ):             {omega_l/omega_p:.4f}")
print()

print("LASER PARAMETERS:")
print("-" * 70)
print(f"  Laser power (P):                     {P_laser*1e-12:.4f} TW")
print(f"  Critical power (Pₖ):                 {P_c*1e-12:.4f} TW")
print(f"  Power ratio (P/Pₖ):                  {P_laser/P_c:.4f}")
print(f"  Rayleigh length (ZR):                {Z_R*1e6:.2f} µm")
print()

print("CHARACTERISTIC LENGTHS:")
print("-" * 70)
print(f"  Dephasing length (Ld):               {L_d*1e3:.4f} mm  ({L_d*1e6:.2f} µm)")
print(f"  Pump depletion length (Lpd):         {L_pd*1e3:.4f} mm  ({L_pd*1e6:.2f} µm)")
print(f"  Limiting length (min):               {min(L_d, L_pd)*1e3:.4f} mm")
print()

print("=" * 70)
print("NOTES:")
print("  - These values assume ideal blowout regime (a₀ > 2)")
print("  - Dephasing length: distance before electrons dephase from wake")
print("  - Pump depletion: distance before laser energy significantly depletes")
print("  - The shorter of Ld and Lpd typically limits acceleration")
print("=" * 70)

# Save to file for reference
output_filename = "theoretical_parameters.txt"
with open(output_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("THEORETICAL LWFA PARAMETERS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("INPUT PARAMETERS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Normalized laser amplitude (a₀):    {a0:.2f}\n")
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
    f.write(f"  Frequency ratio (ωₗ/ωₚ):             {omega_l/omega_p:.4f}\n\n")
    
    f.write("LASER PARAMETERS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Laser power (P):                     {P_laser*1e-12:.4f} TW\n")
    f.write(f"  Critical power (Pₖ):                 {P_c*1e-12:.4f} TW\n")
    f.write(f"  Power ratio (P/Pₖ):                  {P_laser/P_c:.4f}\n")
    f.write(f"  Rayleigh length (ZR):                {Z_R*1e6:.2f} µm\n\n")
    
    f.write("CHARACTERISTIC LENGTHS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Dephasing length (Ld):               {L_d*1e3:.4f} mm  ({L_d*1e6:.2f} µm)\n")
    f.write(f"  Pump depletion length (Lpd):         {L_pd*1e3:.4f} mm  ({L_pd*1e6:.2f} µm)\n")
    f.write(f"  Limiting length (min):               {min(L_d, L_pd)*1e3:.4f} mm\n\n")

print(f"\nResults saved to: {output_filename}")
