"""
Calibration Simulation Script for Ionization Injection Tuning

This script performs an a0 scan to calibrate the ionization injection threshold.
It compares Pure Helium vs. Nitrogen-Doped Helium to isolate the injection signal.

Usage:
------
python a0_calib_simulation.py --a0 2.0 --mode pure_he
python a0_calib_simulation.py --a0 2.0 --mode doped --dopant_conc 0.01
"""

# -------
# Imports
# -------
import numpy as np
import math
import argparse
import sys
import os
from scipy.constants import c, e, m_e, m_p, epsilon_0, pi
from fbpic.main import Simulation
from fbpic.utils.random_seed import set_random_seed
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
    ParticleChargeDensityDiagnostic

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description='Run ionization calibration simulation')
parser.add_argument('--a0', type=float, default=2.0, help='Laser normalized amplitude')
parser.add_argument('--mode', type=str, choices=['pure_he', 'doped'], default='pure_he', help='Simulation mode')
parser.add_argument('--dopant_conc', type=float, default=0.01, help='Dopant concentration (fraction)')
args = parser.parse_args()

# ==========================================
# FUNDAMENTAL PARAMETERS
# ==========================================

# Computational settings
use_cuda = False
n_order = -1  # -1 for infinite order (single GPU)

# Target electron density (CONSTANT across all simulations)
n_e_target = 7.0e24  # electrons/m³

# Laser parameters
a0 = args.a0
lambda0 = 0.8e-6  # Laser wavelength (m)
w0 = 5.e-6        # Laser waist (m)
tau = 16.e-15     # Laser duration (s)
z0 = -5.e-6       # Laser centroid (m)
z_foc = 20.e-6    # Focal position (m)

# Plasma structure
p_zmin = 0.e-6       # Start of plasma (m)
ramp_length = 20.e-6  # Length of entrance ramp (m)

# Particle resolution per cell
p_nz = 2  # Particles per cell along z
p_nr = 2  # Particles per cell along r
p_nt = 4  # Particles per cell along theta

# Moving window
v_window = c

# Diagnostics
diag_period = 50
track_electrons = False

# Simulation length
L_interact = 50.e-6  # Interaction length (m)

# ==========================================
# GAS DENSITY CALCULATION
# ==========================================
# Atomic numbers
Z_He = 2
Z_N = 7

# Calculate densities based on mode
if args.mode == 'pure_he':
    # Pure Helium case: n_He * Z_He = n_e_target
    n_He = n_e_target / Z_He
    n_N = 0.0
    print(f"Mode: Pure Helium (a0={a0})")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_N : {n_N:.4e} m^-3")
    
elif args.mode == 'doped':
    # Doped case: n_He * Z_He + n_N * Z_N = n_e_target
    # n_N = n_He * dopant_conc
    # n_He * Z_He + n_He * dopant_conc * Z_N = n_e_target
    # n_He * (Z_He + dopant_conc * Z_N) = n_e_target
    
    n_He = n_e_target / (Z_He + args.dopant_conc * Z_N)
    n_N = n_He * args.dopant_conc
    print(f"Mode: Nitrogen-Doped Helium (a0={a0}, conc={args.dopant_conc*100}%)")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_N : {n_N:.4e} m^-3")

# Verify total electron density
n_e_total = n_He * Z_He + n_N * Z_N
print(f"  Total potential n_e: {n_e_total:.4e} m^-3 (Target: {n_e_target:.4e})")

# Particle masses
m_He = 4.0 * m_p
m_N = 14.0 * m_p

# ==========================================
# CALCULATED PLASMA PARAMETERS
# ==========================================

# Plasma parameters at target density
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
lambda_p = 2 * pi * c / omega_p
skin_depth = c / omega_p

# ==========================================
# SIMULATION BOX PARAMETERS
# ==========================================

# Box dimensions
zmax = 10.e-6
zmin = -30.e-6
rmax = 20.e-6

# Calculate box lengths
Lz = zmax - zmin
Lr = rmax

# ==========================================
# INTELLIGENT GRID RESOLUTION
# ==========================================

# Axial resolution: 15 points per laser wavelength
dz_target = lambda0 / 15.0
Nz = int(np.ceil(Lz / dz_target))

# Radial resolution: 5 points per plasma skin depth
dr_target = skin_depth / 5.0
Nr = int(np.ceil(Lr / dr_target))

# Number of azimuthal modes
Nm = 2

# Timestep
dt = (zmax - zmin) / Nz / c

# Interaction time
T_interact = (L_interact + Lz) / v_window
N_step = int(T_interact / dt)

# ==========================================
# TRANSVERSE PARABOLIC DENSITY PROFILE
# ==========================================

# Matched spot size for guiding
w_matched = w0

# Parabolic channel parameter
r_e = e**2 / (4 * pi * epsilon_0 * m_e * c**2)
rel_delta_n_over_w2 = 1.0 / (pi * r_e * w_matched**4 * n_e_target)

# ==========================================
# DENSITY FUNCTION
# ==========================================

def dens_func(z, r):
    """Returns relative density at position z and r"""
    # Start with uniform density
    n = np.ones_like(z)
    
    # Smooth entrance ramp
    n = np.where(z < ramp_length, 
                 np.sin(pi/2 * z / ramp_length)**2, 
                 n)
    
    # Suppress density before plasma starts
    n = np.where(z < 0, 0.0, n)
    
    # Add transverse parabolic profile for laser guiding
    n = n * (1.0 + rel_delta_n_over_w2 * r**2)
    
    return n

# ==========================================
# MAIN SIMULATION
# ==========================================

if __name__ == '__main__':
    
    # Set random seed for reproducibility
    set_random_seed(0)
    
    # Initialize simulation
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, 
                     zmin=zmin,
                     n_order=n_order, 
                     use_cuda=use_cuda,
                     boundaries={'z': 'open', 'r': 'reflective'})
    
    # ==========================================
    # ADD PARTICLES
    # ==========================================
    
    # 1. Helium Atoms (Neutral)
    print("Adding neutral Helium atoms...")
    atoms_he = sim.add_new_species(
        q=0, m=m_He, n=n_He,
        dens_func=dens_func,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    # 2. Nitrogen Atoms (Neutral) - Only if doped
    atoms_n = None
    if n_N > 0:
        print("Adding neutral Nitrogen atoms...")
        atoms_n = sim.add_new_species(
            q=0, m=m_N, n=n_N,
            dens_func=dens_func,
            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
            p_zmin=p_zmin
        )
    
    # 3. Electron Species - SEPARATED
    print("Creating electron species...")
    electrons_he = sim.add_new_species(q=-e, m=m_e)
    electrons_n = sim.add_new_species(q=-e, m=m_e)
    
    # ==========================================
    # ACTIVATE FIELD IONIZATION
    # ==========================================
    
    # Ionize Helium -> electrons_he
    # Levels 0 -> 2 (Full ionization)
    atoms_he.make_ionizable('He', target_species=electrons_he, level_start=0, level_max=2)
    
    # Ionize Nitrogen -> electrons_n (if present)
    # Levels 0 -> 7 (Full ionization)
    if atoms_n:
        atoms_n.make_ionizable('N', target_species=electrons_n, level_start=0, level_max=7)
    
    # ==========================================
    # LASER PULSE
    # ==========================================
    
    print(f"Adding laser pulse (a0={a0})...")
    laser_profile = GaussianLaser(a0=a0, waist=w0, tau=tau, z0=z0, zf=z_foc, lambda0=lambda0)
    add_laser_pulse(sim, laser_profile)
    
    # ==========================================
    # MOVING WINDOW
    # ==========================================
    
    sim.set_moving_window(v=v_window)
    
    # ==========================================
    # DIAGNOSTICS
    # ==========================================
    
    # Output directory
    write_dir = f"diags_calib/a{a0}_{args.mode}"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Field Diagnostics
    sim.diags = [
        FieldDiagnostic(period=diag_period, fldobject=sim.fld, 
                        comm=sim.comm, fieldtypes=['rho', 'E', 'B'],
                        write_dir=write_dir)
    ]
    
    # Particle Diagnostics - Save BOTH electron species
    # We want to distinguish source of electrons
    species_dict = {'electrons_he': electrons_he, 'electrons_n': electrons_n}
    
    sim.diags.append(
        ParticleDiagnostic(period=diag_period, species=species_dict,
                           comm=sim.comm, 
                           particle_data=['position', 'momentum', 'weighting'],
                           write_dir=write_dir)
    )
    
    # ==========================================
    # RUN SIMULATION
    # ==========================================
    
    print(f"Running simulation for {N_step} steps...")
    sim.step(N_step)
    print("Simulation complete.")
