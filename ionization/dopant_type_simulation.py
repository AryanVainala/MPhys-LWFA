"""
Simulation Script for Ionization Injection for different dopant Types

This script performs simulations for ionisation injection using different dopant types.
The user defines either Pure Helium or Nitrogen-Doped Helium to isolate the injection signal, dopant type, and concentration.

Usage:
------
python dopant_type_simulation.py
"""

# -------
# Imports
# -------
import numpy as np
import math
import os
from scipy.constants import c, e, m_e, m_p, epsilon_0, pi
from fbpic.main import Simulation
from fbpic.utils.random_seed import set_random_seed
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
    ParticleChargeDensityDiagnostic


# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# Computational settings
use_cuda = False
n_order = -1  # -1 for infinite order (single GPU)

# Target electron density (CONSTANT across all simulations)
n_e_target = 2.e18*1.e6  # electrons/m³

# Simulation configuration
a0 = 2.0  # Laser normalized amplitude
mode = 'doped'  # 'pure_he' or 'doped'
dopant_species = 'N'  # 'N', 'Ne', 'Ar'
dopant_conc = 0.01  # Dopant concentration (fraction)

# Laser parameters
a0_param = 2.0
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

# Dopant parameters
dopant_params = {
    'N': {'Z': 7, 'm': 14 * m_p, 'levels': 7, 'name': 'Nitrogen'},
    'Ne': {'Z': 10, 'm': 20 * m_p, 'levels': 10, 'name': 'Neon'},
    'Ar': {'Z': 18, 'm': 39 * m_p, 'levels': 18, 'name': 'Argon'}
}

if dopant_species not in dopant_params:
    raise ValueError(f"Unknown dopant species: {dopant_species}")

Z_dopant = dopant_params[dopant_species]['Z']
m_dopant = dopant_params[dopant_species]['m']
levels_dopant = dopant_params[dopant_species]['levels']
name_dopant = dopant_params[dopant_species]['name']

# Calculate densities based on mode
if mode == 'pure_he':
    # Pure Helium case: n_He * Z_He = n_e_target
    n_He = n_e_target / Z_He
    n_dopant = 0.0
    print(f"Mode: Pure Helium (a0={a0})")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_{dopant_species} : {n_dopant:.4e} m^-3")
    
elif mode == 'doped':
    # Doped case: n_He * Z_He + n_dopant * Z_dopant = n_e_target
    # n_dopant = n_He * dopant_conc
    
    n_He = n_e_target / (Z_He + dopant_conc * Z_dopant)
    n_dopant = n_He * dopant_conc
    print(f"Mode: {name_dopant}-Doped Helium (a0={a0}, conc={dopant_conc*100}%)")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_{dopant_species} : {n_dopant:.4e} m^-3")

# Verify total electron density
n_e_total = n_He * Z_He + n_dopant * Z_dopant
print(f"  Total potential n_e: {n_e_total:.4e} m^-3 (Target: {n_e_target:.4e})")

# Particle masses
m_He = 4. * m_p

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

# Axial resolution: no points per laser wavelength
dz_target = lambda0 / 10.0
Nz = int(np.ceil(Lz / dz_target))

# Radial resolution: no points per plasma skin depth
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
    
    # 2. Dopant Atoms (Neutral) - Only if doped
    atoms_dopant = None
    if n_dopant > 0:
        print(f"Adding neutral {name_dopant} atoms...")
        atoms_dopant = sim.add_new_species(
            q=0, m=m_dopant, n=n_dopant,
            dens_func=dens_func,
            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
            p_zmin=p_zmin
        )
    
    # 3. Electron Species - SEPARATED
    print("Creating electron species...")
    electrons_he = sim.add_new_species(q=-e, m=m_e)
    electrons_dopant = sim.add_new_species(q=-e, m=m_e)
    
    # ==========================================
    # ACTIVATE FIELD IONIZATION
    # ==========================================
    
    # Ionize Helium -> electrons_he
    # Levels 0 -> 2 (Full ionization)
    atoms_he.make_ionizable('He', target_species=electrons_he, level_start=0, level_max=2)
    
    # Ionize Dopant -> electrons_dopant (if present)
    if atoms_dopant:
        atoms_dopant.make_ionizable(dopant_species, target_species=electrons_dopant, level_start=0, level_max=levels_dopant)
    
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
    write_dir = f"diags_doped/a{a0}_{mode}_{dopant_species}" if mode == 'doped' else f"diags_doped/a{a0}_{mode}"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Field Diagnostics
    sim.diags = [
        FieldDiagnostic(period=diag_period, fldobject=sim.fld, 
                        comm=sim.comm,
                        write_dir=write_dir)
    ]
    
    # Particle Diagnostics - Save BOTH electron species
    # We want to distinguish source of electrons
    species_dict = {'electrons_he': electrons_he, 'electrons_dopant': electrons_dopant}
    
    sim.diags.append(
        ParticleDiagnostic(period=diag_period, species=species_dict,
                           comm=sim.comm, 
                           write_dir=write_dir)
    )

    # Particle Charge Density Diagnostic - To get proper electron density
    sim.diags.append(
        ParticleChargeDensityDiagnostic(period=diag_period, sim=sim,
                                        species=species_dict,
                                        write_dir=write_dir)
    )
    
    # ==========================================
    # RUN SIMULATION
    # ==========================================
    
    print(f"Running simulation for {N_step} steps...")
    sim.step(N_step)
    print("Simulation complete.")
