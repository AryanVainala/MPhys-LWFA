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
    ParticleChargeDensityDiagnostic, set_periodic_checkpoint, restart_from_checkpoint


# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# Computational settings
use_cuda = True
n_order = -1  # -1 for infinite order (single GPU)

# Target electron density (CONSTANT across all simulations)
n_e_target = 3.5e18*1.e6  # electrons/m³

# Simulation configuration
mode = 'doped'  # 'pure_he' or 'doped'
dopant_species = 'Ar'  # 'N', 'Ne', 'Ar'
dopant_conc = 0.01  # Dopant concentration (fraction)

# Laser parameters
a0 = 2.5  # Laser normalized amplitude
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
diag_period = 1000 # Higher means less frequent measurements
save_checkpoints = False
checkpoint_period = 500
use_restart = False
track_electrons = False # I TURNED IT OFF

# Simulation length
L_interact = 1500.e-6  # Interaction length (m)

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

# Define the background charge state (Z that contributes to plasma density)
bg_charge = {'N': 5, 'Ne': 8, 'Ar': 8}

if mode == 'doped':
    if dopant_species not in dopant_params:
        raise ValueError(f"Unknown dopant species: {dopant_species}")

    Z_dopant = dopant_params[dopant_species]['Z']
    m_dopant = dopant_params[dopant_species]['m']
    levels_dopant = dopant_params[dopant_species]['levels']
    name_dopant = dopant_params[dopant_species]['name']
else:
    # Dummy values for pure_he mode
    Z_dopant = 0
    m_dopant = 0
    levels_dopant = 0
    name_dopant = "None"

# Calculate densities based on mode
if mode == 'pure_he':
    # Pure Helium case: n_He * Z_He = n_e_target
    n_He = n_e_target / Z_He
    n_dopant = 0.0
    print(f"Mode: Pure Helium (a0={a0})")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_dopant: 0.0000e+00 m^-3")
    
elif mode == 'doped':
    # Doped case: n_He * Z_He + n_dopant * Z_dopant_bg = n_e_target
    # where n_dopant = n_gas_total * dopant_conc
    # and n_He = n_gas_total * (1 - dopant_conc)
    
    Z_dopant_bg = bg_charge[dopant_species]
    effective_Z = (1.0 - dopant_conc) * Z_He + (dopant_conc * Z_dopant_bg)
    n_gas_total = n_e_target / effective_Z
    
    n_He = n_gas_total * (1.0 - dopant_conc)
    n_dopant = n_gas_total * dopant_conc
    
    print(f"Mode: {name_dopant}-Doped Helium (a0={a0}, conc={dopant_conc*100}%)")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_{dopant_species} : {n_dopant:.4e} m^-3")

# Verify total electron density
if mode == 'doped':
    n_e_total = n_He * Z_He + n_dopant * bg_charge[dopant_species]
else:
    n_e_total = n_He * Z_He
print(f"  Total background n_e: {n_e_total:.4e} m^-3 (Target: {n_e_target:.4e})")

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
# Bigger box 60 µm
# zmax = 15.e-6
# zmin = -45.e-6
# rmax = 15.e-6

# Calculate box lengths
Lz = zmax - zmin
Lr = rmax

# ==========================================
# GRID RESOLUTION
# ==========================================

# Axial resolution: no points per laser wavelength
dz_target = lambda0 / 30
Nz = int(np.ceil(Lz / dz_target))

# Radial resolution: no points per plasma skin depth
dr_target = skin_depth / 30
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
    
    # Set random seed for reproducibility``
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
    
    # 1. Helium Ions (Pre-ionized He2+)
    # We assume He is fully ionized by the laser foot, so we start with He2+ ions
    print("Adding Helium ions (He2+)...")
    atoms_he = sim.add_new_species(
        q=2*e, m=m_He, n=n_He,
        dens_func=dens_func,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    # 2. Dopant Ions (Pre-ionized to background level)
    atoms_dopant = None
    if n_dopant > 0:
        q_dopant = bg_charge[dopant_species] * e
        print(f"Adding {name_dopant} ions (Charge {bg_charge[dopant_species]}+)...")
        atoms_dopant = sim.add_new_species(
            q=q_dopant, m=m_dopant, n=n_dopant,
            dens_func=dens_func,
            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
            p_zmin=p_zmin
        )
    
    # 3. Electron Species
    # Bulk electrons: Neutralize the pre-ionized ions (He2+ and Dopant^Z+)
    # These form the wakefield.
    print(f"Adding bulk electrons (n_e = {n_e_total:.2e})...")
    electrons_bulk = sim.add_new_species(
        q=-e, m=m_e, n=n_e_total,
        dens_func=dens_func,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    # Injected electrons: Initially empty.
    # These will be born from further ionization of the dopant.
    electrons_injected = sim.add_new_species(q=-e, m=m_e)
    
    # ==========================================
    # ACTIVATE FIELD IONIZATION
    # ==========================================
    
    # Helium is fully ionized (He2+), so no further ionization source.
    
    # Dopant ionizes further -> electrons_injected
    if atoms_dopant:
        # Start ionization from the background level
        level_start = bg_charge[dopant_species]
        print(f"Activating ionization for {dopant_species} from level {level_start}...")
        atoms_dopant.make_ionizable(dopant_species, target_species=electrons_injected, level_start=level_start)
    
    # ==========================================
    # LASER PULSE
    # ==========================================
    
    print(f"Adding laser pulse (a0={a0})...")
    laser_profile = GaussianLaser(a0=a0, waist=w0, tau=tau, z0=z0, zf=z_foc, lambda0=lambda0)
    add_laser_pulse(sim, laser_profile)
    
    # ==========================================
    # MOVING WINDOW & CHECKPOINTS
    # ==========================================
    
    if use_restart is False:
        # Track injected electrons if required
        if track_electrons:
            print("Tracking injected electrons enabled.")
            electrons_injected.track(sim.comm)
    else:
        # Load the fields and particles from the latest checkpoint file
        print("Restarting from checkpoint...")
        restart_from_checkpoint(sim)

    sim.set_moving_window(v=v_window)
    
    # Add checkpoints
    if save_checkpoints:
        print(f"Checkpoints enabled (every {checkpoint_period} steps).")
        set_periodic_checkpoint(sim, checkpoint_period)
    
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
    
    # Particle Diagnostics
    # electrons_bulk: The wakefield driver
    # electrons_injected: The accelerated bunch
    species_dict = {'electrons_bulk': electrons_bulk, 'electrons_injected': electrons_injected}
    
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
    sim.step( N_step )
    print("Simulation complete.")
