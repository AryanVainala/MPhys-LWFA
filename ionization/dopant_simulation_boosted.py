"""
Simulation Script for Ionization Injection for different dopant Types
(Boosted Frame Version)

This script performs simulations for ionisation injection using different dopant types.
The user defines either Pure Helium or Nitrogen-Doped Helium to isolate the injection signal, dopant type, and concentration.

Usage:
------
python dopant_simulation_boosted.py
"""

# -------
# Imports
# -------
import numpy as np
import os
import argparse
from scipy.constants import c, e, m_e, m_p, epsilon_0, pi
from fbpic.main import Simulation
from fbpic.utils.random_seed import set_random_seed
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
    ParticleChargeDensityDiagnostic, set_periodic_checkpoint, restart_from_checkpoint, \
    BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic


# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run LWFA simulation with dopant')
parser.add_argument('--a0', type=float, default=2.5, help='Laser normalized amplitude')
parser.add_argument('--ne', type=float, default=2.5e23, help='Target electron density in m^-3')
parser.add_argument('--mode', type=str, default='doped', choices=['pure_he', 'doped'], help='Simulation mode')
parser.add_argument('--dopant', type=str, default='N', choices=['N', 'Ne', 'Ar'], help='Dopant species')
parser.add_argument('--conc', type=float, default=0.01, help='Dopant concentration fraction')

args = parser.parse_args()

# Computational settings
use_cuda = True
n_order = -1  # -1 for infinite order (single GPU)

# Boosted Frame Settings
gamma_boost = 10.
boost = BoostConverter(gamma_boost)

# Target electron density (CONSTANT across all simulations)
n_e_target = args.ne  # electrons/m³

# Simulation configuration
mode = args.mode  # 'pure_he' or 'doped'
dopant_species = args.dopant  # 'N', 'Ne', 'Ar'
dopant_conc = args.conc  # Dopant concentration (fraction)

# Laser parameters
a0 = args.a0  # Laser normalized amplitude
lambda0 = 0.8e-6  # Laser wavelength (m)
w0 = 18.7e-6       # Laser waist (m)
tau = 60.e-15    # Laser duration (s)
z0 = -18.7e-6       # Laser centroid (m)
z_foc = -74.8e-6    # Focal position (m)

# Plasma structure
p_zmin = 0.e-6       # Start of plasma (m)
ramp_length = 74.8e-6  # Length of entrance ramp (m)

# Particle resolution per cell
p_nz = 2  # Particles per cell along z
p_nr = 2  # Particles per cell along r
p_nt = 4  # Particles per cell along theta

# Moving window (Group velocity in plasma)
v_window = c * np.sqrt(1. - n_e_target / 1.742e27)

# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving = - c * np.sqrt( 1. - 1./boost.gamma0**2 )

# Diagnostics
N_lab_diag = 50+1  # Number of snapshots in the lab frame
write_period = 50 # How often the cache flushes to disk
save_checkpoints = True
checkpoint_period = 1000
use_restart = False
track_electrons = True

# Simulation length
L_interact = 78e-3  # Interaction length (m)

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
    n_He = n_e_target / Z_He
    n_dopant = 0.0
    print(f"Mode: Pure Helium (a0={a0}, ne={n_e_target:.2e})")
    print(f"  n_He: {n_He:.4e} m^-3")
    print(f"  n_dopant: 0.0000e+00 m^-3")
    
elif mode == 'doped':
    Z_dopant_bg = bg_charge[dopant_species]
    effective_Z = (1.0 - dopant_conc) * Z_He + (dopant_conc * Z_dopant_bg)
    n_gas_total = n_e_target / effective_Z
    
    n_He = n_gas_total * (1.0 - dopant_conc)
    n_dopant = n_gas_total * dopant_conc
    
    print(f"Mode: {name_dopant}-Doped Helium (a0={a0}, ne={n_e_target:.2e}, conc={dopant_conc*100}%)")
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
zmax = 37.e-6
zmin = -112.e-6
rmax = 74.8e-6

# Calculate box lengths
Lz = zmax - zmin
Lr = rmax

# ==========================================
# GRID RESOLUTION
# ==========================================

# Axial resolution: no points per laser wavelength
dz_target = lambda0 / 10
Nz = int(np.ceil(Lz / dz_target))

# Radial resolution: no points per plasma skin depth
dr_target = skin_depth / 60
print('dr',dr_target)
Nr = int(np.ceil(Lr / dr_target))

# Number of azimuthal modes
Nm = 2

# Timestep adjusted for boosted frame calculation
dt = min( rmax/(2*boost.gamma0*Nr)/c, (zmax-zmin)/Nz/c )
print('dt=',dt)

print('Rel. CFL=', dr_target / ((2*gamma_boost) * c *dt), 'this MUST be greater than 1')

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
    
    # Initialize simulation with boosted frame parameters
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, 
                     zmin=zmin,
                     v_comoving=v_comoving, gamma_boost=boost.gamma0,
                     n_order=n_order, 
                     use_cuda=use_cuda,
                     boundaries={'z': 'open', 'r': 'reflective'})
    
    # Calculate N_step using the boosted frame timestep
    T_interact_boost = boost.interaction_time( L_interact, Lz, v_window )
    N_step = int(T_interact_boost / sim.dt)
    
    # ==========================================
    # ADD PARTICLES
    # ==========================================
    
    print("Adding Helium ions (He2+)...")
    atoms_he = sim.add_new_species(
        q=2*e, m=m_He, n=n_He,
        dens_func=dens_func, boost_positions_in_dens_func=True,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    atoms_dopant = None
    if n_dopant > 0:
        q_dopant = bg_charge[dopant_species] * e
        print(f"Adding {name_dopant} ions (Charge {bg_charge[dopant_species]}+)...")
        atoms_dopant = sim.add_new_species(
            q=q_dopant, m=m_dopant, n=n_dopant,
            dens_func=dens_func, boost_positions_in_dens_func=True,
            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
            p_zmin=p_zmin
        )
    
    print(f"Adding bulk electrons (n_e = {n_e_total:.2e})...")
    electrons_bulk = sim.add_new_species(
        q=-e, m=m_e, n=n_e_total,
        dens_func=dens_func, boost_positions_in_dens_func=True,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    electrons_injected = sim.add_new_species(q=-e, m=m_e)
    
    # ==========================================
    # ACTIVATE FIELD IONIZATION
    # ==========================================
    
    if atoms_dopant:
        level_start = bg_charge[dopant_species]
        print(f"Activating ionization for {dopant_species} from level {level_start}...")
        atoms_dopant.make_ionizable(dopant_species, target_species=electrons_injected, level_start=level_start)
    
    # ==========================================
    # LASER PULSE
    # ==========================================
    
    print(f"Adding laser pulse (a0={a0})...")
    laser_profile = GaussianLaser(a0=a0, waist=w0, tau=tau, z0=z0, zf=z_foc, lambda0=lambda0)
    add_laser_pulse(sim, laser_profile, gamma_boost=boost.gamma0, method='antenna', z0_antenna=0)
    
    # ==========================================
    # MOVING WINDOW & CHECKPOINTS
    # ==========================================
    
    if use_restart is False:
        if track_electrons:
            print("Tracking injected electrons enabled.")
            electrons_injected.track(sim.comm)
    else:
        print("Restarting from checkpoint...")
        restart_from_checkpoint(sim)

    # Convert window velocity to boosted frame
    v_window_boosted, = boost.velocity([v_window])
    sim.set_moving_window(v=v_window_boosted)
    
    if save_checkpoints:
        print(f"Checkpoints enabled (every {checkpoint_period} steps).")
        set_periodic_checkpoint(sim, checkpoint_period)
    
    # ==========================================
    # DIAGNOSTICS
    # ==========================================
    
    # Output directory
    write_dir_base = f"test_n{n_e_target:.1e}"
    if mode == 'doped':
        write_dir = os.path.join(write_dir_base, f"a{a0}_doped_{dopant_species}")
    else:
        write_dir = os.path.join(write_dir_base, "a{a0}_pure_he")

    # Time interval between diagnostics *in the lab frame*
    # Note: N_lab_diag - 1 to ensure we cover the range
    dt_lab_diag_period = (L_interact + Lz) / v_window / (N_lab_diag - 1)
    
    print(f"Diagnostics: {N_lab_diag} lab-frame snapshots over {L_interact*1e3:.1f} mm.")

    species_dict = {'electrons_bulk': electrons_bulk, 'electrons_injected': electrons_injected}

    # Diagnostics in the lab frame (back-transformed from simulation frame to lab frame)
    # saving particles separately allows reconstruction of species density (rho_electrons_bulk, etc.) in post-processing
    sim.diags = [
        BackTransformedFieldDiagnostic(zmin, zmax, v_window,
            dt_lab_diag_period, N_lab_diag, boost.gamma0,
            fieldtypes=['rho_electrons_bulk', 'rho_electrons_injected','E','B'], period=write_period,
            fldobject=sim.fld, comm=sim.comm, write_dir=write_dir),

        BackTransformedParticleDiagnostic(zmin, zmax, v_window,
            dt_lab_diag_period, N_lab_diag, boost.gamma0,
            write_period, sim.fld, species=species_dict, 
            comm=sim.comm, write_dir=write_dir)
    ]
    
    # ==========================================
    # RUN SIMULATION
    # ==========================================
    
    print(f"Running simulation for {N_step} steps in the boosted frame (gamma={boost.gamma0})...")
    # sim.step( N_step )
    print("Simulation complete.")