"""
Base Simulation Script for LWFA Ionization Study

This script simulates laser-wakefield acceleration with field ionization
for different atomic species (H, He, N). The key innovation is that all
three species are configured to produce the SAME final electron density
after full ionization, allowing direct comparison of ionization dynamics.

Key Features:
- Single neutral atomic species (configurable)
- Intelligent grid resolution based on physical scales
- Transverse parabolic density profile for laser guiding
- Full field ionization from neutral state

Usage:
------
1. Set gas_type to 'H', 'He', or 'N'
2. The script automatically calculates the required atomic density
3. Run: python base_simulation.py
4. Rename output directory after each run (diags_H, diags_He, diags_N)
"""

# -------
# Imports
# -------
import numpy as np
import math
from scipy.constants import c, e, m_e, m_p, epsilon_0, pi
from fbpic.main import Simulation
from fbpic.utils.random_seed import set_random_seed
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
    ParticleChargeDensityDiagnostic, set_periodic_checkpoint, restart_from_checkpoint

# ==========================================
# CONFIGURATION: SELECT GAS SPECIES
# ==========================================
# Change this to 'H', 'He', or 'N' for each simulation run
gas_type = 'H'  # <--- MODIFY THIS FOR EACH RUN

# ==========================================
# FUNDAMENTAL PARAMETERS
# ==========================================

# Computational settings
use_cuda = False
n_order = -1  # -1 for infinite order (single GPU)

# Target electron density (CONSTANT across all simulations)
# This is our control variable - same final n_e for all gases
n_e_target = 7.0e24  # electrons/m³

# Laser parameters (from ionization_script.py)
a0 = 4.0          # Laser normalized amplitude
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

# Diagnostics and checkpoints
diag_period = 50
save_checkpoints = False
checkpoint_period = 100
use_restart = False
track_electrons = False

# Simulation length
# Note: Must be at least 1-2× dephasing length (L_d) for meaningful acceleration
L_interact = 50.e-6  # Interaction length (m) - approximately 1× L_d

# ==========================================
# GAS-SPECIFIC CONFIGURATION
# ==========================================
# This is the CRITICAL SECTION that ensures equal final electron density

gas_properties = {
    'H': {
        'symbol': 'H',
        'mass_amu': 1.0,      # Atomic mass (amu)
        'Z': 1,               # Atomic number (electrons per atom)
        'description': 'Hydrogen (H)'
    },
    'He': {
        'symbol': 'He',
        'mass_amu': 4.0,
        'Z': 2,
        'description': 'Helium (He)'
    },
    'N': {
        'symbol': 'N',
        'mass_amu': 14.0,
        'Z': 7,
        'description': 'Nitrogen (N)'
    }
}

# Get properties for selected gas
if gas_type not in gas_properties:
    raise ValueError(f"Invalid gas_type: {gas_type}. Must be 'H', 'He', or 'N'")

gas = gas_properties[gas_type]

# Calculate the atomic density required to achieve the target electron density
# Core principle: n_atom = n_e / Z
# Each atom of type X provides Z electrons upon full ionization
atomic_density = n_e_target / gas['Z']

print(f"\n{'='*70}")
print(f"Gas: {gas['description']}")
print(f"Atomic number (electrons per atom): {gas['Z']}")
print(f"Required atomic density: {atomic_density:.4e} atoms/m³")
print(f"Target electron density: {n_e_target:.4e} e⁻/m³")
print(f"  (Verification: {atomic_density:.4e} atoms/m³ × {gas['Z']} e⁻/atom = {atomic_density * gas['Z']:.4e} e⁻/m³)")
print(f"{'='*70}\n")

# The particle mass is always the atomic mass in our atom-centric model
particle_mass = gas['mass_amu'] * m_p

# ==========================================
# CALCULATED PLASMA PARAMETERS
# ==========================================

# Plasma parameters at target density
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
lambda_p = 2 * pi * c / omega_p
k_p = omega_p / c
skin_depth = c / omega_p

print(f"Plasma wavelength (λₚ): {lambda_p*1e6:.4f} µm")
print(f"Plasma skin depth (c/ωₚ): {skin_depth*1e6:.4f} µm")
print()

# ==========================================
# SIMULATION BOX PARAMETERS
# ==========================================

# Box dimensions (from ionization_script.py)
zmax = 10.e-6
zmin = -30.e-6
rmax = 20.e-6

# Calculate box lengths
Lz = zmax - zmin
Lr = rmax

# ==========================================
# INTELLIGENT GRID RESOLUTION
# ==========================================
# Resolution based on physical scales, not arbitrary numbers

# Axial resolution: 15 points per laser wavelength
dz_target = lambda0 / 15.0
Nz = int(np.ceil(Lz / dz_target))

# Radial resolution: 5 points per plasma skin depth
dr_target = skin_depth / 5.0
Nr = int(np.ceil(Lr / dr_target))

# Number of azimuthal modes
Nm = 2

print("GRID RESOLUTION:")
print(f"  Axial: Nz = {Nz} ({Lz/lambda0:.1f} wavelengths, {dz_target*1e6:.4f} µm/cell)")
print(f"  Radial: Nr = {Nr} ({Lr/skin_depth:.1f} skin depths, {dr_target*1e6:.4f} µm/cell)")
print(f"  Azimuthal modes: Nm = {Nm}")
print()

# Timestep
dt = (zmax - zmin) / Nz / c

# Interaction time
T_interact = (L_interact + Lz) / v_window
N_step = int(T_interact / dt)

print("SIMULATION:")
print(f"  Timestep: {dt*1e15:.4f} fs")
print(f"  Interaction length: {L_interact*1e6:.1f} µm")
print(f"  Total steps: {N_step}")
print() #These are line breaks 

# ==========================================
# TRANSVERSE PARABOLIC DENSITY PROFILE
# ==========================================
# Implements laser guiding channel (from lwfa_script.py)

# Matched spot size for guiding
w_matched = w0

# Parabolic channel parameter
# Δn/n = (r²/w²) × [1/(π × r_e × w⁴ × n_e)]
# where r_e = classical electron radius = e²/(4πε₀m_e c²)
r_e = e**2 / (4 * pi * epsilon_0 * m_e * c**2)  # Classical electron radius
rel_delta_n_over_w2 = 1.0 / (pi * r_e * w_matched**4 * n_e_target)

print("GUIDING CHANNEL:")
print(f"  Matched spot size: {w_matched*1e6:.2f} µm")
print(f"  Channel parameter (Δn/n/w²): {rel_delta_n_over_w2*1e12:.4e} m⁻²")
print()

# ==========================================
# DENSITY FUNCTION
# ==========================================

def dens_func(z, r):
    """
    Returns relative density at position z and r
    
    Features:
    - Sine-squared ramp at plasma entrance
    - Transverse parabolic profile for guiding
    
    Parameters:
    -----------
    z, r : 1D arrays
        Position coordinates (m)
        
    Returns:
    --------
    n : 1D array
        Relative density (0 to 1+)
    """
    # Start with uniform density
    n = np.ones_like(z)
    
    # Smooth entrance ramp (sine-squared profile)
    # Smoother than linear ramp, reduces numerical noise
    n = np.where(z < ramp_length, 
                 np.sin(pi/2 * z / ramp_length)**2, 
                 n)
    
    # Suppress density before plasma starts
    n = np.where(z < 0, 0.0, n)
    
    # Add transverse parabolic profile for laser guiding
    # n(r) = n₀ × (1 + Δn/n × r²/w²)
    # This creates a density channel that guides the laser
    n = n * (1.0 + rel_delta_n_over_w2 * r**2)
    
    return n

# ==========================================
# MAIN SIMULATION
# ==========================================

if __name__ == '__main__':
    
    print(f"{'='*70}")
    print(f"STARTING SIMULATION FOR {gas['description'].upper()}")
    print(f"{'='*70}\n")
    
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
    
    # Create NEUTRAL atoms/molecules (q=0)
    # All electrons will come from ionization
    print(f"Adding neutral {gas_type} atoms...")
    atoms = sim.add_new_species(
        q=0,  # NEUTRAL initially
        m=particle_mass,
        n=atomic_density,
        dens_func=dens_func,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    # Create EMPTY electron species
    # Will be populated by ionization
    print("Creating empty electron species...")
    electrons = sim.add_new_species(q=-e, m=m_e)
    
    # ==========================================
    # ACTIVATE FIELD IONIZATION
    # ==========================================
    
    print(f"Activating field ionization for {gas['symbol']}...")
    # Ionize from neutral (level_start=0) to fully stripped
    # All ionized electrons go into the 'electrons' species
    atoms.make_ionizable(
        element=gas['symbol'],
        target_species=electrons,
        level_start=0  # Start from neutral
    )
    
    print(f"Ionization configured: {gas['symbol']}⁰ → {gas['symbol']}^{gas['Z']}⁺ + {gas['Z']}e⁻")
    print()
    
    # ==========================================
    # ADD LASER
    # ==========================================
    
    # print("Adding Gaussian laser pulse...")
    # laser_profile = GaussianLaser(a0, w0, tau, z0, zf=z_foc)
    # add_laser_pulse(sim, laser_profile)
    # print(f"  a₀ = {a0}, w₀ = {w0*1e6:.2f} µm, τ = {tau*1e15:.2f} fs")
    # print()
    
    # ==========================================
    # PARTICLE TRACKING (optional)
    # ==========================================
    
    if use_restart is False:
        if track_electrons:
            print("Enabling particle tracking...")
            electrons.track(sim.comm)
    else:
        print("Restarting from checkpoint...")
        restart_from_checkpoint(sim)
    
    # ==========================================
    # CONFIGURE MOVING WINDOW
    # ==========================================
    
    sim.set_moving_window(v=v_window)
    
    # ==========================================
    # DIAGNOSTICS
    # ==========================================
    
    print(f"Configuring diagnostics (every {diag_period} steps)...")
    sim.diags = [
        # Field diagnostics (E, B, rho, etc.)
        FieldDiagnostic(diag_period, sim.fld, comm=sim.comm),
        
        # Particle diagnostics (position, momentum, etc.)
        ParticleDiagnostic(diag_period, 
                          {"electrons": electrons},
                          comm=sim.comm),
        
        # Charge density diagnostics
        # Useful since field rho is ~0 in neutral plasma
        ParticleChargeDensityDiagnostic(diag_period, sim,
                                       {"electrons": electrons})
    ]
    
    # Checkpoints (optional)
    if save_checkpoints:
        set_periodic_checkpoint(sim, checkpoint_period)
    
    print()
    print(f"{'='*70}")
    print(f"RUNNING SIMULATION: {N_step} iterations")
    print(f"{'='*70}\n")
    
    # ==========================================
    # RUN SIMULATION
    # ==========================================
    
    sim.step(N_step)
    
    print()
    print(f"{'='*70}")
    print(f"SIMULATION COMPLETE!")
    print(f"Output directory: ./diags/hdf5/")
    print(f"\nREMINDER: Rename './diags' to './diags_{gas_type}' before next run!")
    print(f"{'='*70}\n")
