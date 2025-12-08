"""
Calibration Analysis Script for Ionization Injection Tuning

This script analyzes the a0 scan to determine the optimal laser amplitude
for ionization injection. It generates proof plots for phase space separation,
injection thresholds, and wakefield nonlinearity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0, pi
import os
import sys

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ==========================================
# CONFIGURATION
# ==========================================

a0 = 2.5  # Fixed a0 value
modes = ['pure_he', 'doped']
dopant_list = ['N', 'Ne', 'Ar']  # List of dopants to compare
base_dir = './diags_doped'

# Physical parameters
n_e_target = 7.0e24
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
E_wb = 96 * np.sqrt(n_e_target / 1e6) # Cold wavebreaking limit (V/m) approx formula

# ==========================================
# DATA LOADING
# ==========================================

def load_data(a0,dopant_species, mode):
    if mode == 'doped':
        path = f"{base_dir}/a{a0}_{mode}_{dopant_species}/hdf5"
    else:
        path = f"{base_dir}/a{a0}_{mode}/hdf5"
    if not os.path.exists(path):
        print(f"Warning: Data not found for a0={a0}, mode={mode}, dopant={dopant_species} at {path}")
        return None
    return LpaDiagnostics(path)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def make_alpha_smooth(n_bins, x0=0.45, k=6.0, alpha_min=0.0, alpha_max=1.0):
    """
    Smooth logistic alpha, explicitly mapped so:
      r = 0  -> alpha_min
      r = 1  -> alpha_max
    """
    r = np.linspace(0.0, 1.0, n_bins)

    # raw logistic
    L = 1.0 / (1.0 + np.exp(-k * (r - x0)))

    # values at the ends of [0,1]
    L0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    L1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))

    # normalize to [0, 1] on this interval
    Ln = (L - L0) / (L1 - L0)

    alpha = alpha_min + (alpha_max - alpha_min) * Ln
    return np.clip(alpha, 0.0, 1.0)


def get_transparent_inferno(
    n_bins=512,
    desat_factor=0.3,
    x0=0.45,
    k=6.0,
    alpha_min=0.0,
    alpha_max=1.0,
):
    inferno = plt.get_cmap("inferno")
    # clip dark part to remove black
    colors = inferno(np.linspace(0.20, 1.00, n_bins))

    # desaturate toward white
    if desat_factor > 0:
        white = np.array([1.0, 1.0, 1.0])
        colors[:, :3] = colors[:, :3] * (1 - desat_factor) + white * desat_factor

    # smooth alpha
    alpha = make_alpha_smooth(
        n_bins,
        x0=x0,
        k=k,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    colors[:, 3] = alpha

    return mcolors.LinearSegmentedColormap.from_list("inferno_smooth", colors)

# ==========================================
# PLOTS
# ==========================================

def plot_phase_space(a0_target, dopant_species):
    print(f"\nGenerating Plot : Phase Space Separation ({dopant_species}-doped)...")
    
    ts = load_data(a0_target, dopant_species, 'doped')
    if ts is None:
        return

    iteration = ts.iterations[-1] # Change interation if needed
    
    # Get data for both species
    # electrons from helium and preionised dopant -> 'electrons_bulk'
    # outer shell electrons from  -> 'electrons_injected'
    
    try:
        z_bulk, uz_bulk, w_bulk = ts.get_particle(['z', 'uz', 'w'], species='electrons_bulk', iteration=iteration)
    except:
        print("Warning: 'electrons_bulk' not found, trying 'electrons_he' (legacy)")
        z_bulk, uz_bulk, w_bulk = ts.get_particle(['z', 'uz', 'w'], species='electrons_he', iteration=iteration)

    try:
        z_injected, uz_injected, w_injected = ts.get_particle(['z', 'uz', 'w'], species='electrons_injected', iteration=iteration)
    except:
        # Fallback or empty if no injection
        print("Warning: 'electrons_injected' not found, trying 'electrons_dopant' (legacy)")
        try:
            z_injected, uz_injected, w_injected = ts.get_particle(['z', 'uz', 'w'], species='electrons_dopant', iteration=iteration)
        except:
             z_injected, uz_injected, w_injected = np.array([]), np.array([]), np.array([])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Dopant electrons (Injected) - Plot FIRST (Background layer)
    if len(z_injected) > 0:
        ax.scatter(z_injected*1e6, uz_injected, 
                   s=5, color='red', alpha=0.3, label=f'{dopant_species} Electrons (Injected)', zorder=2)
    else:
        ax.text(0.5, 0.5, f"No {dopant_species} Electrons Injected", 
                transform=ax.transAxes, ha='center', color='red')

    # Plot Helium electrons (Bulk) - Plot SECOND (Foreground layer)
    if len(z_bulk) > 0:
        indices_bulk = np.random.choice(len(z_bulk), size=min(10000, len(z_bulk)), replace=False)
        ax.scatter(z_bulk[indices_bulk]*1e6, uz_bulk[indices_bulk], 
                   s=5, color='blue', alpha=0.5, label='Bulk Electrons', zorder=1)
    
    ax.set_xlabel('z (µm)')
    ax.set_ylabel('$p_z / m_e c$')
    ax.set_title(f'Phase Space Separation ({dopant_species}-doped)')
    ax.legend()

    filename = f'{dopant_species}_phase_space.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

def plot_dopant_comparison():
    print(f"\nGenerating Plot : Injected Charge Comparison (a0={a0})...")
    
    # Metric: Total Injected Charge (pC)
    # We define "Injected" as E > 2 MeV to exclude bulk fluid
    E_threshold_MeV = 2.0
    
    charges = []
    species_labels = []

    # Calculate for each dopant
    for species in dopant_list:
        ts = load_data(a0, species, 'doped')
        if ts is None:
            charges.append(0.0)
            species_labels.append(species)
            continue
            
        iteration = ts.iterations[-1]
        try:
            gamma_threshold = 1 + E_threshold_MeV / 0.511
            uz_threshold = np.sqrt(gamma_threshold**2 - 1)
            
            # Try new species name first, then legacy
            try:
                I_dopant, info_dopant = ts.get_current(species='electrons_injected', iteration=iteration,
                                                    select={'uz': [uz_threshold, None]})
            except:
                I_dopant, info_dopant = ts.get_current(species='electrons_dopant', iteration=iteration,
                                                    select={'uz': [uz_threshold, None]})
                                                    
            Q_dopant = np.sum(np.abs(I_dopant)) * info_dopant.dz / c * 1e12  # pC
            charges.append(Q_dopant)
            species_labels.append(species)
        except Exception as err:
            print(f"  Error reading {species} data: {err}")
            charges.append(0.0)
            species_labels.append(species)

    # Also calculate Pure He noise level
    ts_he = load_data(a0, None, 'pure_he')
    Q_he = 0.0
    if ts_he is not None:
        try:
            iteration = ts_he.iterations[-1]
            gamma_threshold = 1 + E_threshold_MeV / 0.511
            uz_threshold = np.sqrt(gamma_threshold**2 - 1)
            
            # Try new species name first, then legacy
            try:
                I_he, info_he = ts_he.get_current(species='electrons_bulk', iteration=iteration, 
                                               select={'uz': [uz_threshold, None]})
            except:
                I_he, info_he = ts_he.get_current(species='electrons_he', iteration=iteration, 
                                               select={'uz': [uz_threshold, None]})
                                               
            Q_he = np.sum(np.abs(I_he)) * info_he.dz / c * 1e12
        except Exception as err:
            print(f"  Error reading He data: {err}")

    # Plot Bar Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(species_labels, charges, color=['red', 'green', 'blue'], alpha=0.7, label='Doped (Signal)')
    
    # Add He noise line
    ax.axhline(y=Q_he, color='grey', linestyle='--', label='Pure He (Noise)')
    
    ax.set_ylabel(f'Injected Charge (pC) [E > {E_threshold_MeV} MeV]')
    ax.set_title(f'Injected Charge Comparison (a0={a0})')
    ax.legend()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('dopant_charge_comparison.png', dpi=300)
    print("Saved: dopant_charge_comparison.png")
    plt.close()

def plot_e_density(a0_target, dopant_species):
    """
    Plots the plasma density along with the laser profile using correct masking and scaling.
    """
    print(f"\nGenerating Plot : Wakefield Structure (a0={a0_target}, dopant={dopant_species})...")

    mode = 'doped'
    label = f'{dopant_species}-Doped'

    # Load Data (Assuming load_data is defined elsewhere in your scope)
    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration or change it
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
    
    # --- 1. GET PLASMA DENSITY ---
    rho, info_rho = ts.get_field(field='rho', iteration=iteration)
    
    # Convert to relative density (assuming n_e_target and e are global constants)
    # Note: ensure n_e_target and e are defined in your script
    rho_rel = -rho / n_e_target / e 
    
    # Get coordinates for extent
    z_rho = info_rho.z * 1e6 # Convert to microns
    r_rho = info_rho.r * 1e6
    extent_rho = [z_rho.min(), z_rho.max(), r_rho.min(), r_rho.max()]

    # --- 2. GET LASER ENVELOPE ---
    laser, info_laser = ts.get_laser_envelope(iteration=iteration, pol='x', m=1)
    
    # Get coordinates for laser extent (CRITICAL FOR ALIGNMENT)
    z_laser = info_laser.z * 1e6 # Convert to microns
    r_laser = info_laser.r * 1e6
    extent_laser = [z_laser.min(), z_laser.max(), r_laser.min(), r_laser.max()]

    # --- 3. GET ELECTRIC FIELD Ez ---
    Ez, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration, m=0, slice_across='r')
    z_Ez = info_Ez.z * 1e6 # Convert to microns
    Ez_GV = Ez / 1e9       # Convert to GV/m

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 5)) # Slightly wider for better aspect ratio

    # Layer 1: Plasma Density

    colors = ['1', '0.5', '0']
    nodes = [0.0, 0.01, 1.0]
    cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))


    im_rho = ax.imshow(rho_rel,
                       extent=extent_rho,
                       origin='lower',
                       aspect='auto',
                       cmap='grey_r',
                       vmin=0, vmax=1)
    
    # Colorbar for Density
    cbar = plt.colorbar(im_rho, ax=ax, pad=0.02)
    cbar.set_label(r'$n_e/n_0$', fontsize=10)
    
    # Layer 2: Laser envelope
    
    cmp_laser = get_transparent_inferno(
    n_bins=512,
    desat_factor=0.3,
    x0=0.2,     # move transition toward higher intensities
    k=10.0,      # smaller k = softer, more gradual fade
    alpha_min=0.0,  # don't let outskirts go fully transparent
    alpha_max=0.85)


    im_laser = ax.imshow(laser,
                         cmap=cmp_laser,
                         interpolation='bicubic',
                         extent=extent_laser, # THIS FIXES THE SQUISHED LINE
                         origin='lower',
                         aspect='auto',
                         vmin=0)

    # --- TWIN AXIS FOR Ez ---
    ax2 = ax.twinx()
    
    # Plot Ez on top
    # Using a bright color (like Cyan or Blue) often pops better against black/inferno
    ax2.plot(z_Ez, Ez_GV, color='cyan', linewidth=1.5, alpha=0.9, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Manually setting x-limits ensures both plots are locked to the same view
    ax.set_xlim(z_rho.min(), z_rho.max())
    
    # Labels
    ax.set_xlabel(r'$z \; (\mu m)$', fontsize=12)
    ax.set_ylabel(r'$r \; (\mu m)$', fontsize=12)
    ax.set_title(f"{label} (t = {t_fs:.1f} fs)")
    
    # Right Axis Styling
    ax2.set_ylabel(r'$E_z$ (GV/m)', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    ax2.set_ylim(-np.max(np.abs(Ez_GV))*1.5, np.max(np.abs(Ez_GV))*1.5) # Auto-scale y-axis symmetrically

    # Legend
    # Combine legend handles if needed, or just place Ez label manually
    ax2.text(0.02, 0.9, r'$E_z$ Field', transform=ax2.transAxes, color='cyan', fontweight='bold')

    plt.tight_layout()
    
    filename = f'{dopant_species}_wakefield_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_e_injected(a0_target, dopant_species):
    print(f"\nGenerating Plot : Injected Electron Density (a0={a0_target}, dopant={dopant_species})...")
    mode = 'doped'
    label = f'{dopant_species}-Doped'

    # Load Data
    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15

    # --- GET INJECTED DENSITY ---

    rho, info_rho = ts.get_field(field='rho_electrons_injected', iteration=iteration)
    
    # Convert to number density (m^-3)
    # rho is charge density (C/m^3). Electrons have negative charge.
    n_injected = -rho / e
    
    # Get coordinates
    z = info_rho.z * 1e6 # microns
    r = info_rho.r * 1e6 # microns
    extent = [z.min(), z.max(), r.min(), r.max()]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot density
    im = ax.imshow(n_injected, extent=extent, origin='lower', aspect='auto', cmap='inferno')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$n_{injected} \; (m^{-3})$')
    
    ax.set_xlabel(r'$z \; (\mu m)$')
    ax.set_ylabel(r'$r \; (\mu m)$')
    ax.set_title(f"Injected Electron Density - {label} (t = {t_fs:.1f} fs)")
    
    plt.tight_layout()
    filename = f'{dopant_species}_injected_charge_density.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_dopant_emittance(a0_target, dopant_species):
    print(f"\nGenerating Plot : Injected Electron Emittance (a0={a0_target}, dopant={dopant_species})...")
    mode = 'doped'
    label = f'{dopant_species}-Doped'

    # Load Data
    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15

    # Calculate emittance

    emittance_x, emittance_y = ts.get_emittance(species='electrons_injected', iteration=iteration, description='projected', select={'uz': [100, None]})
    print(emittance_x, emittance_y)

    # Calculate error of the emittance
    # 1. Get the raw particle arrays using the same selection as before
    x, ux, w = ts.get_particle(
        var_list=['x', 'ux', 'w'], 
        species='electrons_injected', 
        iteration=iteration, 
        select={'uz': [100, None]}  # YOUR ENERGY FILTER
    )

    # # 2. Calculate Effective Number of Particles (taking weights into account)
    # # If weights are uniform, N_eff = len(w). If weights vary, use Kish's formula:
    # N_eff = (np.sum(w)**2) / np.sum(w**2)

    # # 3. Calculate Relative Error
    # rel_error = 1.0 / np.sqrt(2 * N_eff)

    # print(f"Number of macroparticles used: {len(w)}")
    # print(f"Effective N: {N_eff:.2f}")
    # print(f"Estimated Relative Error: {rel_error * 100:.2f}%")

    def calc_emittance(x, ux, w):
        avg_x = np.average(x, weights=w)
        avg_ux = np.average(ux, weights=w)
        # Centered moments
        cov_xx = np.average((x - avg_x)**2, weights=w)
        cov_uu = np.average((ux - avg_ux)**2, weights=w)
        cov_xu = np.average((x - avg_x)*(ux - avg_ux), weights=w)
        return np.sqrt(cov_xx*cov_uu - cov_xu**2)

    # 3. Bootstrap (Resampling)
    n_resamples = 50
    emittance_values = []
    n_particles = len(x)

    for _ in range(n_resamples):
        # Random indices with replacement
        indices = np.random.randint(0, n_particles, n_particles)
        e_val = calc_emittance(x[indices], ux[indices], w[indices])
        emittance_values.append(e_val)

    # 4. Results
    mean_emit = np.mean(emittance_values)
    std_error = np.std(emittance_values)

    print(f"Emittance: {mean_emit:.2e} +/- {std_error:.2e} m-rad")
    print(f"Relative Error: {std_error/mean_emit * 100:.2f}%")



# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Run Charge Comparison
    # plot_dopant_comparison()
    
    # plot_phase_space(a0, dopant_species=dopant_list[0])
    # plot_dopant_comparison()
    # plot_e_density(a0, 'Ar')
    # plot_laser_envelope()


    # Run detailed plots for each dopant at fixed a0
    for species in dopant_list:
    #     plot_phase_space(a0_target=a0, dopant_species=species)
    #     plot_e_density(a0_target=a0, dopant_species=species)
        pass
    
