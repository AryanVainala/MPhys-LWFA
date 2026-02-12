"""
Calibration Analysis Script for Ionization Injection Tuning

This script analyzes the a0 scan to determine the optimal laser amplitude
for ionization injection. It generates proof plots for phase space separation,
injection thresholds, and wakefield nonlinearity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0, pi
import os
import sys

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': "Helvetica",
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


# ==========================================
# CONFIGURATION
# ==========================================

a0 = 2.5  # Fixed a0 value
modes = ['pure_he', 'doped']
dopant_list = ['N', 'Ar', 'Ne']  # List of dopants to compare
base_dir = './diags_doped'

# Physical parameters
n_e_target = 3.5e24
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
E_wb = 96 * np.sqrt(n_e_target / 1e6) # Cold wavebreaking limit (V/m) approx formula

# Energy threshold for injected electrons
E_threshold_MeV = 50
gamma_threshold = 1 + (E_threshold_MeV * 1e6 * e) / (m_e * c**2)
uz_threshold = np.sqrt(gamma_threshold**2 - 1)

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
    plt.savefig(filename, dpi=800)
    print(f"Saved: {filename}")
    plt.close()

def plot_e_density(a0_target, dopant_species):
    """
    Plots the plasma density along with the laser profile using correct masking and scaling.
    """
    print(f"\nGenerating Plot : Wakefield Structure (a0={a0_target}, dopant={dopant_species})...")

    mode = 'doped'

    # Load Data (Assuming load_data is defined elsewhere in your scope)
    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration or change it
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    t_ps = ts.t[ts.iterations.tolist().index(iteration)] * 1e12
    
    # --- 1. GET PLASMA DENSITY ---
    # Try to get bulk electrons specifically to avoid plotting injected electrons in the background
    try:
        rho, info_rho = ts.get_field(field='rho_electrons_bulk', iteration=iteration)
        print("Using 'rho_electrons_bulk' for wakefield structure.")
    except:
        print("Could not find 'rho_electrons_bulk', falling back to total 'rho'.")
        rho, info_rho = ts.get_field(field='rho', iteration=iteration)
    
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

    # --- 4. GET INJECTED ELECTRON DENSITY ---
    n_inj = None
    extent_inj = None
    try:
        rho_inj, info_inj = ts.get_field(field='rho_electrons_injected', iteration=iteration)
        z_inj = info_inj.z * 1e6
        r_inj = info_inj.r * 1e6
        extent_inj = [z_inj.min(), z_inj.max(), r_inj.min(), r_inj.max()]
    except Exception as err:
        print(f"Could not load injected electrons: {err}")

    # --- PLOTTING ---
    # Use gridspec for proper colorbar placement
    fig = plt.figure(figsize=(12, 12))
    
    # Create gridspec: main plot on top, two colorbars at the bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1],
                          hspace=0.4, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, :])  # Main plot spans both columns
    ax.set_axisbelow(True)          # Ensure grid lines are behind data
    cax_bulk = fig.add_subplot(gs[1, 0])  # Left colorbar for bulk
    cax_inj = fig.add_subplot(gs[1, 1])   # Right colorbar for injected

    # Convert to relative density
    n_bulk = -rho / e
    n_inj =  -rho_inj / e

    # Calculate n0
    z_indices = n_bulk.shape[1]
    sample_region = int(z_indices * 0.9) 
    
    # Take the median of the far-right region to exclude any weird noise spikes
    n0 = np.median(n_bulk[:, sample_region:])

    # Normalise the densities
    n_bulk = -rho / e / n0
    n_inj =  -rho_inj / e / n0

    # Layer 1: Plasma Density (Bulk)
    im_rho = ax.imshow(n_bulk,
                       extent=extent_rho,
                       origin='lower',
                       aspect='auto',
                       cmap=cmc.grayC_r,
                       interpolation='bilinear',
                       vmin = np.percentile(n_bulk, 0.01), # set as 0.01 for Neon otherwise 0
                       vmax = np.percentile(n_bulk, 99.9)
            )
    im_rho.cmap.set_over('black')

    # Horizontal colorbar for bulk density at bottom left
    cbar = plt.colorbar(im_rho, cax=cax_bulk, orientation='horizontal')
    cbar.set_label(r'Bulk Plasma: $n_e/n_0$', fontsize=14)

    # Layer 2: Laser envelope (Middle Layer)
    cmp_laser = get_transparent_inferno(
        n_bins=512,
        desat_factor=0.1,
        x0=0.2,     # move transition toward higher intensities
        k=12.0,      # larger k = softer, more gradual fade
        alpha_min=0.0,  # 0 means outskirts are fully transparent
        alpha_max=0.85)

    im_laser = ax.imshow(laser,
                         cmap=cmp_laser,
                         interpolation='bicubic',
                         extent=extent_laser,
                         origin='lower',
                         aspect='auto',
                         vmin=0)

    # Layer 1.5: Injected Electrons (Top Layer)
    im_inj = None
    if n_inj is not None:

        # Create a transparent-to-hot colormap
        n_bins_inj = 5000
        colors_inj = plt.cm.BuGn(np.linspace(0.9, 1, n_bins_inj))
        # colors_inj[:50, 3] = np.linspace(0, 0.7, 50)  # Faster transparency fade at low values
        # colors_inj[50:, 3] = np.linspace(0.5, 0.9, n_bins_inj-50)  # Rest is more opaque
        alpha_curve = np.linspace(0, 1, n_bins_inj) ** 0.5
        colors_inj[:, 3] = alpha_curve
        cmap_inj = LinearSegmentedColormap.from_list('blues_alpha', colors_inj)

        im_inj = ax.imshow(n_inj,
                           extent=extent_inj,
                           origin='lower',
                           aspect='auto',
                           cmap=cmap_inj,
                           interpolation='bilinear',
                           vmin = np.percentile(n_inj, 98.99),   # For Neon clip it np.percentile(n_inj, 98.99)  |  Otherwise np.percentile(n_inj, 0)
                           vmax = np.percentile(n_inj, 99.99)    # and np.percentile(n_inj, 99.9)                |  and  np.percentile(n_inj, 0)
              )
        
        # Horizontal colorbar for injected electrons at bottom right
        cbar_inj = plt.colorbar(im_inj, cax=cax_inj, orientation='horizontal')
        cbar_inj.set_label(r'Injected Electrons: $n_e/n_0$', fontsize=14)
    else:
        cax_inj.axis('off')  # Hide if no injected electrons

    # --- TWIN AXIS FOR Ez ---
    ax2 = ax.twinx()
    
    # Plot Ez on top
    ax2.plot(z_Ez, Ez_GV, color="#0606B6", linewidth=1.5, alpha=0.9, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Manually setting x-limits ensures both plots are locked to the same view
    ax.set_xlim(z_rho.min(), z_rho.max())
    
    # Labels
    ax.set_xlabel(r'$z \; (\mu m)$')
    ax.set_ylabel(r'$r \; (\mu m)$')
    ax.set_title(f"{dopant_species}-Doped He Wakefield at (t = {t_ps:.2f} ps)")
    
    # Right Axis Styling
    ax2.set_ylabel(r'$E_z$ (GV/m)', color='#0606B6')
    ax2.tick_params(axis='y', labelcolor='#0606B6')
    ax2.set_ylim(-1000, 1000)

    # Don't use tight_layout with gridspec - it's already handled
    
    filename = f'{dopant_species}_wakefield_fixed_last.png'
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_e_density_pure_he(a0_target):
    print(f"\nGenerating Plot: Pure Helium Wakefield (a0={a0_target})...")
    
    # Load Data
    ts = load_data(a0_target, None, 'pure_he')
    if ts is None:
        return

    iteration = ts.iterations[1]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e12

    # --- 1. GET BULK PLASMA DENSITY ---
    # In pure_he mode, we use rho_electrons_bulk
    rho, info_rho = ts.get_field(field='rho_electrons_bulk', iteration=iteration)
    z_rho = info_rho.z * 1e6 # Convert to microns
    r_rho = info_rho.r * 1e6
    extent_rho = [z_rho.min(), z_rho.max(), r_rho.min(), r_rho.max()]

    # --- 2. GET LASER ENVELOPE ---
    laser, info_laser = ts.get_laser_envelope(iteration=iteration, pol='x', m=1)
    z_laser = info_laser.z * 1e6
    r_laser = info_laser.r * 1e6
    extent_laser = [z_laser.min(), z_laser.max(), r_laser.min(), r_laser.max()]

    # --- 3. GET ELECTRIC FIELD Ez ---
    Ez, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration, m=0, slice_across='r')
    z_Ez = info_Ez.z * 1e6
    Ez_GV = Ez / 1e9

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 10))
    
    # Create gridspec: main plot on top, one colorbar at the bottom
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05], hspace=0.25)
    
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axisbelow(True)          # Ensure grid lines are behind data
    cax_bulk = fig.add_subplot(gs[1, 0])

    # Convert to relative density
    n_bulk = -rho / e
    
    # Calculate n0
    z_indices = n_bulk.shape[1]
    sample_region = int(z_indices * 0.9) 
    n0 = np.median(n_bulk[:, sample_region:])
    n_bulk = n_bulk / n0

    # Layer 1: Plasma Density (Bulk)
    im_rho = ax.imshow(n_bulk,
                       extent=extent_rho,
                       origin='lower',
                       aspect='auto',
                       cmap=cmc.grayC_r,
                       interpolation='bilinear',
                       vmin = np.percentile(n_bulk, 0.01),
                       vmax = np.percentile(n_bulk, 99.9)
            )
    im_rho.cmap.set_over('black')

    # Horizontal colorbar for bulk density
    cbar = plt.colorbar(im_rho, cax=cax_bulk, orientation='horizontal')
    cbar.set_label(r'Bulk Plasma: $n_e/n_0$', fontsize=14)

    # Layer 2: Laser envelope
    cmp_laser = get_transparent_inferno(
        n_bins=512,
        desat_factor=0.1,
        x0=0.2,
        k=12.0,
        alpha_min=0.0,
        alpha_max=0.85)

    im_laser = ax.imshow(laser,
                         cmap=cmp_laser,
                         interpolation='bicubic',
                         extent=extent_laser,
                         origin='lower',
                         aspect='auto',
                         vmin=0)

    # --- TWIN AXIS FOR Ez ---
    ax2 = ax.twinx()
    ax2.plot(z_Ez, Ez_GV, color="#0606B6", linewidth=1.5, alpha=0.9, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.set_xlim(z_rho.min(), z_rho.max())
    ax.set_xlabel(r'$z \; (\mu m)$')
    ax.set_ylabel(r'$r \; (\mu m)$')
    ax.set_title(f"Pure Helium Wakefield at (t = {t_fs:.2f} ps)")
    
    ax2.set_ylabel(r'$E_z$ (GV/m)', color='#0606B6')
    ax2.tick_params(axis='y', labelcolor='#0606B6')
    ax2.set_ylim(-1000, 1000)

    filename = 'pure_he_wakefield.png'
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_energy_spectra_comparison(a0_target):
    print(f"\nGenerating Plot : Energy Spectra Comparison (Stacked) (a0={a0_target})...")
    
    # Define plot order and parameters
    plot_configs = [
        {'label': 'Pure He', 'species': None, 'mode': 'pure_he', 'color': (0.5, 0.5, 0.5), 'alpha': 0.4},
        {'label': 'N-doped He', 'species': 'N',   'mode': 'doped',   'color': (191/255, 44/255, 35/255), 'alpha': 0.4}, # Vermillion
        {'label': 'Ne-doped He', 'species': 'Ne',  'mode': 'doped',   'color': (0, 158/255, 115/255), 'alpha': 0.4}, # Bluish Green
        {'label': 'Ar-doped He',   'species': 'Ar',  'mode': 'doped',   'color': (0, 114/255, 178/255), 'alpha': 0.4}  # Blue
    ]
    
    # Setup figure: 4 rows, 1 column, sharing x-axis
    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0.1})
    
    # Main plot threshold
    E_min_main = 0 # Start from 0 MeV as requested
    bin_width_main = 1 # MeV
    
    for ax, config in zip(axes, plot_configs):
        species = config['species']
        mode = config['mode']
        color = config['color']
        label = config['label']
        alpha = config['alpha']
        
        ts = load_data(a0_target, species, mode)
        
        if ts is None:
            ax.text(0.5, 0.5, "Data Not Available", transform=ax.transAxes, ha='center')
            continue
            
        iteration = ts.iterations[-1]
        
        # Determine particle species name to fetch
        if mode == 'pure_he':
            species_names = ['electrons_bulk', 'electrons_he']
        else:
            species_names = ['electrons_injected', 'electrons_dopant']
            
        ux, uy, uz, w = np.array([]), np.array([]), np.array([]), np.array([])
        
        for s_name in species_names:
            try:
                ux, uy, uz, w = ts.get_particle(['ux', 'uy', 'uz', 'w'], species=s_name, iteration=iteration)
                if len(uz) > 0:
                    break
            except:
                continue
                
        if len(uz) == 0:
            ax.text(0.5, 0.5, "No Particles Found", transform=ax.transAxes, ha='center')
            continue
            
        # Calculate Energy (Rigorous)
        gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
        E_MeV = (gamma - 1) * (m_e * c**2 / (e * 1e6))
        
        # --- MAIN PLOT DATA ---
        mask_main = E_MeV > E_min_main
        E_main = E_MeV[mask_main]
        w_main = w[mask_main]
        
        if len(E_main) > 0:
            # Calculate Statistics
            total_charge_pC = np.sum(w_main) * e * 1e12
            avg_energy_MeV = np.average(E_main, weights=w_main)

            E_max = np.max(E_main)
            # Ensure at least one bin
            if E_max <= E_min_main:
                 bins = np.array([E_min_main, E_min_main + bin_width_main])
            else:
                 bins = np.arange(E_min_main, np.ceil(E_max) + bin_width_main, bin_width_main)
            
            hist_weights, bin_edges = np.histogram(E_main, bins=bins, weights=w_main)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dQ_dE = (hist_weights * e * 1e12) / bin_width_main
            
            # Plot
            ax.plot(bin_centers, dQ_dE, color=color, linewidth=2.5)
            ax.fill_between(bin_centers, dQ_dE, color=color, alpha=alpha, label=label)
            
            # Add Statistics Text
            stats_text = f"$Q_{{tot}}$: {total_charge_pC:.1f} pC\n$\\langle E \\rangle$: {avg_energy_MeV:.1f} MeV"
            ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Styling per subplot
        ax.set_ylabel(r"$dQ/dE$ (pC/MeV)")
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-4) # Set a small positive floor for log scale
        ax.legend(loc='upper left')
        ax.grid(True, which='both', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)

    # Global Styling
    axes[-1].set_xlabel(r"Energy (MeV)")
    axes[-1].set_xlim(left=E_min_main)
    
    # fig.suptitle(f"Injected Electron Energy Spectra (E > {E_min_main} MeV) (a0={a0_target})", y=0.92)

    plt.tight_layout()
    filename = f'energy_spectra_comparison_stacked_a{a0_target}.png'
    plt.savefig(filename, dpi=800)
    print(f"Saved: {filename}")
    plt.close()

def plot_energy_divergence_comparison(a0_target):
    print(f"\nGenerating Plot : Energy-Divergence Spectra (Stacked) (a0={a0_target})...")
    
    # Define plot order and parameters (excluding Ne)
    plot_configs = [
        {'label': 'Pure He', 'species': None, 'mode': 'pure_he'},
        {'label': 'N-doped He', 'species': 'N',   'mode': 'doped'},
        {'label': 'Ar-doped He',   'species': 'Ar',  'mode': 'doped'}
    ]
    
    # Setup figure: 3 rows, 1 column, sharing x-axis
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3, left=0.15, right=0.75, bottom=0.15)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    
    # Main plot threshold
    E_min_main = E_threshold_MeV 
    E_max_fixed = 400 # Fixed maximum energy for alignment
    theta_lim = 20 # mrad
    theta_int_lim = 3 # mrad for dQ/dE integration
    
    # To store the last image for colorbar
    last_im = None
    
    for i, (ax, config) in enumerate(zip(axes, plot_configs)):
        species = config['species']
        mode = config['mode']
        label = config['label']
        
        # Apply styling to all axes regardless of data
        ax.set_ylabel(r'$\theta_x$ (mrad)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', colors='black', labelsize=12, length=6, width=1.5)
        ax.set_ylim(-theta_lim, theta_lim)
        ax.text(0.02, 0.9, label, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
        
        # Remove x-ticks for top plots
        if i < 2:
            ax.set_xticklabels([])

        ts = load_data(a0_target, species, mode)
        
        if ts is None:
            ax.text(0.5, 0.5, "Data Not Available", transform=ax.transAxes, ha='center', color='white')
            ax.set_facecolor('black')
            ax.set_xlim(E_min_main, E_max_fixed)
            continue
            
        iteration = ts.iterations[-1]
        
        # Determine particle species name
        if mode == 'pure_he':
            species_names = ['electrons_bulk', 'electrons_he']
        else:
            species_names = ['electrons_injected', 'electrons_dopant']
            
        ux, uy, uz, w = np.array([]), np.array([]), np.array([]), np.array([])
        
        for s_name in species_names:
            try:
                ux, uy, uz, w = ts.get_particle(['ux', 'uy', 'uz', 'w'], species=s_name, iteration=iteration)
                if len(uz) > 0:
                    break
            except:
                continue
                
        if len(uz) == 0:
            ax.text(0.5, 0.5, "No Particles Found", transform=ax.transAxes, ha='center', color='white')
            ax.set_facecolor('black')
            ax.set_xlim(E_min_main, E_max_fixed) # Ensure alignment even if empty
            continue
            
        # Calculate Energy and Divergence
        gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
        E_MeV = (gamma - 1) * (m_e * c**2 / (e * 1e6))
        theta_x_mrad = np.arctan2(ux, uz) * 1000
        
        # Filter
        mask = E_MeV > E_min_main
        E_sel = E_MeV[mask]
        theta_sel = theta_x_mrad[mask]
        w_sel = w[mask]
        
        if len(E_sel) == 0:
            ax.text(0.5, 0.5, "No Particles > Threshold", transform=ax.transAxes, ha='center', color='white')
            ax.set_facecolor('black')
            ax.set_xlim(E_min_main, E_max_fixed) # Ensure alignment even if empty
            continue

        # Binning
        # Changed np.max(E_sel) to E_max_fixed to ensure identical bin width and alignment across all plots
        bins_E = np.linspace(E_min_main, E_max_fixed, 200)
        bins_theta = np.linspace(-theta_lim, theta_lim, 100)
        
        # 2D Histogram (Density)
        H, xedges, yedges = np.histogram2d(E_sel, theta_sel, bins=[bins_E, bins_theta], weights=w_sel)
        dE = bins_E[1] - bins_E[0]
        dTheta = bins_theta[1] - bins_theta[0]
        
        # Convert to pC / (MeV * mrad)
        Density = (H.T * e * 1e12) / (dE * dTheta)
        
        # Plot Heatmap
        im = ax.imshow(Density, origin='lower', aspect='auto',
                       extent=[bins_E[0], bins_E[-1], bins_theta[0], bins_theta[-1]],
                       cmap='magma', vmin=0)
        
        # Set fixed x-limits for alignment
        ax.set_xlim(E_min_main, E_max_fixed)
        
        # Add grey box for integration limits
        ax.axhspan(-theta_int_lim, theta_int_lim, color='grey', alpha=0.3, zorder=0)
        
        # 1D Projections
        centers_Theta = (bins_theta[:-1] + bins_theta[1:]) / 2
        centers_E = (bins_E[:-1] + bins_E[1:]) / 2

        # dQ/dE (White Line) - Integrate d^2 Q/dE dTheta_x over Theta_x within grey box limits (+/- 5 mrad)
        mask_theta = (centers_Theta >= -theta_int_lim) & (centers_Theta <= theta_int_lim)
        dQ_dE = np.sum(H[:, mask_theta], axis=1) * e * 1e12 / dE  # Sum over theta bins, result is pC/MeV
        
        # dQ/dTheta (Green Line) - Integrate d^2 Q/dE dTheta_x over all Energy
        dQ_dTheta = np.sum(H, axis=0) * e * 1e12 / dTheta  # Sum over energy bins, result is pC/mrad
        
        # --- LEFT AXIS: dQ/dE (White Line) ---
        # Use twinx and move the axis to the left side using spines
        # Since we are moving the main y-axis to the right, we can put this on the left directly
        ax_dE = ax.twinx()
        ax_dE.spines['left'].set_position(('axes', 0))  # Standard left
        ax_dE.spines['left'].set_visible(True)
        ax_dE.spines['right'].set_visible(False)
        ax_dE.yaxis.set_label_position('left')
        ax_dE.yaxis.set_ticks_position('left')
        
        ax_dE.plot(centers_E, dQ_dE, color='white', linewidth=1.5, alpha=0.9)
        max_dq_de = np.max(dQ_dE) if np.max(dQ_dE) > 0 else 1
        ax_dE.set_ylim(0, max_dq_de * 1.3)
        ax_dE.set_ylabel(r"$dQ/dE$ (pC/MeV)", color='black', fontsize=15)
        ax_dE.tick_params(axis='y', colors='black', labelsize=12)
        ax_dE.spines['left'].set_color('black')
        ax_dE.grid(False)
        
        # --- RIGHT SIDE: dQ/dθx (Sky Blue Line) ---
        # Use inset_axes to create a small plot on the right
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_dTheta = inset_axes(ax, width="15%", height="100%", loc='center right',
                               bbox_to_anchor=(0.28, 0, 1, 1), bbox_transform=ax.transAxes,
                               borderpad=0)

        
        # Plot dQ/dθx as vertical profile
        div_color = '#56B4E9' # Sky Blue (Colorblind friendly)
        ax_dTheta.plot(dQ_dTheta, centers_Theta, color=div_color, linewidth=1.5, alpha=0.9)
        
        # Fill the entire area with sky blue first to ensure no gaps (the "base" layer)
        ax_dTheta.fill_betweenx(centers_Theta, 0, dQ_dTheta, color=div_color, alpha=0.2)
        
        # Overlay the integration limits in grey. 
        # This sits on top of the sky blue, ensuring the transition is perfectly flush.
        ax_dTheta.fill_betweenx(centers_Theta, 0, dQ_dTheta, 
                                where=(np.abs(centers_Theta) <= theta_int_lim),
                                color='grey', alpha=0.3, interpolate=True)

        max_dq_dtheta = np.max(dQ_dTheta) if np.max(dQ_dTheta) > 0 else 1
        ax_dTheta.set_xlim(0, max_dq_dtheta * 1.3)
        ax_dTheta.set_ylim(-theta_lim, theta_lim)
        ax_dTheta.set_yticks(np.arange(-theta_lim, theta_lim + 5, 10))
        ax_dTheta.set_xlabel(r"$dQ/d\theta_x$" + "\n(pC/mrad)", color='black', fontsize=12)
        ax_dTheta.tick_params(axis='x', colors='black', labelsize=12,)
        ax_dTheta.tick_params(axis='y', labelleft=True, labelsize=12, colors='black')
        ax_dTheta.set_facecolor('black')
        ax_dTheta.grid(True)

        # Styling
        ax.set_ylabel(r'$\theta_x$ (mrad)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', colors='black', labelsize=12, length=6, width=1.5)
        ax.set_ylim(-theta_lim, theta_lim)
        ax.text(0.02, 0.9, label, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')

        last_im = im

    # Force consistent x-axis tick marks for all subplots
    for ax in axes:
        ax.set_xticks(np.arange(100, 401, 50)) # Extended slightly to include 400
        ax.set_xlim(E_min_main, E_max_fixed)
    
    axes[-1].set_xlabel(r"Energy (MeV)")
    
    # Add Colorbar (Bottom)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.15, 0.06, 0.6, 0.02])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r"$d^2Q/dEd\theta_x$ (pC/MeV/mrad)", fontsize=14)
    
    # plt.tight_layout() # tight_layout can mess up manually added axes
    filename = f'energy_divergence_stacked_a{a0_target}.png'
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_dopant_emittance_histogram(a0_target, dopant_species):
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
    t_ps = ts.t[ts.iterations.tolist().index(iteration)] * 1e12

    # 1. Get the raw particle arrays
    # We use a loose uz filter as a pre-filter to reduce data volume
    x, y, z, ux, uy, uz, w = ts.get_particle(
        var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], 
        species='electrons_injected', 
        iteration=iteration, 
        select={'uz': [uz_threshold * 0.9, None]} 
    )

    if len(x) == 0:
        print("No particles found for emittance calculation.")
        return

    # Rigorous Energy Calculation and Filtering
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    E_MeV = (gamma - 1) * (m_e * c**2 / (e * 1e6))
    mask = E_MeV > E_threshold_MeV
    
    x, y, z, ux, uy, uz, w = x[mask], y[mask], z[mask], ux[mask], uy[mask], uz[mask], w[mask]
    
    if len(x) == 0:
        print(f"No particles found above {E_threshold_MeV} MeV.")
        return

    q = w * e * 1e12 

    # --- HELPER FOR EMITTANCE CALCULATION ---
    def calculate_emittance_axis(pos, mom, weights):
        # Weighted means
        avg_pos = np.average(pos, weights=weights)
        avg_mom = np.average(mom, weights=weights)

        # Centered variances and covariance
        var_pos = np.average((pos - avg_pos)**2, weights=weights)
        var_mom = np.average((mom - avg_mom)**2, weights=weights)
        cov_pos_mom = np.average((pos - avg_pos)*(mom - avg_mom), weights=weights)

        # RMS values
        rms_pos = np.sqrt(var_pos)
        rms_mom = np.sqrt(var_mom)
        
        # Emittance
        emit = np.sqrt(var_pos * var_mom - cov_pos_mom**2)

        # Errors
        N = len(weights)
        sigma_rms_pos = rms_pos / np.sqrt(N)
        sigma_rms_mom = rms_mom / np.sqrt(N)
        sigma_cov = np.abs(cov_pos_mom) / np.sqrt(N)

        term_1 = 4 * (rms_pos**2) * (rms_mom**4) * (sigma_rms_pos**2) + 4 * (rms_pos**4) * (rms_mom**2) * (sigma_rms_mom**2)
        term_2 = 4 * (cov_pos_mom**2) * (sigma_cov**2)
        sigma_emit_sq = term_1 + term_2
        sigma_emit = np.sqrt(sigma_emit_sq) / (2 * emit)
        
        return emit, sigma_emit, avg_pos, avg_mom, rms_pos, rms_mom

    # --- CALCULATE FOR X AND Y ---
    emit_x, sigma_emit_x, avg_x, avg_ux, rms_x, rms_ux = calculate_emittance_axis(x, ux, w)
    emit_y, sigma_emit_y, avg_y, avg_uy, rms_y, rms_uy = calculate_emittance_axis(y, uy, w)

    emit_x1, emity_y1 = ts.get_emittance(iteration, species='electrons_injected',select={'uz': [uz_threshold, None]},
                     kind='normalized', description='projected')
    

    # --- OUTPUT ---
    print(f"Emittance X ({dopant_species}): {emit_x*1e6:.2e} +/- {sigma_emit_x*1e6:.2e} mrad")
    print(f"Emittance Y ({dopant_species}): {emit_y*1e6:.2e} +/- {sigma_emit_y*1e6:.2e} mrad")

    # --- PLOTTING (3 Subplots) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Manual limits toggle
    use_manual_limits = True
    manual_xlim = [-3, 3]  # in µm
    manual_ylim = [-6, 6]  # in p/mec
    
    n_sigma_x = 3.5
    
    # Plot X Phase Space
    x_microns = x * 1e6
    mean_x_um = avg_x * 1e6
    rms_x_um = rms_x * 1e6
    
    if use_manual_limits:
        xlims_x = manual_xlim
        ylims_x = manual_ylim
    else:
        xlims_x = [mean_x_um - n_sigma_x*rms_x_um, mean_x_um + n_sigma_x*rms_x_um]
        ylims_x = [avg_ux - n_sigma_x*rms_ux, avg_ux + n_sigma_x*rms_ux]
    
    # Colorbar limits
    cbar_vmin = 0

    h1 = ax1.hist2d(x_microns, ux, bins=250, range=[xlims_x, ylims_x], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax1.set_xlabel(r'$x \; (\mu m)$')
    ax1.set_ylabel(r'$p_x / m_e c$')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax1.set_xlim(manual_xlim)
        ax1.set_ylim(manual_ylim)
    else:
        ax1.set_xlim(xlims_x)
        ax1.set_ylim(ylims_x)
    plt.colorbar(h1[3], ax=ax1, label='Charge per bin (pC)')

    n_sigma_y = 3.5

    # Plot Y Phase Space
    y_microns = y * 1e6
    mean_y_um = avg_y * 1e6
    rms_y_um = rms_y * 1e6
    
    if use_manual_limits:
        xlims_y = manual_xlim
        ylims_y = manual_ylim
    else:
        xlims_y = [mean_y_um - n_sigma_y*rms_y_um, mean_y_um + n_sigma_y*rms_y_um]
        ylims_y = [avg_uy - n_sigma_y*rms_uy, avg_uy + n_sigma_y*rms_uy]
    
    h2 = ax2.hist2d(y_microns, uy, bins=250, range=[xlims_y, ylims_y], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax2.set_xlabel(r'$y \; (\mu m)$')
    ax2.set_ylabel(r'$p_y / m_e c$')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax2.set_xlim(manual_xlim)
        ax2.set_ylim(manual_ylim)
    else:
        ax2.set_xlim(xlims_y)
        ax2.set_ylim(ylims_y)
    plt.colorbar(h2[3], ax=ax2, label='Charge per bin (pC)')

    # Plot Longitudinal Phase Space (Z vs pz)
    z_microns = z * 1e6
    avg_z = np.average(z, weights=w)
    avg_uz = np.average(uz, weights=w)
    rms_z = np.sqrt(np.average((z - avg_z)**2, weights=w))
    rms_uz = np.sqrt(np.average((uz - avg_uz)**2, weights=w))
    
    mean_z_um = avg_z * 1e6
    rms_z_um = rms_z * 1e6
    
    n_sigma_z = 5
    xlims_z = [mean_z_um - n_sigma_z*rms_z_um, mean_z_um + n_sigma_z*rms_z_um]
    # ylims_z = [avg_uz - n_sigma_z*rms_uz, avg_uz + n_sigma_z*rms_uz]

    # xlims_z = None
    ylims_z = None
    
    h3 = ax3.hist2d(z_microns, uz, bins=250, range=[xlims_z, ylims_z], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax3.set_xlabel(r'$z \; (\mu m)$')
    ax3.set_ylabel(r'$p_z / m_e c$')
    ax3.set_xlim(xlims_z)
    ax3.set_ylim(ylims_z)
    plt.colorbar(h3[3], ax=ax3, label='Charge per bin (pC)')

    fig.suptitle(f"{dopant_species}-Doped Phase Space at t = {t_ps:.2f} ps", fontsize=16)
    plt.tight_layout()
    filename = f'{dopant_species}_phase_space_histogram.png'
    plt.savefig(filename, dpi=800)
    print(f"Saved: {filename}")
    plt.close()

def plot_pure_he_emittance_histogram(a0_target):
    print(f"\nGenerating Plot : Pure Helium Electron Emittance (a0={a0_target})...")
    mode = 'pure_he'

    # Load Data
    ts = load_data(a0_target, None, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    t_ps = ts.t[ts.iterations.tolist().index(iteration)] * 1e12
    
    # Determine particle species name to fetch
    species_names = ['electrons_bulk', 'electrons_he']
    x, y, z, ux, uy, uz, w = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    for s_name in species_names:
        try:
            # 1. Get the raw particle arrays
            # We use a loose uz filter as a pre-filter to reduce data volume
            x, y, z, ux, uy, uz, w = ts.get_particle(
                var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], 
                species=s_name, 
                iteration=iteration, 
                select={'uz': [uz_threshold * 0.9, None]} 
            )
            if len(x) > 0:
                break
        except:
            continue

    if len(x) == 0:
        print("No particles found for emittance calculation.")
        return

    # Rigorous Energy Calculation and Filtering
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    E_MeV = (gamma - 1) * (m_e * c**2 / (e * 1e6))
    mask = E_MeV > E_threshold_MeV
    
    x, y, z, ux, uy, uz, w = x[mask], y[mask], z[mask], ux[mask], uy[mask], uz[mask], w[mask]
    
    if len(x) == 0:
        print(f"No particles found above {E_threshold_MeV} MeV.")
        return

    q = w * e * 1e12 

    # --- HELPER FOR EMITTANCE CALCULATION ---
    def calculate_emittance_axis(pos, mom, weights):
        # Weighted means
        avg_pos = np.average(pos, weights=weights)
        avg_mom = np.average(mom, weights=weights)

        # Centered variances and covariance
        var_pos = np.average((pos - avg_pos)**2, weights=weights)
        var_mom = np.average((mom - avg_mom)**2, weights=weights)
        cov_pos_mom = np.average((pos - avg_pos)*(mom - avg_mom), weights=weights)

        # RMS values
        rms_pos = np.sqrt(var_pos)
        rms_mom = np.sqrt(var_mom)
        
        # Emittance
        emit = np.sqrt(var_pos * var_mom - cov_pos_mom**2)

        # Errors
        N = len(weights)
        sigma_rms_pos = rms_pos / np.sqrt(N)
        sigma_rms_mom = rms_mom / np.sqrt(N)
        sigma_cov = np.abs(cov_pos_mom) / np.sqrt(N)

        term_1 = 4 * (rms_pos**2) * (rms_mom**4) * (sigma_rms_pos**2) + 4 * (rms_pos**4) * (rms_mom**2) * (sigma_rms_mom**2)
        term_2 = 4 * (cov_pos_mom**2) * (sigma_cov**2)
        sigma_emit_sq = term_1 + term_2
        sigma_emit = np.sqrt(sigma_emit_sq) / (2 * emit)
        
        return emit, sigma_emit, avg_pos, avg_mom, rms_pos, rms_mom

    # --- CALCULATE FOR X AND Y ---
    emit_x, sigma_emit_x, avg_x, avg_ux, rms_x, rms_ux = calculate_emittance_axis(x, ux, w)
    emit_y, sigma_emit_y, avg_y, avg_uy, rms_y, rms_uy = calculate_emittance_axis(y, uy, w)

    # --- OUTPUT ---
    print(f"Emittance X (Pure He): {emit_x*1e6:.2e} +/- {sigma_emit_x*1e6:.2e} mrad")
    print(f"Emittance Y (Pure He): {emit_y*1e6:.2e} +/- {sigma_emit_y*1e6:.2e} mrad")

    # --- PLOTTING (3 Subplots) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Manual limits toggle
    use_manual_limits = True
    manual_xlim = [-3, 3]  # in µm
    manual_ylim = [-6, 6]  # in p/mec
    
    n_sigma_x = 3.5
    
    # Plot X Phase Space
    x_microns = x * 1e6
    mean_x_um = avg_x * 1e6
    rms_x_um = rms_x * 1e6
    
    if use_manual_limits:
        xlims_x = manual_xlim
        ylims_x = manual_ylim
    else:
        xlims_x = [mean_x_um - n_sigma_x*rms_x_um, mean_x_um + n_sigma_x*rms_x_um]
        ylims_x = [avg_ux - n_sigma_x*rms_ux, avg_ux + n_sigma_x*rms_ux]
    
    # Colorbar limits
    cbar_vmin = 0

    h1 = ax1.hist2d(x_microns, ux, bins=250, range=[xlims_x, ylims_x], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax1.set_xlabel(r'$x \; (\mu m)$')
    ax1.set_ylabel(r'$p_x / m_e c$')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax1.set_xlim(manual_xlim)
        ax1.set_ylim(manual_ylim)
    else:
        ax1.set_xlim(xlims_x)
        ax1.set_ylim(ylims_x)
    plt.colorbar(h1[3], ax=ax1, label='Charge per bin (pC)')

    n_sigma_y = 3.5

    # Plot Y Phase Space
    y_microns = y * 1e6
    mean_y_um = avg_y * 1e6
    rms_y_um = rms_y * 1e6
    
    if use_manual_limits:
        xlims_y = manual_xlim
        ylims_y = manual_ylim
    else:
        xlims_y = [mean_y_um - n_sigma_y*rms_y_um, mean_y_um + n_sigma_y*rms_y_um]
        ylims_y = [avg_uy - n_sigma_y*rms_uy, avg_uy + n_sigma_y*rms_uy]
    
    h2 = ax2.hist2d(y_microns, uy, bins=250, range=[xlims_y, ylims_y], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax2.set_xlabel(r'$y \; (\mu m)$')
    ax2.set_ylabel(r'$p_y / m_e c$')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax2.set_xlim(manual_xlim)
        ax2.set_ylim(manual_ylim)
    else:
        ax2.set_xlim(xlims_y)
        ax2.set_ylim(ylims_y)
    plt.colorbar(h2[3], ax=ax2, label='Charge per bin (pC)')

    # Plot Longitudinal Phase Space (Z vs pz)
    z_microns = z * 1e6
    avg_z = np.average(z, weights=w)
    avg_uz = np.average(uz, weights=w)
    rms_z = np.sqrt(np.average((z - avg_z)**2, weights=w))
    rms_uz = np.sqrt(np.average((uz - avg_uz)**2, weights=w))
    
    mean_z_um = avg_z * 1e6
    rms_z_um = rms_z * 1e6
    
    n_sigma_z = 5
    xlims_z = [mean_z_um - n_sigma_z*rms_z_um, mean_z_um + n_sigma_z*rms_z_um]
    ylims_z = None
    
    h3 = ax3.hist2d(z_microns, uz, bins=250, range=[xlims_z, ylims_z], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax3.set_xlabel(r'$z \; (\mu m)$')
    ax3.set_ylabel(r'$p_z / m_e c$')
    ax3.set_xlim(xlims_z)
    ax3.set_ylim(ylims_z)
    plt.colorbar(h3[3], ax=ax3, label='Charge per bin (pC)')

    fig.suptitle(f"Pure Helium Phase Space at t = {t_ps:.2f} ps", fontsize=16)
    plt.tight_layout()
    filename = 'pure_he_phase_space_histogram.png'
    plt.savefig(filename, dpi=800)
    print(f"Saved: {filename}")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Run Charge Comparison
    # plot_dopant_comparison()
    
    # plot_e_injected(a0, 'Ar')
    
    # Analyse density distribution to help set vmin/vmax
    # analyse_injection_density_distribution(a0, 'Ar')
    
    # plot_e_density(a0, 'N')
    # plot_injection_dynamics(a0,'Ar')
    # plot_dopant_emittance_histogram(a0, 'N')
    # plot_pure_he_emittance_histogram(a0)
    # plot_energy_spectra_comparison(a0)
    # plot_energy_divergence_comparison(a0)
    
    # Plot Pure He Wakefield
    plot_e_density_pure_he(a0)

    # Run detailed plots for each dopant at fixed a0
    for species in dopant_list:
        # plot_dopant_emittance_histogram(a0_target=a0, dopant_species=species)
        # plot_phase_space(a0_target=a0, dopant_species=species)
        # plot_e_density(a0_target=a0, dopant_species=species)
        pass
    
