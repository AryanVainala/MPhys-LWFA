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
dopant_list = ['N', 'Ar']  # List of dopants to compare
base_dir = './diags_doped'

# Physical parameters
n_e_target = 3.5e24
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
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
    
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
    fig = plt.figure(figsize=(12, 6))
    
    # Create gridspec: main plot on top, two colorbars at the bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1],
                          hspace=0.25, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, :])  # Main plot spans both columns
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
                       cmap='binary',
                       interpolation='bilinear',
                       vmin = np.percentile(n_bulk, 0),
                       vmax = np.percentile(n_bulk, 99.95)
            )
    im_rho.cmap.set_over('black')

    # Horizontal colorbar for bulk density at bottom left
    cbar = plt.colorbar(im_rho, cax=cax_bulk, orientation='horizontal')
    cbar.set_label(r'Bulk Plasma: $n_e/n_0$', fontsize=10)

    # Layer 1.5: Injected Electrons with SEPARATE scaling
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
                           vmin = np.percentile(n_inj, 0),     # For Neon clip it np.percentile(n_inj, 98.99), argon 0.01
                           vmax = np.percentile(n_inj, 99.95)    #                  np.percentile(n_inj, 99.9) 
              )
        
        # Horizontal colorbar for injected electrons at bottom right
        cbar_inj = plt.colorbar(im_inj, cax=cax_inj, orientation='horizontal')
        cbar_inj.set_label(r'Injected Electrons: $n_e/n_0$', fontsize=10)
    else:
        cax_inj.axis('off')  # Hide if no injected electrons
    
    # Layer 2: Laser envelope
    
    cmp_laser = get_transparent_inferno(
    n_bins=512,
    desat_factor=0.2,
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

    # --- TWIN AXIS FOR Ez ---
    ax2 = ax.twinx()
    
    # Plot Ez on top
    ax2.plot(z_Ez, Ez_GV, color="#0606B6", linewidth=1.5, alpha=0.9, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Manually setting x-limits ensures both plots are locked to the same view
    ax.set_xlim(z_rho.min(), z_rho.max())
    
    # Labels
    ax.set_xlabel(r'$z \; (\mu m)$', fontsize=12)
    ax.set_ylabel(r'$r \; (\mu m)$', fontsize=12)
    ax.set_title(f"{dopant_species}-Doped He at (t = {t_fs:.1f} fs)")
    
    # Right Axis Styling
    ax2.set_ylabel(r'$E_z$ (GV/m)', color='#0606B6')
    ax2.tick_params(axis='y', labelcolor='#0606B6')
    ax2.set_ylim(-1000, 1000)

    # Don't use tight_layout with gridspec - it's already handled
    
    filename = f'{dopant_species}_wakefield_fixed_last.png'
    plt.savefig(filename, dpi=800, bbox_inches='tight')
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
    z = info_rho.z * 1e6  # microns
    r = info_rho.r * 1e6  # microns
    extent = [z.min(), z.max(), r.min(), r.max()]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot density
    im = ax.imshow(n_injected, extent=extent, origin='lower', aspect='auto', cmap='Greens')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$n_{injected} \; (m^{-3})$')

    ax.set_xlabel(r'$z \; (\mu m)$')
    ax.set_ylabel(r'$r \; (\mu m)$')
    ax.set_title(f"Injected Electron Density - {label} (t = {t_fs:.1f} fs)")

    plt.tight_layout()
    filename = f'{dopant_species}_injected_charge_density.png'
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_energy_spectra_comparison(a0_target):
    print(f"\nGenerating Plot : Energy Spectra Comparison (a0={a0_target})...")
    
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten() # Flatten to 1D array for easy indexing
    
    # Define plot order and parameters
    plot_configs = [
        {'title': 'Pure Helium (Reference)', 'species': None, 'mode': 'pure_he', 'color': 'grey'},
        {'title': 'Nitrogen (N) Doped',     'species': 'N',   'mode': 'doped',   'color': 'black'},
        {'title': 'Argon (Ar) Doped',        'species': 'Ar',  'mode': 'doped',   'color': 'black'}
    ]
    
    E_min_cutoff = 50.0 # MeV
    bin_width = 1.0 # MeV
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        species = config['species']
        mode = config['mode']
        
        ts = load_data(a0_target, species, mode)
        
        if ts is None:
            ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center', transform=ax.transAxes)
            continue
            
        iteration = ts.iterations[-1]
        
        # Determine particle species name to fetch
        if mode == 'pure_he':
            species_names = ['electrons_bulk', 'electrons_he']
        else:
            species_names = ['electrons_injected', 'electrons_dopant']
            
        uz, w = np.array([]), np.array([])
        
        for s_name in species_names:
            try:
                uz, w = ts.get_particle(['uz', 'w'], species=s_name, iteration=iteration)
                if len(uz) > 0:
                    break
            except:
                continue
                
        if len(uz) == 0:
            ax.text(0.5, 0.5, "No Particles Found", ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Calculate Energy
        E_MeV = (np.sqrt(uz**2 + 1) - 1) * 0.511
        
        # Filter
        mask = E_MeV > E_min_cutoff
        E_selected = E_MeV[mask]
        w_selected = w[mask]
        
        if len(E_selected) == 0:
            ax.text(0.5, 0.5, f"No Particles > {E_min_cutoff} MeV", ha='center', va='center', transform=ax.transAxes)
            continue

        # Histogram
        E_max = np.max(E_selected)
        if E_max <= E_min_cutoff:
             bins = np.array([E_min_cutoff, E_min_cutoff + bin_width])
        else:
             bins = np.arange(np.floor(np.min(E_selected)), np.ceil(E_max) + bin_width, bin_width)
        
        hist_weights, bin_edges = np.histogram(E_selected, bins=bins, weights=w_selected)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Convert to dQ/dE [pC/MeV]
        dQ_dE = (hist_weights * e * 1e12) / bin_width
        
        # Plot
        ax.plot(bin_centers, dQ_dE, color=config['color'], linewidth=2)
        ax.fill_between(bin_centers, dQ_dE, color=config['color'], alpha=0.1)
        
        # Statistics
        total_charge = np.sum(w_selected) * e * 1e12 # pC
        mean_energy = np.average(E_selected, weights=w_selected)
        
        stats_text = f"Q = {total_charge:.1f} pC\n<E> = {mean_energy:.1f} MeV"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Styling
        ax.set_title(config['title'])
        ax.set_xlabel(r"Energy (MeV)")
        ax.set_ylabel(r"$dQ/dE$ (pC/MeV)")
        ax.set_xlim(left=E_min_cutoff)
        ax.set_ylim(bottom=0)

    plt.suptitle(f"Injected Electron Energy Spectra (a0={a0_target})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f'energy_spectra_comparison_a{a0_target}.png'
    plt.savefig(filename, dpi=800)
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

    # 1. Get the raw particle arrays
    x, y, ux, uy, w = ts.get_particle(
        var_list=['x', 'y', 'ux', 'uy', 'w'], 
        species='electrons_injected', 
        iteration=iteration, 
        select={'uz': [100, None]}  # YOUR ENERGY/MOMENTA FILTER
    )

    if len(x) == 0:
        print("No particles found for emittance calculation.")
        return

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
    print(f"Emittance X ({dopant_species}): {emit_x*1e6:.2e} +/- {sigma_emit_x*1e6:.2e} mrad")
    print(f"Emittance Y ({dopant_species}): {emit_y*1e6:.2e} +/- {sigma_emit_y*1e6:.2e} mrad")

    # --- PLOTTING (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Manual limits toggle
    use_manual_limits = True
    manual_xlim = [-6, 6]  # in µm
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

    h1 = ax1.hist2d(x_microns, ux, bins=250, range=[xlims_x, ylims_x], weights=w*1e-4, cmap='turbo', vmin=cbar_vmin, vmax=25)
    ax1.set_xlabel(r'$x \; (\mu m)$')
    ax1.set_ylabel(r'$p_x / m_e c$')
    ax1.set_title(f'X Phase Space')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax1.set_xlim(manual_xlim)
        ax1.set_ylim(manual_ylim)
    else:
        ax1.set_xlim(xlims_x)
        ax1.set_ylim(ylims_x)
    plt.colorbar(h1[3], ax=ax1, label='Charge Density (arb.)')

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
    
    h2 = ax2.hist2d(y_microns, uy, bins=250, range=[xlims_y, ylims_y], weights=w*1e-4, cmap='turbo', vmin=cbar_vmin, vmax=75)
    ax2.set_xlabel(r'$y \; [\mu m]$')
    ax2.set_ylabel(r'$p_y / m_e c$')
    ax2.set_title(f'Y Phase Space')
    
    # Apply limits: manual if enabled, otherwise n_sigma based
    if use_manual_limits:
        ax2.set_xlim(manual_xlim)
        ax2.set_ylim(manual_ylim)
    else:
        ax2.set_xlim(xlims_y)
        ax2.set_ylim(ylims_y)
    plt.colorbar(h2[3], ax=ax2, label='Charge Density (arb.)')

    plt.suptitle(f"Transverse Phase Space ({dopant_species}-doped)", fontsize=14)
    plt.tight_layout()
    filename = f'{dopant_species}_transverse_phase_space.png'
    plt.savefig(filename, dpi=800)
    print(f"Saved: {filename}")
    plt.close()

def analyse_injection_density_distribution(a0_target, dopant_species):
    """
    Analyses the distribution of injected and bulk electron density values.
    Generates a histogram of density values.
    """
    print(f"\nAnalysing Injection and Bulk Density Distribution (a0={a0_target}, dopant={dopant_species})...")

    mode = 'doped'
    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        print("Data Not Available")
        return

    # Use last iteration
    iteration_idx = -1
    iteration = ts.iterations[iteration_idx]
    
    # --- GET PLASMA DENSITY (BULK) ---
    try:
        rho_bulk, _ = ts.get_field(field='rho_electrons_bulk', iteration=iteration)
    except:
        rho_bulk, _ = ts.get_field(field='rho', iteration=iteration)
    
    n_bulk_raw = -rho_bulk / e
    z_indices = n_bulk_raw.shape[1]
    sample_region = int(z_indices * 0.9) 
    n0 = np.median(n_bulk_raw[:, sample_region:])
    
    n_bulk = n_bulk_raw / n0
    
    # --- GET INJECTED ELECTRON DENSITY ---
    try:
        rho_inj, _ = ts.get_field(field='rho_electrons_injected', iteration=iteration)
    except Exception as err:
        print(f"Could not load injected electrons: {err}")
        rho_inj = None

    # Normalize Injected
    if rho_inj is not None:
        n_inj = -rho_inj / e / n0
    else:
        n_inj = np.array([])
    
    # Flatten and filter
    flat_bulk = n_bulk.flatten()
    flat_bulk = flat_bulk[flat_bulk > 1e-5]
    
    flat_inj = n_inj.flatten()
    flat_inj = flat_inj[flat_inj > 1e-5]

    # Statistics
    print(f"--- Bulk Density Statistics ---")
    print(f"Mean: {np.mean(flat_bulk):.2e}, Median: {np.median(flat_bulk):.2e}")
    print(f"99th Percentile: {np.percentile(flat_bulk, 99):.2e}")

    if len(flat_inj) > 0:
        print(f"--- Injected Density Statistics ---")
        print(f"Min: {np.min(flat_inj):.2e}")
        print(f"Max: {np.max(flat_inj):.2e}")
        print(f"Mean: {np.mean(flat_inj):.2e}")
        print(f"Median: {np.median(flat_inj):.2e}")
        print(f"90th Percentile: {np.percentile(flat_inj, 90):.2e}")
        print(f"99th Percentile: {np.percentile(flat_inj, 99):.2e}")
        print(f"99.9th Percentile: {np.percentile(flat_inj, 99.9):.2e}")

    # Plot Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Log-Log Histogram
    # Bulk
    ax.hist(flat_bulk, bins=100, log=True, color='grey', alpha=0.5, label='Bulk Electrons')
    
    # Injected
    if len(flat_inj) > 0:
        ax.hist(flat_inj, bins=100, log=True, color='blue', alpha=0.7, label='Injected Electrons')
        
        # Add percentile lines for injected
        p99 = np.percentile(flat_inj, 99)
        ax.axvline(p99, color='red', linestyle='--', label=f'Inj 99th %: {p99:.2e}')

    ax.set_xlabel(r'Density $n/n_0$')
    ax.set_ylabel('Frequency (Number of Grid Points)')
    ax.set_title(f'Distribution of Density ({dopant_species})')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    # Add top axis for percentiles (based on injected if available, else bulk)
    if len(flat_inj) > 0:
        target_flat = flat_inj
        label_prefix = "Inj"
    else:
        target_flat = flat_bulk
        label_prefix = "Bulk"

    ax2 = ax.twiny()
    percentile_levels = [0, 50, 90, 99, 100]
    tick_vals = np.percentile(target_flat, percentile_levels)
    
    ax2.set_xlim(ax.get_xlim())
    # Ensure ticks are within plot limits
    valid_ticks = []
    valid_labels = []
    current_xlim = ax.get_xlim()
    for p, val in zip(percentile_levels, tick_vals):
        if current_xlim[0] <= val <= current_xlim[1]:
            valid_ticks.append(val)
            valid_labels.append(f"{label_prefix} {p}%")
            
    ax2.set_xticks(valid_ticks)
    ax2.set_xticklabels(valid_labels, rotation=45, fontsize=8)
    ax2.set_xlabel(f"{label_prefix} Percentile")

    filename = f'{dopant_species}_density_distribution.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved distribution plot: {filename}")
    plt.close()


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Run Charge Comparison
    # plot_dopant_comparison()
    
    # plot_phase_space(a0, dopant_species='Ar')
    # plot_e_injected(a0, 'Ne')
    
    # Analyse density distribution to help set vmin/vmax
    # analyse_injection_density_distribution(a0, 'Ar')
    
    plot_e_density(a0, 'N')
    # plot_laser_envelope()
    # plot_dopant_emittance(a0, 'Ar')
    

    # plot_energy_spectra_comparison(a0)

    # Run detailed plots for each dopant at fixed a0
    for species in dopant_list:
        # plot_dopant_emittance(a0_target=a0, dopant_species=species)
        # plot_phase_space(a0_target=a0, dopant_species=species)
        # plot_e_density(a0_target=a0, dopant_species=species)
        pass
    
