"""
Dopant Animation Script

This script generates animations of the wakefield structure by iterating through
simulation frames and rendering them into a video file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0, pi
import os

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
base_dir = './Archived/diags_n2.5e+23'

# ==========================================
# DATA LOADING
# ==========================================

def load_data(a0, dopant_species, mode):
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
    r = np.linspace(0.0, 1.0, n_bins)
    L = 1.0 / (1.0 + np.exp(-k * (r - x0)))
    L0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    L1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
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
# ANIMATION LOGIC
# ==========================================

def update_plot_doped(iteration, ts, fig, dopant_species):
    """
    Renders a single frame for the doped simulation.
    Logic adapted from plot_e_density.
    """
    fig.clear()
    
    t_ps = ts.t[ts.iterations.tolist().index(iteration)] * 1e12
    
    # --- 1. GET PLASMA DENSITY ---
    try:
        rho, info_rho = ts.get_field(field='rho_electrons_bulk', iteration=iteration)
    except:
        rho, info_rho = ts.get_field(field='rho', iteration=iteration)
    
    z_rho = info_rho.z * 1e6
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

    # --- 4. GET INJECTED ELECTRON DENSITY ---
    n_inj = None
    extent_inj = None
    try:
        rho_inj, info_inj = ts.get_field(field='rho_electrons_injected', iteration=iteration)
        z_inj = info_inj.z * 1e6
        r_inj = info_inj.r * 1e6
        extent_inj = [z_inj.min(), z_inj.max(), r_inj.min(), r_inj.max()]
        n_inj = -rho_inj / e
    except Exception:
        pass

    # --- PLOTTING ---
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1],
                          hspace=0.4, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, :])
    ax.set_axisbelow(True)
    cax_bulk = fig.add_subplot(gs[1, 0])
    cax_inj = fig.add_subplot(gs[1, 1])

    n_bulk = -rho / e
    z_indices = n_bulk.shape[1]
    sample_region = int(z_indices * 0.9)
    n0 = np.median(n_bulk[:, sample_region:])
    if n0 == 0: n0 = 1.0 # Avoid division by zero
    n_bulk = n_bulk / n0
    if n_inj is not None:
        n_inj = n_inj / n0

    # Layer 1: Plasma Density
    im_rho = ax.imshow(n_bulk,
                       extent=extent_rho,
                       origin='lower',
                       aspect='auto',
                       cmap=cmc.grayC_r,
                       interpolation='bilinear',
                       vmin=np.percentile(n_bulk, 0.01),
                       vmax=np.percentile(n_bulk, 99.9))
    im_rho.cmap.set_over('black')
    cbar = plt.colorbar(im_rho, cax=cax_bulk, orientation='horizontal')
    cbar.set_label(r'Bulk Plasma: $n_e/n_0$', fontsize=14)

    # Layer 2: Laser
    cmp_laser = get_transparent_inferno(n_bins=512, desat_factor=0.1, x0=0.2, k=12.0, alpha_min=0.0, alpha_max=0.85)
    im_laser = ax.imshow(laser, cmap=cmp_laser, interpolation='bicubic', extent=extent_laser, origin='lower', aspect='auto', vmin=0)

    # Layer 3: Injected
    if n_inj is not None:
        n_bins_inj = 5000
        colors_inj = plt.cm.BuGn(np.linspace(0.9, 1, n_bins_inj))
        alpha_curve = np.linspace(0, 1, n_bins_inj) ** 0.5
        colors_inj[:, 3] = alpha_curve
        cmap_inj = LinearSegmentedColormap.from_list('blues_alpha', colors_inj)

        im_inj = ax.imshow(n_inj,
                           extent=extent_inj,
                           origin='lower',
                           aspect='auto',
                           cmap=cmap_inj,
                           interpolation='bilinear',
                           vmin=np.percentile(n_inj, 98.99),
                           vmax=np.percentile(n_inj, 99.99))
        
        cbar_inj = plt.colorbar(im_inj, cax=cax_inj, orientation='horizontal')
        cbar_inj.set_label(r'Injected Electrons: $n_e/n_0$', fontsize=14)
    else:
        cax_inj.axis('off')

    # Twin Axis
    ax2 = ax.twinx()
    ax2.plot(z_Ez, Ez_GV, color="#0606B6", linewidth=1.5, alpha=0.9, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(z_rho.min(), z_rho.max())
    
    ax.set_xlabel(r'$z \; (\mu m)$')
    ax.set_ylabel(r'$r \; (\mu m)$')
    ax.set_title(f"{dopant_species}-Doped He Wakefield at (t = {t_ps:.2f} ps)")
    
    ax2.set_ylabel(r'$E_z$ (GV/m)', color='#0606B6')
    ax2.tick_params(axis='y', labelcolor='#0606B6')
    ax2.set_ylim(-1000, 1000)


def update_plot_pure_he(iteration, ts, fig):
    """
    Renders a single frame for the pure helium simulation.
    Logic adapted from plot_e_density_pure_he.
    """
    fig.clear()
    
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e12

    # --- 1. GET BULK PLASMA DENSITY ---
    rho, info_rho = ts.get_field(field='rho_electrons_bulk', iteration=iteration)
    z_rho = info_rho.z * 1e6
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
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05], hspace=0.25)
    
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axisbelow(True)
    cax_bulk = fig.add_subplot(gs[1, 0])

    n_bulk = -rho / e
    z_indices = n_bulk.shape[1]
    sample_region = int(z_indices * 0.9)
    n0 = np.median(n_bulk[:, sample_region:])
    if n0 == 0: n0 = 1.0
    n_bulk = n_bulk / n0

    # Layer 1
    im_rho = ax.imshow(n_bulk,
                       extent=extent_rho,
                       origin='lower',
                       aspect='auto',
                       cmap=cmc.grayC_r,
                       interpolation='bilinear',
                       vmin=np.percentile(n_bulk, 0.01),
                       vmax=np.percentile(n_bulk, 99.9))
    im_rho.cmap.set_over('black')
    cbar = plt.colorbar(im_rho, cax=cax_bulk, orientation='horizontal')
    cbar.set_label(r'Bulk Plasma: $n_e/n_0$', fontsize=14)

    # Layer 2
    cmp_laser = get_transparent_inferno(n_bins=512, desat_factor=0.1, x0=0.2, k=12.0, alpha_min=0.0, alpha_max=0.85)
    im_laser = ax.imshow(laser, cmap=cmp_laser, interpolation='bicubic', extent=extent_laser, origin='lower', aspect='auto', vmin=0)

    # Twin Axis
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

def animate_simulation(a0_target, dopant_species, fps=10):
    """
    Creates an animation of the simulation steps.
    Calls update_plot_doped or update_plot_pure_he for each frame.
    """
    print(f"\nInitializing Animation for a0={a0_target}, dopant={dopant_species}...")
    
    if dopant_species is None or dopant_species.lower() == 'none' or dopant_species == '':
        mode = 'pure_he'
        label = 'Pure He'
    else:
        mode = 'doped'
        label = f"{dopant_species}-doped"
    
    ts = load_data(a0_target, dopant_species if mode == 'doped' else None, mode)
    
    if ts is None:
        print("Data source not found. Exiting animation.")
        return

    # Only one figure creation
    fig = plt.figure(figsize=(12, 12) if mode == 'doped' else (12, 10))
    iterations = ts.iterations
    
    def frame_generator(iteration):
        print(f"Rendering iteration {iteration}...")
        if mode == 'doped':
            update_plot_doped(iteration, ts, fig, dopant_species)
        else:
            update_plot_pure_he(iteration, ts, fig)
    
    # Create the animation using FuncAnimation
    anim = animation.FuncAnimation(fig, frame_generator, frames=iterations, interval=1000/fps)
    
    output_filename = f"animation_{mode}_{dopant_species if dopant_species else 'pure_he'}.mp4"
    print(f"Solving animation to {output_filename}...")
    
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_filename, writer=writer, dpi=200)
        print(f"Animation saved successfully: {output_filename}")
    except Exception as e:
        print(f"Error saving animation (FFmpeg might be missing or failed): {e}")
        try:
            output_filename = output_filename.replace('.mp4', '.gif')
            print(f"Attempting to save as GIF: {output_filename}")
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_filename, writer=writer, dpi=100)
            print(f"GIF saved successfully: {output_filename}")
        except Exception as e2:
            print(f"Failed to save animation: {e2}")
    
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Wakefield Animation")
    parser.add_argument("--a0", type=float, default=2.5, help="Laser a0 parameter")
    parser.add_argument("--dopant", type=str, default="N", help="Dopant species (N, Ne, Ar) or None for pure He")
    
    args = parser.parse_args()
    
    dopant_val = args.dopant
    if dopant_val.lower() == "none" or dopant_val.lower() == "pure_he":
        dopant_val = None
        
    animate_simulation(args.a0, dopant_val)
