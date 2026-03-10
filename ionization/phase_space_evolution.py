import os
os.environ["OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0
import argparse

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'sans-serif'],
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

base_dir = './Archived/diags_doped/'

# Energy threshold for injected electrons
E_threshold_MeV = 50
gamma_threshold = 1 + (E_threshold_MeV * 1e6 * e) / (m_e * c**2)
uz_threshold = np.sqrt(gamma_threshold**2 - 1)

def load_data(a0, dopant_species, mode):
    if mode == 'doped':
        path = f"{base_dir}/a{a0}_{mode}_{dopant_species}/hdf5"
    else:
        path = f"{base_dir}/a{a0}_{mode}/hdf5"
    if not os.path.exists(path):
        print(f"Warning: Data not found for a0={a0}, mode={mode}, dopant={dopant_species} at {path}")
        return None
    return LpaDiagnostics(path)

def update_phase_space_plot(iteration, ts, fig, dopant_species, mode):
    fig.clear()
    t_ps = ts.t[ts.iterations.tolist().index(iteration)] * 1e12
    print(f"Rendering iteration {iteration} (t = {t_ps:.2f} ps)...")

    # Determine particle species name to fetch
    if mode == 'pure_he':
        species_names = ['electrons_bulk', 'electrons_he']
    else:
        species_names = ['electrons_injected']

    x, y, z, ux, uy, uz, w = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    for s_name in species_names:
        try:
            # We use a loose uz filter as a pre-filter to reduce data volume
            x_s, y_s, z_s, ux_s, uy_s, uz_s, w_s = ts.get_particle(
                var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], 
                species=s_name, 
                iteration=iteration, 
                select={'uz': [uz_threshold * 0.9, None]} 
            )
            x = np.concatenate((x, x_s))
            y = np.concatenate((y, y_s))
            z = np.concatenate((z, z_s))
            ux = np.concatenate((ux, ux_s))
            uy = np.concatenate((uy, uy_s))
            uz = np.concatenate((uz, uz_s))
            w = np.concatenate((w, w_s))
        except Exception as e_err:
            print(f"Could not load species {s_name} at iteration {iteration}: {e_err}")

    if len(x) == 0:
        print(f"No particles found for emittance calculation at iteration {iteration}.")
        return

    # Rigorous Energy Calculation and Filtering
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    E_MeV = (gamma - 1) * (m_e * c**2 / (e * 1e6))
    mask = E_MeV > E_threshold_MeV
    
    x, y, z, ux, uy, uz, w = x[mask], y[mask], z[mask], ux[mask], uy[mask], uz[mask], w[mask]
    
    if len(x) == 0:
        print(f"No particles found above {E_threshold_MeV} MeV at iteration {iteration}.")
        return

    q = w * e * 1e12 

    # --- PLOTTING (3 Subplots) ---
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Manual limits toggle
    use_manual_limits = True
    manual_xlim = [-3, 3]  # in µm
    manual_ylim = [-10, 10]  # in p/mec
    
    # Plot X Phase Space
    x_microns = x * 1e6
    avg_x = np.average(x, weights=w)
    rms_x = np.sqrt(np.average((x - avg_x)**2, weights=w))
    avg_ux = np.average(ux, weights=w)
    rms_ux = np.sqrt(np.average((ux - avg_ux)**2, weights=w))
    
    mean_x_um = avg_x * 1e6
    rms_x_um = rms_x * 1e6
    n_sigma_x = 3.5
    
    if use_manual_limits:
        xlims_x = manual_xlim
        ylims_x = manual_ylim
    else:
        xlims_x = [mean_x_um - n_sigma_x*rms_x_um, mean_x_um + n_sigma_x*rms_x_um]
        ylims_x = [avg_ux - n_sigma_x*rms_ux, avg_ux + n_sigma_x*rms_ux]
    
    cbar_vmin = 0

    h1 = ax1.hist2d(x_microns, ux, bins=250, range=[xlims_x, ylims_x], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax1.set_xlabel(r'$x \; (\mu m)$')
    ax1.set_ylabel(r'$p_x / m_e c$')
    
    if use_manual_limits:
        ax1.set_xlim(manual_xlim)
        ax1.set_ylim(manual_ylim)
    else:
        ax1.set_xlim(xlims_x)
        ax1.set_ylim(ylims_x)
    fig.colorbar(h1[3], ax=ax1, label='Charge per bin (pC)')

    # Plot Y Phase Space
    y_microns = y * 1e6
    avg_y = np.average(y, weights=w)
    rms_y = np.sqrt(np.average((y - avg_y)**2, weights=w))
    avg_uy = np.average(uy, weights=w)
    rms_uy = np.sqrt(np.average((uy - avg_uy)**2, weights=w))
    
    mean_y_um = avg_y * 1e6
    rms_y_um = rms_y * 1e6
    n_sigma_y = 3.5
    
    if use_manual_limits:
        xlims_y = manual_xlim
        ylims_y = manual_ylim
    else:
        xlims_y = [mean_y_um - n_sigma_y*rms_y_um, mean_y_um + n_sigma_y*rms_y_um]
        ylims_y = [avg_uy - n_sigma_y*rms_uy, avg_uy + n_sigma_y*rms_uy]
    
    h2 = ax2.hist2d(y_microns, uy, bins=250, range=[xlims_y, ylims_y], weights=q, cmap='turbo', vmin=cbar_vmin)
    ax2.set_xlabel(r'$y \; (\mu m)$')
    ax2.set_ylabel(r'$p_y / m_e c$')
    
    if use_manual_limits:
        ax2.set_xlim(manual_xlim)
        ax2.set_ylim(manual_ylim)
    else:
        ax2.set_xlim(xlims_y)
        ax2.set_ylim(ylims_y)
    fig.colorbar(h2[3], ax=ax2, label='Charge per bin (pC)')

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
    if ylims_z is not None:
        ax3.set_ylim(ylims_z)
    fig.colorbar(h3[3], ax=ax3, label='Charge per bin (pC)')

    title_prefix = f"{dopant_species}-Doped" if mode == 'doped' else "Pure Helium"
    fig.suptitle(f"{title_prefix} Phase Space at t = {t_ps:.2f} ps (Iteration {iteration})", fontsize=16)
    fig.tight_layout()

def animate_phase_space(a0_target, dopant_species, mode, step=10, fps=10):
    print(f"\nInitializing Phase Space Animation for a0={a0_target}, dopant={dopant_species}, mode={mode}, step={step}...")
    
    ts = load_data(a0_target, dopant_species, mode)
    if ts is None:
        print("Data Not Available")
        return

    iterations = ts.iterations[::step]
    
    fig = plt.figure(figsize=(18, 5))
    
    def frame_generator(iteration):
        update_phase_space_plot(iteration, ts, fig, dopant_species, mode)
        
    anim = animation.FuncAnimation(fig, frame_generator, frames=iterations, interval=1000/fps, repeat=False)
    
    output_filename = f"phase_space_animation_{mode}_{dopant_species if dopant_species else 'pure_he'}.mp4"
    print(f"Saving animation to {output_filename}...")
    
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_filename, writer=writer, dpi=200)
        print(f"Animation saved successfully: {output_filename}")
    except Exception as exception:
        print(f"Error saving animation (FFmpeg might be missing or failed): {exception}")
        try:
            output_filename = output_filename.replace('.mp4', '.gif')
            print(f"Attempting to save as GIF: {output_filename}")
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_filename, writer=writer, dpi=100)
            print(f"GIF saved successfully: {output_filename}")
        except Exception as exception2:
            print(f"Failed to save animation: {exception2}")
            
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate phase space evolution.")
    parser.add_argument("--step", type=int, default=1, help="Plot every nth iteration (default: 1)")
    parser.add_argument("--dopant", type=str, default="Ar", help="Dopant species (e.g., Ar, N) or 'pure_he'")
    parser.add_argument("--mode", type=str, default="doped", help="Mode: 'doped' or 'pure_he'")
    parser.add_argument("--a0", type=float, default=2.5, help="a0 value (default: 2.5)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the animation (default: 10)")
    
    args = parser.parse_args()
    
    animate_phase_space(args.a0, args.dopant, args.mode, step=args.step, fps=args.fps)
