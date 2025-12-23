"""
Advanced Dopant Analysis Script
Implements: Beam Tables, Energy-Sliced Phase Space, and Beam Loading Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e

# ===========================
# CONFIGURATION
# ===========================
base_dir = './diags_doped'
a0 = 2.5
# Define the cases to compare
cases = [
    {'label': 'Pure He',  'mode': 'pure_he', 'species': None, 'color': 'gray'},
    {'label': 'N-doped',  'mode': 'doped',   'species': 'N',    'color': 'tab:red'},
    {'label': 'Ar-doped', 'mode': 'doped',   'species': 'Ar',   'color': 'tab:blue'}
]
# Threshold to define "Trapped Beam" (removes thermal background)
E_min_MeV = 50.0

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_beam_data(ts, iteration, mode):
    """Extracts beam arrays (x, ux, y, uy, z, w) for the injected species."""
    # Heuristic to find the right species
    species_list = list(ts.avail_species)
    target = None
    
    if mode == 'doped':
        if 'electrons_injected' in species_list: target = 'electrons_injected'
        elif 'electrons_dopant' in species_list: target = 'electrons_dopant'
    else:
        # For Pure He, we look for high energy bulk electrons
        if 'electrons_bulk' in species_list: target = 'electrons_bulk'
        elif 'electrons_he' in species_list: target = 'electrons_he'
            
    if target is None: return None

    x, y, z, ux, uy, uz, w = ts.get_particle(
        ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], 
        species=target, iteration=iteration
    )
    
    # Filter by Energy
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    E_MeV = (gamma - 1) * 0.511
    mask = E_MeV > E_min_MeV
    
    return {
        'x': x[mask], 'y': y[mask], 'z': z[mask],
        'ux': ux[mask], 'uy': uy[mask], 'uz': uz[mask],
        'w': w[mask], 'E_MeV': E_MeV[mask]
    }

def get_emittance(u, u_prime, w):
    """Calculates normalized emittance [mm mrad] from arrays."""
    # u = position (m), u_prime = normalized momentum (dimensionless)
    avg_u = np.average(u, weights=w)
    avg_up = np.average(u_prime, weights=w)
    
    du = u - avg_u
    dup = u_prime - avg_up
    
    u2 = np.average(du**2, weights=w)
    up2 = np.average(dup**2, weights=w)
    uup = np.average(du*dup, weights=w)
    
    # epsilon_n = sqrt(<x^2><p^2> - <xp>^2) / mc 
    # Since u_prime is p/mc, the result is in meters
    emit_m = np.sqrt(u2 * up2 - uup**2)
    return emit_m * 1e6 # Convert to mm mrad (or microns)

# ===========================
# MAIN ANALYSIS
# ===========================
results = {}

print(f"{'Case':<12} | {'Q (pC)':<8} | {'E_mean':<8} | {'dE/E %':<8} | {'en_x':<6} | {'en_y':<6} | {'Div_x':<6}")
print("-" * 85)

for case in cases:
    # 1. Load Data
    path = f"{base_dir}/a{a0}_{case['mode']}"
    if case['species']: path += f"_{case['species']}"
    path += "/hdf5"
    
    try:
        ts = OpenPMDTimeSeries(path)
        it = ts.iterations[-1]
        beam = get_beam_data(ts, it, case['mode'])
    except Exception as e:
        print(f"Skipping {case['label']}: {e}")
        continue
        
    if beam is None or len(beam['x']) < 10:
        continue

    # 2. Compute Table Metrics
    q_pC = np.sum(beam['w']) * e * 1e12
    E_mean = np.average(beam['E_MeV'], weights=beam['w'])
    E_std = np.sqrt(np.average((beam['E_MeV'] - E_mean)**2, weights=beam['w']))
    spread_percent = (E_std / E_mean) * 100
    
    en_x = get_emittance(beam['x'], beam['ux'], beam['w'])
    en_y = get_emittance(beam['y'], beam['uy'], beam['w'])
    
    # Divergence (mrad) ~ px/pz
    theta_x = np.arctan2(beam['ux'], beam['uz'])
    div_x = np.sqrt(np.average((theta_x - np.average(theta_x, weights=beam['w']))**2, weights=beam['w'])) * 1e3
    
    print(f"{case['label']:<12} | {q_pC:<8.1f} | {E_mean:<8.1f} | {spread_percent:<8.1f} | {en_x:<6.2f} | {en_y:<6.2f} | {div_x:<6.1f}")
    
    results[case['label']] = {'beam': beam, 'ts': ts, 'it': it, 'color': case['color']}

# ===========================
# PLOT 1: SLICE PHASE SPACE
# ===========================
# Plotting Top 20% Energy vs Bottom 20% Energy for N and Ar
fig1, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
fig1.suptitle(f"Energy-Sliced Transverse Phase Space (E > {E_min_MeV} MeV)")

for i, label in enumerate(['N-doped', 'Ar-doped']):
    if label not in results: continue
    beam = results[label]['beam']
    
    # Sort by Energy
    idxs = np.argsort(beam['E_MeV'])
    x_sort = beam['x'][idxs] * 1e6 # microns
    ux_sort = beam['ux'][idxs]
    w_sort = beam['w'][idxs]
    
    # Cumulative weight for slicing
    cum_w = np.cumsum(w_sort)
    total_w = cum_w[-1]
    
    # Slices
    mask_low = cum_w < 0.2 * total_w
    mask_high = cum_w > 0.8 * total_w
    
    # Plot Bottom 20%
    axes[i, 0].hist2d(x_sort[mask_low], ux_sort[mask_low], bins=100, cmap='Blues')
    axes[i, 0].set_title(f"{label}: Low Energy Tail (Bottom 20%)")
    
    # Plot Top 20%
    axes[i, 1].hist2d(x_sort[mask_high], ux_sort[mask_high], bins=100, cmap='Oranges')
    axes[i, 1].set_title(f"{label}: High Energy Head (Top 20%)")
    
    # Formatting
    axes[i, 0].set_ylabel(r"$p_x / m_e c$")
    if i == 1:
        axes[i, 0].set_xlabel(r"$x (\mu m)$")
        axes[i, 1].set_xlabel(r"$x (\mu m)$")

plt.tight_layout()
plt.savefig("sliced_phase_space.png")
print("\nSaved: sliced_phase_space.png")

# ===========================
# PLOT 2: CURRENT & BEAM LOADING
# ===========================
fig2, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

for label in ['Pure He', 'N-doped', 'Ar-doped']:
    if label not in results: continue
    res = results[label]
    beam = res['beam']
    
    # 1. Current Profile I(z)
    # Histogram particles in z
    z_um = beam['z'] * 1e6
    hist, bin_edges = np.histogram(z_um, bins=100, weights=beam['w'])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dz = (bin_edges[1] - bin_edges[0]) * 1e-6 # meters
    current_kA = (hist * e * c / dz) / 1e3
    
    # Plot Current (only for doped to reduce clutter, or all)
    if 'doped' in label:
        ax1.plot(bin_centers, current_kA, color=res['color'], alpha=0.6, lw=2, label=f"$I(z)$ {label}")
        ax1.fill_between(bin_centers, current_kA, color=res['color'], alpha=0.1)

    # 2. Ez Lineout
    # Get field on axis
    Ez, info = res['ts'].get_field(field='E', coord='z', iteration=res['it'], m=0, slice_across='r')
    ax2.plot(info.z*1e6, Ez/1e9, color=res['color'], linestyle='--' if 'He' in label else '-', lw=1.5, label=f"$E_z$ {label}")

ax1.set_xlabel(r"$z (\mu m)$")
ax1.set_ylabel("Current (kA)")
ax2.set_ylabel(r"$E_z$ (GV/m)")
ax1.set_xlim(np.min(z_um)-10, np.max(z_um)+10) # Zoom on bunch
ax2.set_ylim(-600, 400)
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
plt.title("Beam Loading Evidence: Current Profile vs Wakefield")
plt.tight_layout()
plt.savefig("beam_loading.png")
print("Saved: beam_loading.png")