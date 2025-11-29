"""
Comprehensive Analysis Script for LWFA Ionization Study

This script analyzes simulation data from three gas species (H, He, N) and generates
three publication-quality comparison plots:

1. Electron Energy Spectra - Final energy distributions overlaid
2. Laser Amplitude Evolution - On-axis a₀ vs propagation distance
3. Wakefield Structure - 2D charge density with on-axis Ez overlay
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ==========================================
# CONFIGURATION
# ==========================================

# Data directories for each gas
data_dirs = {
    'H': './diags_H/hdf5',
    'He': './diags_He/hdf5',
    'N': './diags_N/hdf5'
}

# Display parameters
gas_labels = {
    'H': 'Hydrogen (H)',
    'He': 'Helium (He)',
    'N': 'Nitrogen (N)'
}

gas_colors = {
    'H': '#FF6B6B',   # Red
    'He': '#4ECDC4',  # Cyan
    'N': "#95E1AC"    # Light green
}

# Physical parameters (from simulation)
lambda0 = 0.8e-6      # Laser wavelength (m)
n_e = 7.0e24          # Target electron density (m^-3)
a0 = 4.0              # Laser amplitude

# Calculate plasma parameters
omega_p = np.sqrt(n_e * e**2 / (m_e * epsilon_0))
lambda_p = 2 * pi * c / omega_p
k_p = omega_p / c

# Load theoretical values
print("Loading theoretical parameters...")
try:
    with open('theoretical_parameters.txt', 'r') as f:
        theory_text = f.read()
        # Extract Ld and Lpd values
        # Format: "  Dephasing length (Ld):               0.6664 mm  (666.44 µm)"
        for line in theory_text.split('\n'):
            if 'Dephasing length' in line and 'µm)' in line:
                # Extract value in µm from second parentheses: (666.44 µm)
                parts = line.split('(')
                if len(parts) >= 3:  # Has at least 2 opening parentheses
                    L_d = float(parts[2].split('µm')[0].strip()) * 1e-6  # Convert to meters
            if 'Pump depletion length' in line and 'µm)' in line:
                # Extract value in µm from second parentheses
                parts = line.split('(')
                if len(parts) >= 3:
                    L_pd = float(parts[2].split('µm')[0].strip()) * 1e-6
    print(f"  Dephasing length (Ld): {L_d*1e6:.2f} µm")
    print(f"  Pump depletion length (Lpd): {L_pd*1e6:.2f} µm")
except (FileNotFoundError, ValueError, UnboundLocalError, IndexError) as e:
    print(f"Warning: Could not load theoretical_parameters.txt ({e})")
    print("Using approximate values...")
    L_d = (2/3) * (2*pi*c/lambda0/omega_p)**2 * np.sqrt(a0) * (c/omega_p)
    L_pd = (2*pi*c/lambda0/omega_p)**2 * a0 * (c/omega_p)

print()

# ==========================================
# VERIFY DATA DIRECTORIES
# ==========================================

print("Verifying data directories...")
for gas, path in data_dirs.items():
    if not os.path.exists(path):
        print(f"ERROR: Directory not found: {path}")
        print(f"Please run simulation for {gas} and rename output to diags_{gas}")
        sys.exit(1)
    else:
        n_files = len([f for f in os.listdir(path) if f.endswith('.h5')])
        print(f"  {gas}: Found {n_files} HDF5 files in {path}")

print("\nAll data directories verified!\n")

# ==========================================
# LOAD TIME SERIES DATA
# ==========================================

print("Loading OpenPMD time series...")
ts_data = {}
lpa_data = {}
for gas, path in data_dirs.items():
    print(f"  Loading {gas_labels[gas]}...")
    ts_data[gas] = OpenPMDTimeSeries(path)
    lpa_data[gas] = LpaDiagnostics(path)

print("\nData loaded successfully!\n")

# ==========================================
# PLOT 1: ELECTRON ENERGY SPECTRA
# ==========================================

print("="*70)
print("GENERATING PLOT 1: Electron Energy Spectra")
print("="*70)

fig1, ax1 = plt.subplots(figsize=(10, 6))

for gas in ['H', 'He', 'N']:
    print(f"\nProcessing {gas_labels[gas]}...")
    ts = ts_data[gas]
    
    # Get final iteration
    final_iteration = ts.iterations[-1]
    print(f"  Final iteration: {final_iteration}")
    
    # Load electron data - momentum components and weight
    # gamma needs to be calculated from ux, uy, uz
    ux, uy, uz, w = ts.get_particle(['ux', 'uy', 'uz', 'w'], 
                                     species='electrons', 
                                     iteration=final_iteration)
    
    # Calculate gamma from momentum
    # gamma = sqrt(1 + (ux^2 + uy^2 + uz^2) / (m_e * c)^2)
    # Note: ux, uy, uz are already in units of m_e * c
    u_squared = ux**2 + uy**2 + uz**2
    gamma = np.sqrt(1 + u_squared)
    
    print(f"  Total particles: {len(gamma)}")
    
    # Convert gamma to kinetic energy in MeV
    # E_kinetic = (gamma - 1) * m_e * c^2
    E_MeV = (gamma - 1) * m_e * c**2 / (1e6 * e)  # Convert J to MeV
    
    # Filter for accelerated electrons only (exclude bulk plasma)
    # Bulk plasma has gamma ~ 1, accelerated electrons have gamma >> 1
    # Use threshold of 10 MeV to separate accelerated from bulk
    E_threshold = 10.0  # MeV
    mask = E_MeV > E_threshold
    
    E_accelerated = E_MeV[mask]
    w_accelerated = w[mask]
    
    print(f"  Accelerated electrons (E > {E_threshold} MeV): {len(E_accelerated)}")
    
    if len(E_accelerated) == 0:
        print(f"  WARNING: No accelerated electrons found for {gas}")
        continue
    
    print(f"  Energy range: {E_accelerated.min():.2f} - {E_accelerated.max():.2f} MeV")
    print(f"  Mean energy: {np.average(E_accelerated, weights=w_accelerated):.2f} MeV")
    
    # Create weighted histogram
    # Bin edges from 0 to max energy, with fine resolution
    E_max = max(E_MeV.max(), 200)  # At least 200 MeV range
    bins = np.linspace(0, E_max, 100)
    
    # Weight by particle weight for correct charge representation
    hist, bin_edges = np.histogram(E_accelerated, bins=bins, weights=w_accelerated)
    
    # Normalize to get dN/dE
    bin_width = bin_edges[1] - bin_edges[0]
    hist_norm = hist / (bin_width * np.sum(hist))
    
    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot
    ax1.plot(bin_centers, hist_norm, 
             label=gas_labels[gas], 
             color=gas_colors[gas],
             linewidth=2.5,
             alpha=0.8)

# Formatting
ax1.set_xlabel('Electron Energy (MeV)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Normalized Distribution (arb. units)', fontsize=12, fontweight='bold')
ax1.set_title('Final Electron Energy Spectra Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-5)
ax1.grid(True, alpha=0.3, which='both')

# Add info text
info_text = f'Target: $n_e$ = {n_e:.1e} m$^{{-3}}$\n$a_0$ = {a0:.1f}'
ax1.text(0.98, 0.97, info_text, 
         transform=ax1.transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10)

plt.tight_layout()
plt.savefig('electron_energy_spectra.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: electron_energy_spectra.png")
plt.close()

# ==========================================
# PLOT 2: LASER AMPLITUDE EVOLUTION
# ==========================================

print("\n" + "="*70)
print("GENERATING PLOT 2: Laser Amplitude Evolution")
print("="*70)

fig2, ax2 = plt.subplots(figsize=(12, 6))

for gas in ['H', 'He', 'N']:
    print(f"\nProcessing {gas_labels[gas]}...")
    lpa = lpa_data[gas]
    ts = ts_data[gas]
    
    # Get all iterations and corresponding times
    iterations = ts.iterations
    times = ts.t
    
    print(f"  Total iterations: {len(iterations)}")
    print(f"  Time range: {times[0]:.2e} to {times[-1]:.2e} s")
    
    # Calculate a0 for each iteration
    a0_values = []
    z_positions = []
    
    for i, iteration in enumerate(iterations):
        try:
            # Get a0 at this iteration (polarization is 'x' for linear polarization)
            a0 = lpa.get_a0(iteration=iteration, pol='x')
            
            # Calculate effective propagation distance
            # In moving window: z_prop = c * t (laser has propagated this far)
            z_prop = c * times[i]
            
            a0_values.append(a0)
            z_positions.append(z_prop * 1e6)  # Convert to µm
            
        except Exception as e:
            print(f"  Warning: Could not get a0 for iteration {iteration}: {e}")
            continue
    
    print(f"  Successfully processed {len(a0_values)} iterations")
    print(f"  a₀ range: {min(a0_values):.3f} to {max(a0_values):.3f}")
    print(f"  Propagation distance: {min(z_positions):.1f} to {max(z_positions):.1f} µm")
    
    # Plot
    ax2.plot(z_positions, a0_values,
             label=gas_labels[gas],
             color=gas_colors[gas],
             linewidth=2.5,
             alpha=0.8,
             marker='o',
             markersize=4,
             markevery=2)

# Add theoretical length scales

# Formatting
ax2.set_xlabel('Propagation Distance $z$ (µm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Peak Normalized Amplitude $a_0$', fontsize=12, fontweight='bold')
ax2.set_title('Laser Amplitude Evolution Through Plasma', fontsize=14, fontweight='bold', pad=15)
ax2.legend(framealpha=0.9, loc='best', ncol=2, fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('laser_amplitude_evolution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: laser_amplitude_evolution.png")
plt.close()

# ==========================================
# PLOT 3: WAKEFIELD STRUCTURE
# ==========================================

print("\n" + "="*70)
print("GENERATING PLOT 3: Wakefield Structure")
print("="*70)

fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, gas in enumerate(['H', 'He', 'N']):
    print(f"\nProcessing {gas_labels[gas]}...")
    ts = ts_data[gas]
    ax = axes[idx]
    
    # Use middle time step
    times = ts.t
    mid_idx = len(times) // 2
    mid_time = times[mid_idx]
    print(f"  Using time: {mid_time:.6e} s ({mid_time*1e15:.2f} fs)")
    
    # Get charge density (rho)
    rho, info_rho = ts.get_field(field='rho', t=mid_time)
    z_rho = info_rho.z
    r_rho = info_rho.r
    
    print(f"  Density field shape: {rho.shape}")
    
    # Get longitudinal electric field Ez (mode 0 - axisymmetric wake)
    Ez, info_Ez = ts.get_field(field='E', coord='z', t=mid_time, 
                                m=0, slice_across='r')
    z_Ez = info_Ez.z
    
    print(f"  Ez shape: {Ez.shape}")
    print(f"  Ez range: {Ez.min():.2e} to {Ez.max():.2e} V/m")
    
    # Convert Ez to GV/m for better readability
    Ez_GV = Ez / 1e9
    
    # Convert charge density from C/m³ to electron density cm⁻³
    # rho [C/m³] → n [electrons/cm³] = rho / e / 10^6
    # (divide by elementary charge, then convert m⁻³ to cm⁻³)
    rho_cm3 = rho / 1e6  # electrons/cm³
    
    # Plot charge density as 2D colormap
    # Use symmetric colormap centered at 0
    # Use percentile-based limits to enhance visibility of plasma structure
    # This excludes extreme outliers and focuses on the wake structure
    rho_percentile = 95  # Use 95th percentile to exclude outliers
    rho_max = np.percentile(np.abs(rho_cm3), rho_percentile)
    
    # Alternative: use a fixed fraction of the maximum
    # rho_max = 0.3 * np.max(np.abs(rho_cm3))
    
    im = ax.imshow(rho_cm3,
                   extent=[z_rho.min()*1e6, z_rho.max()*1e6,
                          r_rho.min()*1e6, r_rho.max()*1e6],
                   origin='lower',
                   aspect='auto',
                   cmap='RdBu_r',  # Red-blue diverging colormap
                   vmin=-rho_max,
                   vmax=rho_max,
                   interpolation='bilinear')
    
    # Add colorbar for density
    cbar = plt.colorbar(im, ax=ax, pad=0.15)
    cbar.set_label(r'Electron Density $n_e$ (cm$^{-3}$) at time t', fontsize=10)
    
    # Create twin y-axis for Ez
    ax2 = ax.twinx()
    
    # Plot Ez on the twin axis
    ax2.plot(z_Ez * 1e6, Ez_GV,
            color='grey',
            linewidth=2.5,
            alpha=0.9,
            label='$E_z$ (wake)')
    
    # Add zero line
    ax2.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set consistent y-axis limits for Ez across all subplots for comparison
    ax2.set_ylim(-1000, 1000)  # GV/m
    
    # Formatting for main axis
    ax.set_xlabel('$z$ (µm)', fontsize=11, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('$r$ (µm)', fontsize=11, fontweight='bold')
    ax.set_title(f"{gas_labels[gas]} at t = {mid_time*1e15:.2f} fs", fontsize=12, fontweight='bold')
    
    # Formatting for twin axis
    if idx == 2:
        ax2.set_ylabel('$E_z$ (GV/m)', fontsize=11, fontweight='bold', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.legend(loc='upper right', framealpha=0.7, fontsize=9)
    
    print(f"  ✓ Completed {gas_labels[gas]} subplot")

# Overall title
fig3.suptitle('Wakefield Structure: Charge Density + Longitudinal Electric Field',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('wakefield_structure.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: wakefield_structure.png")
plt.close()

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated plots:")
print("  1. electron_energy_spectra.png")
print("  2. laser_amplitude_evolution.png")
print("  3. wakefield_structure.png")
print("\nAll plots are publication-quality (300 DPI).")
print("\nKey observations to check:")
print("  • Do energy spectra show quasi-monoenergetic peaks?")
print("  • Does nitrogen show faster laser amplitude decay?")
print("  • Are wakefield structures cleaner in hydrogen?")
print("  • How do characteristic lengths compare to theory?")
print("\nRefer to EXECUTION_PLAN.md for interpretation guidance.")
print("="*70 + "\n")
