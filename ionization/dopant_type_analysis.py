"""
Calibration Analysis Script for Ionization Injection Tuning

This script analyzes the a0 scan to determine the optimal laser amplitude
for ionization injection. It generates proof plots for phase space separation,
injection thresholds, and wakefield nonlinearity.
"""

import numpy as np
import matplotlib.pyplot as plt
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

scan_a0 = [2.0]  # Fixed a0 value
modes = ['pure_he', 'doped']
dopant_species = 'N'  # 'N', 'Ne', 'Ar'
base_dir = './diags_calib'

# Physical parameters
n_e_target = 7.0e24
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
E_wb = 96 * np.sqrt(n_e_target / 1e6) # Cold wavebreaking limit (V/m) approx formula

# ==========================================
# DATA LOADING
# ==========================================

def load_data(a0, mode):
    if mode == 'doped':
        path = f"{base_dir}/a{a0}_{mode}_{dopant_species}/hdf5"
    else:
        path = f"{base_dir}/a{a0}_{mode}/hdf5"
    if not os.path.exists(path):
        print(f"Warning: Data not found for a0={a0}, mode={mode}, dopant={dopant_species} at {path}")
        return None
    return LpaDiagnostics(path)

# ==========================================
# PLOT 1: PHASE SPACE SEPARATION (z vs pz)
# ==========================================

def plot_phase_space(a0_target):
    print(f"\nGenerating Plot : Phase Space Separation ({dopant_species}-doped)...")
    
    ts = load_data(a0_target, 'doped')
    if ts is None:
        return

    iteration = ts.iterations[-1] # Change interation if needed
    
    # Get data for both species
    z_he, uz_he, w_he = ts.get_particle(['z', 'uz', 'w'], species='electrons_he', iteration=iteration)
    z_dopant, uz_dopant, w_dopant = ts.get_particle(['z', 'uz', 'w'], species='electrons_dopant', iteration=iteration)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Dopant electrons (Injected) - Plot FIRST (Background layer)
    if len(z_dopant) > 0:
        ax.scatter(z_dopant*1e6, uz_dopant, 
                   s=5, color='red', alpha=0.3, label=f'{dopant_species} Electrons (Injected)', zorder=1)
    else:
        ax.text(0.5, 0.5, f"No {dopant_species} Electrons Injected", 
                transform=ax.transAxes, ha='center', color='red')

    # Plot Helium electrons (Bulk) - Plot SECOND (Foreground layer)
    if len(z_he) > 0:
        indices_he = np.random.choice(len(z_he), size=min(10000, len(z_he)), replace=False)
        ax.scatter(z_he[indices_he]*1e6, uz_he[indices_he], 
                   s=5, color='blue', alpha=0.5, label='He Electrons (Bulk)', zorder=2)
    
    ax.set_xlabel('z (µm)')
    ax.set_ylabel('$p_z / m_e c$')
    ax.set_title(f'Phase Space Separation ({dopant_species}-doped)')
    ax.legend()
    
    filename = f'{dopant_species}_phase_space.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()

# ==========================================
# PLOT 2: INJECTION THRESHOLD A0 SCAN
# ==========================================

def plot_charge_scan():
    print(f"\nGenerating Plot : Injected Charge Scan ({dopant_species}-doped)...")
    
    # Metric: Total Injected Charge (pC)
    # We define "Injected" as E > 2 MeV to exclude bulk fluid
    E_threshold_MeV = 2.0
    
    charges_he = []
    charges_doped = [] 
    valid_a0 = []

    for a0 in scan_a0:
        ts_he = load_data(a0, 'pure_he')
        ts_doped = load_data(a0, 'doped')
        
        if ts_he is None or ts_doped is None:
            continue
            
        valid_a0.append(a0)
        iteration = ts_he.iterations[-1]
        
        # Pure He (Noise)
        try:
            # Get current distribution along z
            # select particles with energy > threshold
            # E = (gamma - 1) * m_e * c^2, where gamma = sqrt(1 + uz^2)
            # For E > E_threshold: uz > sqrt((1 + E_threshold/0.511)^2 - 1)
            gamma_threshold = 1 + E_threshold_MeV / 0.511
            uz_threshold = np.sqrt(gamma_threshold**2 - 1)
            I_he, info_he = ts_he.get_current(species='electrons_he', iteration=iteration, 
                                               select={'uz': [uz_threshold, None]})
            # get_current returns current in Amperes
            # To get charge, integrate: Q = ∫ I dt = I * (dz/c) summed over all bins
            # This gives total charge passing through a cross-section
            Q_he = np.sum(np.abs(I_he)) * info_he.dz / c * 1e12  # Convert to pC
            charges_he.append(Q_he)
                
        except Exception as err:
            print(f"  Error reading He data for a0={a0}: {err}")
            charges_he.append(0.0)
        
        # Doped (Signal)
        try:
            # Get current distribution along z
            gamma_threshold = 1 + E_threshold_MeV / 0.511
            uz_threshold = np.sqrt(gamma_threshold**2 - 1)
            I_dopant, info_dopant = ts_doped.get_current(species='electrons_dopant', iteration=iteration,
                                                select={'uz': [uz_threshold, None]})
            # get_current returns current in Amperes
            # To get charge, integrate: Q = ∫ I dt = I * (dz/c) summed over all bins
            Q_dopant = np.sum(np.abs(I_dopant)) * info_dopant.dz / c * 1e12  # Convert to pC
            charges_doped.append(Q_dopant)
                
        except Exception as err:
            print(f"  Error reading Doped data for a0={a0}: {err}")
            charges_doped.append(0.0)

    if not valid_a0:
        print("No valid data found for threshold scan.")
        return [], [], []

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(valid_a0, charges_he, 'o--', color='grey', label='Pure He (Noise)')
    ax.plot(valid_a0, charges_doped, 'o-', color='red', label=f'{dopant_species}-Doped (Signal)')
    
    ax.set_xlabel('$a_0$')
    ax.set_ylabel(f'Injected Charge (pC) [E > {E_threshold_MeV} MeV]')
    ax.set_title(f'Injected Charge Scan ({dopant_species}-doped)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{dopant_species}_charge_scan.png', dpi=300)
    print(f"✓ Saved: {dopant_species}_charge_scan.png")
    plt.close()
    
    return valid_a0, charges_he, charges_doped

# ==========================================
# PLOT 3: 
# ==========================================

def plot_wakefield_check(a0_target):
    print(f"\nGenerating Plot : Wakefield Nonlinearity Check (Pure He)...")
    
    ts = load_data(a0_target, 'pure_he')
    if ts is None:
        return

    iteration = ts.iterations[-1]
    
    # Get Ez field on axis
    Ez, info = ts.get_field(field='E', coord='z', iteration=iteration, slice_across=['r'])
    
    # Construct z axis
    z = info.z * 1e6 # µm
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z, Ez, color='blue', label='$E_z$')
    
    # Plot Wavebreaking Limit
    ax.axhline(E_wb, color='k', linestyle='--', label='$E_{WB}$ (Cold)')
    ax.axhline(-E_wb, color='k', linestyle='--')
    
    ax.set_xlabel('z (µm)')
    ax.set_ylabel('$E_z$ (V/m)')
    ax.set_title('Wakefield Structure (Pure He)')
    ax.legend()
    
    filename = f'{dopant_species}_wakefield_check.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()


# ==========================================
# PLOT 4: WAKEFIELD STRUCTURE
# ==========================================

def plot_wakefield_structure(a0_target):
    print(f"\nGenerating Plot : Wakefield Structure (a0={a0_target}, dopant={dopant_species})...")
    
    modes_to_plot = ['pure_he', 'doped']
    labels = {'pure_he': 'Pure He', 'doped': f'{dopant_species}-Doped'}
    
    # Create figure with 2 subplots (one for each mode)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, mode in enumerate(modes_to_plot):
        ax = axes[idx]
        ts = load_data(a0_target, mode)
        
        if ts is None:
            ax.text(0.5, 0.5, "Data Not Available", ha='center')
            continue

        # Use last iteration
        iteration = ts.iterations[-1]
        t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
        
        # Get charge density (rho)
        rho, info_rho = ts.get_field(field='rho', iteration=iteration)
        z_rho = info_rho.z
        r_rho = info_rho.r
        
        # Get longitudinal electric field Ez (mode 0 - axisymmetric wake)
        Ez, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration, 
                                    m=0, slice_across='r')
        z_Ez = info_Ez.z
        
        # Convert Ez to GV/m
        Ez_GV = Ez / 1e9
        
        # Convert charge density to relative
        # rho is C/m^3. 
        # to C/cm^3: / 1e6
        rho_cm3 = rho / 1e6
        
        
        # Plot charge density
        rho_percentile = 99
        rho_max = np.percentile(np.abs(rho_cm3), rho_percentile)
        
        im = ax.imshow(rho_cm3,
                       extent=[z_rho.min()*1e6, z_rho.max()*1e6,
                              r_rho.min()*1e6, r_rho.max()*1e6],
                       origin='lower',
                       aspect='auto',
                       cmap='RdBu_r',
                       vmin=-rho_max,
                       vmax=rho_max,
                       interpolation='bilinear')
        
        cbar = plt.colorbar(im, ax=ax, pad=0.15)
        cbar.set_label(r'$n_e$ (cm$^{-3}$)', fontsize=10)
        
        # Twin axis for Ez
        ax2 = ax.twinx()
        ax2.plot(z_Ez * 1e6, Ez_GV, color='grey', linewidth=2, alpha=0.8, label='$E_z$')
        ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax2.set_ylim(-1000, 1000) 
        
        ax.set_xlabel('$z$ (µm)')
        ax.set_ylabel('$r$ (µm)')
        ax.set_title(f"{labels[mode]} (t = {t_fs:.1f} fs)")
        
        if idx == 1:
            ax2.set_ylabel('$E_z$ (GV/m)', color='grey')
        
        ax2.tick_params(axis='y', labelcolor='grey')
        
    fig.suptitle(f'Wakefield Structure ({dopant_species}-doped)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f'{dopant_species}_wakefield_structure.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Use fixed a0
    a0_fixed = scan_a0[0]
    
    # Run Charge Scan first to get data
    a0s, Q_he, Q_doped = plot_charge_scan()
    
    # Run detailed plots for the fixed a0
    plot_phase_space(a0_target=a0_fixed)
    plot_wakefield_check(a0_target=a0_fixed)
    plot_wakefield_structure(a0_target=a0_fixed)
    
    # Run Recommendation
    recommend_operating_point(a0s, Q_he, Q_doped)
