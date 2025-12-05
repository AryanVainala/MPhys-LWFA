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

a0 = 2.5  # Fixed a0 value
modes = ['pure_he', 'doped']
dopant_list = ['N', 'Ne']  # List of dopants to compare
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

def plot_laser_envelope():
    print("\nGenerating Plot : Laser Envelope...")

    ts = load_data(2.5, 'N', 'doped')
    if ts is None:
        return

    # Use the last available iteration
    iteration = ts.iterations[20]

    # Get laser envelope
    # pol='y' assumes polarization in y direction
    envelope, info = ts.get_laser_envelope(iteration=iteration, pol='x', m=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    im = ax.imshow(envelope, origin='lower', aspect='auto', cmap='inferno')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Electric Field Envelope (V/m)')
    
    ax.set_xlabel('$z$ (µm)')
    ax.set_ylabel('$r$ (µm)')
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
    ax.set_title(f'Laser Envelope (t = {t_fs:.1f} fs)')
    
    plt.tight_layout()
    filename = 'laser_envelope.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()


def plot_e_density(a0_target, dopant_species):
    """
    Plots the plasma density along with the laser profile
    """
    print(f"\nGenerating Plot : Wakefield Structure (a0={a0_target}, dopant={dopant_species})...")

    mode = 'doped'
    label = f'{dopant_species}-Doped'

    # Create figure with 1 subplot
    fig, ax = plt.subplots(figsize=(8, 5))

    ts = load_data(a0_target, dopant_species, mode)

    if ts is None:
        ax.text(0.5, 0.5, "Data Not Available", ha='center')
        plt.close()
        return

    # Use last iteration or change it
    iteration_idx = 20
    iteration = ts.iterations[iteration_idx]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
    
    # Get charge density (rho)
    rho, info_rho = ts.get_field(field='rho', iteration=iteration)
    
    z_rho = info_rho.z
    r_rho = info_rho.r

    # Get laser envelope
    laser, laser_info = ts.get_laser_envelope(iteration=iteration, pol='x', m=1)

    # Get longitudinal electric field Ez (mode 0 - axisymmetric wake)
    Ez, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration, 
                                m=0, slice_across='r')
    z_Ez = info_Ez.z
    
    # Convert Ez to GV/m
    Ez_GV = Ez / 1e9
    
    # Convert charge density to relative
    # rho is C/m^3. 
    # to C/cm^3: / 1e6
    rho_cm3 = -rho / n_e_target / e
    
    
    # Plot charge density
    rho_percentile = 100
    rho_max = np.percentile(np.abs(rho_cm3), rho_percentile)
    
    im_rho = ax.imshow(rho_cm3,
                    extent=[z_rho.min()*1e6, z_rho.max()*1e6,
                            r_rho.min()*1e6, r_rho.max()*1e6],
                    origin='lower',
                    aspect='auto',
                    cmap='Greens',
                    vmin=0,
                    vmax=1)
    
    im_laser = ax.imshow(laser,
                    origin='lower',
                    aspect='auto',
                    cmap='inferno')
    
    cbar = plt.colorbar(im_rho, ax=ax, pad=0.15)
    cbar.set_label(r'$n_e/n_0$ (cm$^{-3}$)', fontsize=10)
    
    # Twin axis for Ez
    ax2 = ax.twinx()
    ax2.plot(z_Ez * 1e6, Ez_GV, color='grey', linewidth=2, alpha=0.8, label='$E_z$')
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax2.set_ylim(-1000, 1000) 
    
    ax.set_xlabel('$z$ (µm)')
    ax.set_ylabel('$r$ (µm)')
    ax.set_title(f"{label} (t = {t_fs:.1f} fs)")
    
    ax2.set_ylabel('$E_z$ (GV/m)', color='grey')
    
    ax2.tick_params(axis='y', labelcolor='grey')
        
    fig.suptitle(f'Wakefield Structure ({dopant_species}-doped)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f'{dopant_species}_wakefield_structure.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_dopant_emittance(a0_target, dopant_species):
    print(f"\nGenerating Plot : Wakefield Structure (a0={a0_target}, dopant={dopant_species})...")
    
    mode = 'doped'
    label = f'{dopant_species}-Doped-Helium'
    
    # Create figure with 1 subplot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ts = load_data(a0_target, dopant_species, mode)
    
    if ts is None:
        ax.text(0.5, 0.5, "Data Not Available", ha='center')
        plt.close()
        return
    
    iteration = ts.iterations[len(ts.iterations)//2]
    t_fs = ts.t[ts.iterations.tolist().index(iteration)] * 1e15
    
    # Get charge density (rho)
    emt_proj, emt_slice = ts.get_emittance(iteration=iteration, species="electrons_injected", select=[])
    
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Run Charge Comparison
    # plot_dopant_comparison()
    
    # plot_phase_space(a0, dopant_species=dopant_list[0])
    # plot_dopant_comparison()
    plot_e_density(a0, 'N')
    # plot_laser_envelope()


    # Run detailed plots for each dopant at fixed a0
    for species in dopant_list:
    #     plot_phase_space(a0_target=a0, dopant_species=species)
    #     plot_e_density(a0_target=a0, dopant_species=species)
        pass
    
