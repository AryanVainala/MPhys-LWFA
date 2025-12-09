#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 21:35:04 2025

@author: aryan
"""

from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0

ts = OpenPMDTimeSeries('./diags_doped_3.5_lr/a2.5_doped_Ar/hdf5')

# Simulation parameters (from lwfa_script.py)
lambda0 = 0.8e-6      # Laser wavelength
n_e = 3.5.e18*1.e6      # Plasma density
omega_p = np.sqrt(n_e * e**2 / (m_e * epsilon_0))
lambda_p = 2*np.pi*c/omega_p

iterations = ts.iterations
initial_iteration = iterations[len(iterations)//2]

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

def update_plot(iteration):
    """Update plot for given iteration"""
    iteration = int(iteration)
    
    # Get density
    rho, info_rho = ts.get_field(field='rho', iteration=iteration)
    z = info_rho.z
    r = info_rho.r
    
    # Get plasma wake (mode 0 - axisymmetric)
    E_plasma, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration, m=0, slice_across='r')

    z_rho = info_rho.z * 1e6 # Convert to microns
    r_rho = info_rho.r * 1e6
    extent_rho = [z_rho.min(), z_rho.max(), r_rho.min(), r_rho.max()]

    
    # Get laser field (mode 1 - linearly polarized)
    E_laser, _ = ts.get_field(field='E', coord='z', iteration=iteration, m=1, slice_across='r')
    
    z_E = info_Ez.z * 1e6

    # Clear axes
    ax.clear()
    if hasattr(ax, 'ax2'):
        ax.ax2.remove()
    
    # Plot density
    im = ax.imshow(
        rho,
        extent=extent_rho,
        origin='lower',
        aspect='auto',
        cmap='Greens',
        vmax = 0
    )
    
    if not hasattr(fig, 'cbar'):
        fig.cbar = fig.colorbar(im, ax=ax)
        fig.cbar.set_label(r'$\rho$ (C/m$^3$)')
    else:
        fig.cbar.update_normal(im)
    
    ax.set_xlabel('z (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(f'Iteration: {iteration}')
    
    # Plot E_z components
    ax.ax2 = ax.twinx()
    ax.ax2.plot(z_E, E_laser, 'red', linewidth=1.5, label='Laser (m=1)', alpha=0.7)
    ax.ax2.plot(z_E, E_plasma, 'royalblue', alpha=0.75, linewidth=2.0, label='Plasma Wake (m=0)')
    ax.ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.ax2.set_ylabel(r'$E_z$ (V/m)', color='red')
    ax.ax2.tick_params(axis='y', labelcolor='red')
    ax.ax2.legend(loc='upper right')
    
    fig.canvas.draw_idle()

# Create slider
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(
    ax_slider, 
    'Iteration',
    iterations[0],
    iterations[-1],
    valinit=initial_iteration,
    valstep=iterations
)

slider.on_changed(update_plot)

# Initial plot
update_plot(initial_iteration)

plt.show()