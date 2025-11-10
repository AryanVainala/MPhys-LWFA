#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:35:04 2025

@author: aryan
"""

from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

ts = OpenPMDTimeSeries('./diags/hdf5')

# Get available iterations
iterations = ts.iterations

# Initial iteration
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
    
    # Get E_z field
    E_z, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration)
    z_E = info_Ez.z
    E_z_line = E_z[0, :]  # On-axis
    
    # Clear axes
    ax.clear()
    if hasattr(ax, 'ax2'):
        ax.ax2.remove()
    
    # Plot density
    im = ax.imshow(
        rho,
        extent=[z.min(), z.max(), r.min(), r.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    
    if not hasattr(fig, 'cbar'):
        fig.cbar = fig.colorbar(im, ax=ax)
        fig.cbar.set_label(r'$\rho$ (SI)')
    else:
        fig.cbar.update_normal(im)
    
    ax.set_xlabel('z (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(f'Iteration: {iteration}')
    
    # Plot E_z on second axis
    ax.ax2 = ax.twinx()
    ax.ax2.plot(z_E, E_z_line, color='red', linewidth=1.5)
    ax.ax2.set_ylabel(r'$E_z$ (V/m)', color='red')
    ax.ax2.tick_params(axis='y', labelcolor='red')
    
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