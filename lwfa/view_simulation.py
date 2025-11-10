#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:35:04 2025

@author: aryan
"""

from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt

ts = OpenPMDTimeSeries('./diags/hdf5')

iteration = 900

# Density (rho) on r-z grid
rho, info_rho = ts.get_field(field='rho', iteration=iteration)
z = info_rho.z
r = info_rho.r

# E_z field on r-z grid
E_z, info_Ez = ts.get_field(field='E', coord='z', iteration=iteration)
z_E = info_Ez.z
r_E = info_Ez.r

# Extract E_z at r=0 (on-axis)
r_index = 0  # First radial index (r≈0)
E_z_line = E_z[r_index, :]

fig, ax = plt.subplots(figsize=(10,5))

# Density heatmap
im = ax.imshow(
    rho,
    extent=[z.min(), z.max(), r.min(), r.max()],
    origin='lower',
    aspect='auto',
    cmap='Greens_r'
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\rho$ (SI)')

ax.set_xlabel('z (m)')
ax.set_ylabel('r (m)')

# Second y-axis for E_z
ax2 = ax.twinx()
ax2.plot(z_E, E_z_line, color='red', linewidth=1.5, label='$E_z$ at r=0')
ax2.set_ylabel(r'$E_z$ (V/m)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.show()