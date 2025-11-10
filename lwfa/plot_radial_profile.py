#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:35:04 2025

@author: aryan
"""

from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
import numpy as np

ts = OpenPMDTimeSeries('./diags/hdf5')

# Select iteration and load charge density
iteration = 900
rho, info = ts.get_field(field='rho', iteration=iteration)

# Get the coordinate arrays
z = info.z
r = info.r

# Choose a z-position in the middle of your plasma (e.g., 80 microns)
z_slice = z_slice = z[len(z)//2]
z_idx = np.argmin(np.abs(z - z_slice))

# Extract transverse profile at this z-position
rho_transverse = rho[:, z_idx]

# 1. Line plot of radial profile
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(r * 1e6, z * 1e6)
plt.xlabel('r (μm)')
plt.ylabel('n')
plt.title(f'Radial density profile at z = {z_slice*1e6:.1f} μm')
plt.grid(True)

# 2. 2D polar cross-section (x-y plane)
plt.subplot(1, 2, 2)
# Create 2D Cartesian grid from cylindrical data
Nr = len(r)
x = np.linspace(-r.max(), r.max(), 2*Nr)
y = np.linspace(-r.max(), r.max(), 2*Nr)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Interpolate cylindrical data onto Cartesian grid
rho_2d = np.interp(R.flatten(), r, rho_transverse).reshape(R.shape)

plt.pcolormesh(X*1e6, Y*1e6, rho_2d, shading='auto', cmap='viridis')
plt.colorbar(label='Charge density (C/m³)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title(f'Cross-section at z = {z_slice*1e6:.1f} μm')
plt.axis('equal')

plt.tight_layout()
plt.show()