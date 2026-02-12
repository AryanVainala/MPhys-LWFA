#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:55:13 2025

@author: aryan
"""

import matplotlib
# matplotlib.use('Qt5Agg')  # Try Qt backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from openpmd_viewer import OpenPMDTimeSeries

# Load the data
ts = OpenPMDTimeSeries('./diags/hdf5/')

fig, ax = plt.subplots(figsize=(12, 6))

# Get the metadata from the first iteration to set up the plot dimensions
# We do this once to avoid re-calculating in the animation loop.
rho_initial, info = ts.get_field(iteration=ts.iterations[0], field='rho', species='electrons')

# Create the initial plot object (an image) that will be updated in each frame.
# We pass the initial data to it.
im = ax.imshow(rho_initial,
               origin='lower',
               extent=info.imshow_extent,
               cmap='Greens_r',
               aspect='auto') # 'auto' aspect ratio is good for long simulations

# Add a colorbar and labels
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Electron Charge Density (C/m$^3$)")
ax.set_xlabel("Z (m)")
ax.set_ylabel("R (m)")
title = ax.set_title(f"Iteration: {ts.iterations[0]}")


# Set the View Area (Zoom)

zmin, zmax = info.zmin, info.zmax
rmax = info.rmax
z_buffer = (zmax - zmin) * 0.2
r_buffer = rmax * 0.2

ax.set_xlim(zmin - z_buffer, zmax + z_buffer)
ax.set_ylim(-rmax - r_buffer, rmax + r_buffer)


# --- 4. Define the Animation Function ---
# This function is the core of the animation. It tells Matplotlib how to draw each frame.
# 'i' is the frame number, from 0 to len(ts.iterations)-1.
def update(i):
    # Get the iteration number for the current frame
    iteration = ts.iterations[i]
    
    # Get the density data for this iteration
    rho_data, info_data = ts.get_field(iteration=iteration, field='rho', species='electrons')
    
    # Update the data of the image object
    im.set_data(rho_data)
    
    # The density range might change over time, so we update the color scale
    im.set_clim(rho_data.min(), rho_data.max())
    
    # Update the title
    title.set_text(f"Iteration: {iteration} (Frame {i+1}/{len(ts.iterations)})")
    
    return [im, title]


# --- 5. Create and Run the Animation ---
# FuncAnimation will call the 'update' function for each frame.
# frames: number of frames to generate
# interval: delay between frames in milliseconds
anim = FuncAnimation(fig, update, frames=len(ts.iterations), interval=100, blit=True)

# Display the animation
plt.show()

# anim.save("test.gif", writer="pillow")