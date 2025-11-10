#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 05:03:04 2025

@author: aryan
"""

# %%
data = lpa('./diags/hdf5/') #Load data as a variable

z_prop = data.t*c/ZR  #Calculate z position of diagnostic  (micrometerss)
waist = data.iterate(data.get_laser_waist, pol='x',method='rms') / lambda_p #Gets a list of waist values
output = np.column_stack((z_prop, waist))
np.savetxt('plasma_channel.csv', output,
           delimiter=',',
           header='z_prop [m],waist [m]',
           comments='')
# Load uniform plasma case
no_channel = np.genfromtxt('no_channel.csv', delimiter=',', skip_header=1)
z_no, w_no = no_channel[:, 0], no_channel[:, 1]

# Vacuum diffraction
w_vac = (w0 / lambda_p) * np.sqrt(1 + z_no**2)

# Plot all
plt.plot(z_prop, waist, label='Parabolic Channel')
plt.plot(z_no, w_no, '--', label='Uniform Plasma')
plt.plot(z_no, w_vac, ':', label='Vacuum Diffraction')
plt.xlabel(r"$c\tau / Z_{R}$")
plt.ylabel(r"$r_s / \lambda_p$")
plt.legend()
plt.tight_layout()
plt.show()

# %%
