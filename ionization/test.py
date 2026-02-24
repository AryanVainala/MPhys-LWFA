import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0, pi
import os
import sys

iteration = 287
ts_3d = OpenPMDTimeSeries('/Users/aryan/Documents/Documents - Mac/MPhys LWFA/ionization/diags/hdf5', check_all_files=False )

# Slice across y (i.e. in a plane parallel to x-z)
Ez1, info_Ez1 = ts_3d.get_field( field='E', coord='z', iteration=iteration,
                                    slice_across='r', imshow=True )

# now retrieve and plot the bulk electron density field for the same iteration
# note: the raw dataset does not provide a generic 'rho' field, use the available
# rho_electrons_bulk quantity instead.
rho, info_rho = ts_3d.get_field(field='rho', iteration=iteration)

# build coordinate arrays in microns for plotting
z_rho = info_rho.z * 1e6  # meters -> microns
r_rho = info_rho.r * 1e6
extent_rho = [z_rho.min(), z_rho.max(), r_rho.min(), r_rho.max()]

plt.figure(figsize=(8,6))
# transpose so that axes correspond correctly
plt.imshow(rho, extent=extent_rho, origin='lower', aspect='auto', cmap='RdBu')
plt.colorbar(label=r'$\rho_{e-\mathrm{bulk}}$')
plt.title(f'Bulk electron charge density at iteration {iteration}')
plt.xlabel('z [\mum]')
plt.ylabel('r [\mum]')
plt.tight_layout()
plt.show()
