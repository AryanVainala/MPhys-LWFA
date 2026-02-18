from openpmd_viewer import OpenPMDTimeSeries
import pyvista as pv
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, pi

# Open the simulation outputs using openPMD viewer
species_list = ['N','Ar','Ne']
species = species_list[0]
ts = OpenPMDTimeSeries(f'./diags_doped/a2.5_doped_{species}/hdf5')

# Create the PyVista plotter
pl = pv.Plotter()
pl.set_background("white")

# Retrieve the rho field from the simulation
# The theta=None argument constructs a 3D cartesian grid from the cylindrical data
rho_inj, meta = ts.get_field('rho_electrons_injected', iteration=ts.iterations[-1], theta=None)

# Create the grid on which PyVista can deposit the data
grid = pv.ImageData()
grid.dimensions = rho_inj.shape
grid.origin = [meta.xmin * 1e6, meta.ymin * 1e6, meta.zmin * 1e6]
grid.spacing = [meta.dx * 1e6, meta.dy * 1e6, meta.dz * 1e6]
n_inj = -rho_inj.flatten(order='F') / (e*3.5e24)
grid.point_data['values'] = n_inj
print(np.percentile(n_inj,99.98))
# Add the grid to the plotter
# Use a cutoff for rho via the clim argument since otherwise it shows only a small density spike
pl.add_volume(grid, opacity='linear', clim=[0,np.percentile(n_inj,99.99)], shade=True,
                  cmap='magma', mapper='smart', show_scalar_bar=False)
pl.add_title(f"Rendering of injected electrons for {species}")
pl.add_scalar_bar(title=r'Electron density ($n_{e}$/$n_{0}$)')
pl.show_grid()
pl.camera_position = 'xy'
pl.camera.azimuth = 65
pl.show()
