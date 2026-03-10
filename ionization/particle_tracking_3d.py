import os
os.environ["OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from openpmd_viewer import ParticleTracker
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e, epsilon_0, pi
import matplotlib.animation as animation

# ==========================================
# CONFIGURATION
# ==========================================

a0 = 1.8  # Fixed a0 value
modes = ['pure_he', 'doped']
dopant_list = ['N', 'Ar']  # List of dopants to compare
base_dir = './diags_n3.5e+24_tracked'

# Physical parameters
n_e_target = 2.5e23
omega_p = np.sqrt(n_e_target * e**2 / (m_e * epsilon_0))
E_wb = 96 * np.sqrt(n_e_target / 1e6) # Cold wavebreaking limit (V/m) approx formula

# Energy threshold for injected electrons
E_threshold_MeV = 260
gamma_threshold = 1 + (E_threshold_MeV * 1e6 * e) / (m_e * c**2)
uz_threshold = np.sqrt(gamma_threshold**2 - 1)

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

ts = load_data(a0, 'Ar', 'doped')

pt = ParticleTracker(ts, iteration=57000, select={'uz':[uz_threshold,None],'z':[1500e-6,1502e-6]}, species='electrons_injected', preserve_particle_index=True)

x_trajectories, y_trajectories, z_trajectories = ts.iterate(ts.get_particle, ['x', 'y', 'z'], select=pt, species='electrons_injected')

# Skip the first empty array (iteration 0 has no particles on the grid yet).
x_traj = np.array(x_trajectories[1:])
y_traj = np.array(y_trajectories[1:])
z_traj = np.array(z_trajectories[1:])

# Convert to µm to simplify plotting
x_traj_um = x_traj * 1e6
y_traj_um = y_traj * 1e6
z_traj_um = z_traj * 1e6

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Fixed limits based on the full range of the data
ax.set_xlim3d(np.nanmin(z_traj_um), np.nanmax(z_traj_um))
ax.set_ylim3d(np.nanmin(x_traj_um), np.nanmax(x_traj_um))
ax.set_zlim3d(np.nanmin(y_traj_um), np.nanmax(y_traj_um))

ax.set_xlabel('z (µm)')
ax.set_ylabel('x (µm)')
ax.set_zlabel('y (µm)')
ax.set_title(f'Particle trajectories (a0={a0})')

n_iterations, n_particles = x_traj.shape

# Generate distinct colors for each particle using a colormap
colors = plt.cm.viridis(np.linspace(0, 1, n_particles))

# Create empty lines for each particle
lines = [ax.plot([], [], [], lw=0.75, alpha=0.7, color=colors[i])[0] for i in range(n_particles)]

def update_lines(num, z_arr, x_arr, y_arr, lines):
    # num goes from 0 to n_iterations - 1
    for p, line in enumerate(lines):
        # Update the path up to the current iteration 'num'
        # NaN values (before particle is born) will not be drawn by matplotlib
        line.set_data_3d(z_arr[:num, p], x_arr[:num, p], y_arr[:num, p])
    return lines

print("Rendering animation... This might take a minute...")
ani = animation.FuncAnimation(
    fig, update_lines, frames=n_iterations, 
    fargs=(z_traj_um, x_traj_um, y_traj_um, lines), interval=100
)

writer = animation.FFMpegWriter(fps=10)
ani.save('trajectories.mp4', writer=writer, dpi=600)
print("Saved animation to trajectories.mp4")