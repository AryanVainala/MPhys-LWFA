"""
This is a typical input script that runs a simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Modify the parameters below to suit your needs
- Type "python lwfa_script.py" in a terminal

Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.
"""

# %%
# -------
# Imports
# -------
import numpy as np
import math
from scipy.constants import c, e, m_e, epsilon_0
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint
from openpmd_viewer.addons import LpaDiagnostics as lpa
import matplotlib.pyplot as plt

# Whether to use the GPU
use_cuda = False

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (adviced for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
n_order = -1

# -----------------
# Plasma Parameters
# -----------------

p_zmin = 30.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 500.e-6 # Position of the end of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)

# ----------------
# Laser Parameters
# ----------------

a0 = 2.           # Laser amplitude
lambda0 = 0.8e-6 # Laser wavelength (micrometres)       
w0 = 5.e-6       # Laser waist
tau = 16.e-15    # Laser duration
z0 = 15.e-6      # Laser centroid

# Frequencies, wave numbers, wavelengths
omega_l  = 2*np.pi*c/lambda0                                       # laser angular frequency [rad/s]
omega_p = np.sqrt( n_e * e**2 / ( m_e * epsilon_0 ) )           # plasma angular frequency [rad/s]
kp = omega_p / c                                                  # plasma wavenumber [1/m]
lambda_p = 2. * np.pi * c / omega_p                             # plasma wavelength [m]
gamma_p = omega_l / omega_p                                               # Lorentz factor of laser group (≈ ω0/ωp)

## Blowout condition waist ##

w0 = (2 / kp) * (a0 / (1 + a0**2)**0.25)

# Calc. other laser quantities
L = c*tau                                   # Pulse Length
P = 21.4*(a0 * w0 / lambda0)**2             # Laser Power (GW)
ZR = np.pi*w0**2/lambda0                    # Rayleigh length

# Crit. power for rel. self-focusing
Pc = 17e9 * (omega_l/omega_p)**2            # [W]  (Pc ≈ 17 GW * nc/ne)

# Dephasing length
Np = 1                                                                          # Number of cycles behind pulse
Ld_lin  = (lambda_p**3 / (2 * lambda0**2))                                      # [m]  short-pulse linear estimate
Ld_bub  = (lambda_p**3 / (2 * lambda0**2)) * a0 * (np.sqrt(2)/np.pi) / Np       # [m]  blowout estimate (a0 ≳ 2–4)

# Pump depletion length
Lpd_lin = (lambda_p**3 / lambda0**2) * 2/ a0**2          # [m]  linear, short-pulse
Lpd_bub = (lambda_p**3 / lambda0**2) * a0 * np.sqrt(2)/np.pi  # [m]  blowout scaling (order-of-mag)



# ---------------------
# Simulation Parameters
# ---------------------
zmax = 30.e-6    # Right end of the simulation box (meters)
zmin = -10.e-6   # Left end of the simulation box (meters)
rmax = 20.e-6    # Length of the box along r (meters)
Lz = zmax - zmin
Lr = rmax

# ---------------------
# No. Grid Points Calc.
# ---------------------
# In axial direction 10 cells per laser wavelength
dz_target = lambda0 / 10

# In radial direction 10 cells per plasma skin depth
skin_depth = c / omega_p
dr_target = skin_depth / 10

Nz_req = math.ceil(Lz / dz_target)
Nr_req = math.ceil(Lr / dr_target)


# Defining the number of grid points
Nz = Nz_req      # Number of gridpoints along z
Nr = Nr_req      # Number of gridpoints along r
Nm = 3           # Number of modes used


# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)

# The moving window
v_window = c       # Speed of the window

# The interaction length of the simulation (meters)
L_interact = 200.e-6 # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = ( L_interact + (zmax-zmin) ) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)


# The diagnostics and the checkpoints/restarts
diag_period = 50         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 100  # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

ramp_start = 30.e-6
ramp_length = 40.e-6

# The depletion length

def depletion_length_calc(a0, lambda_p, lambda0):
    if a0 < 1:
        return (lambda_p**3 / lambda0**2) * 2/ a0**2
    elif a0 > 1:
        return (lambda_p**3 / lambda0**2) * a0 * np.sqrt(2)/np.pi

depletion_length = depletion_length_calc(a0, lambda_p, lambda0)
# print(depletion_length)

# The dephasing length

def dephasing_length_calc(a0, lambda_p, lambda0, Np):
    if a0 < 1:
        return (lambda_p**3 / (2 * lambda0**2))
    elif a0 > 1:
        return (lambda_p**3 / (2 * lambda0**2)) * a0 * (np.sqrt(2)/np.pi) / Np

# %%


dephasing_length = dephasing_length_calc(a0, lambda_p, lambda0, Np = 1)
# print(dephasing_length)


# --------------------
# Plasma Density Prof.
# --------------------


# The density profile
w_matched = w0
ramp_up = .5e-3
plateau = 3.5e-3
ramp_down = .5e-3

# Relative change divided by w_matched^2 that allows guiding
rel_delta_n_over_w2 = 1./( np.pi * 2.81e-15 * w_matched**4 * n_e )
# Define the density function
def dens_func( z, r ):
    """
    User-defined function: density profile of the plasma

    It should return the relative density with respect to n_plasma,
    at the position x, y, z (i.e. return a number between 0 and 1)

    Parameters
    ----------
    z, r: 1darrays of floats
        Arrays with one element per macroparticle
    Returns
    -------
    n : 1d array of floats
        Array of relative density, with one element per macroparticles
    """
    # Allocate relative density
    n = np.ones_like(z)
    # Make ramp up
    inv_ramp_up = 1./ramp_up
    n = np.where( z<ramp_up, z*inv_ramp_up, n )
    # Make ramp down
    inv_ramp_down = 1./ramp_down
    n = np.where( (z >= ramp_up+plateau) & \
                  (z < ramp_up+plateau+ramp_down),
              - (z - (ramp_up+plateau+ramp_down) )*inv_ramp_down, n )
    n = np.where( z >= ramp_up+plateau+ramp_down, 0, n)
    # Add transverse guiding parabolic profile
    n = n * ( 1. + rel_delta_n_over_w2 * r**2 )
    return(n)

# def dens_func( z, r ) :
#     """Returns relative density at position z and r"""
#     # Allocate relative density
#     n = np.ones_like(z)
#     # Make linear ramp
#     n = np.where( z<ramp_start+ramp_length, (z-ramp_start)/ramp_length, n )
#     # Supress density before the ramp
#     n = np.where( z<ramp_start, 0., n )
#     return(n)


# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'reflective'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Create the plasma electrons
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=dens_func, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Load initial fields
    # Create a Gaussian laser profile
    laser_profile = GaussianLaser(a0, w0, tau, z0)
    # Add the laser to the fields of the simulation
    add_laser_pulse( sim, laser_profile)

    if use_restart is False:
        # Track electrons if required (species 0 correspond to the electrons)
        if track_electrons:
            elec.track( sim.comm )
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add diagnostics
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
                  ParticleDiagnostic( diag_period, {"electrons" : elec},
                    select={"uz" : [1., None ]}, comm=sim.comm ) ]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    ### Run the simulation
    sim.step( N_step )
    print('')
    
# %%
# data = lpa('./diags/hdf5/') #Load data as a variable

# z_prop = data.t*c/ZR  #Calculate z position of diagnostic  (micrometerss)
# waist = data.iterate(data.get_laser_waist, pol='x',method='rms') / lambda_p #Gets a list of waist values
# output = np.column_stack((z_prop, waist))
# np.savetxt('plasma_channel.csv', output,
#            delimiter=',',
#            header='z_prop [m],waist [m]',
#            comments='')
# # Load uniform plasma case
# no_channel = np.genfromtxt('no_channel.csv', delimiter=',', skip_header=1)
# z_no, w_no = no_channel[:, 0], no_channel[:, 1]

# # Vacuum diffraction
# w_vac = (w0 / lambda_p) * np.sqrt(1 + z_no**2)

# # Plot all
# plt.plot(z_prop, waist, label='Parabolic Channel')
# plt.plot(z_no, w_no, '--', label='Uniform Plasma')
# plt.plot(z_no, w_vac, ':', label='Vacuum Diffraction')
# plt.xlabel(r"$c\tau / Z_{R}$")
# plt.ylabel(r"$r_s / \lambda_p$")
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%
