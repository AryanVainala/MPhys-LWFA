import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from openpmd_viewer import OpenPMDTimeSeries

# =================CONFIGURATION=================
# Point this to your diagnostic folder
path_to_data = "diags_doped/a2.5_doped_N/hdf5" 
species_name = "electrons_injected"

# Which iteration do you want to use for the BACKGROUND image?
# (Pick a time step where injection is actively happening, roughly 20-30% into the run)
background_iteration = 5000  # Adjust based on your available iterations
# ===============================================

ts = OpenPMDTimeSeries(path_to_data)

def find_candidate_ids(ts, species):
    """
    Looks at the LAST iteration to find a 'winner' (trapped) 
    and a 'loser' (untrapped) particle.
    """
    last_iter = ts.iterations[-1]
    
    # Load particles from the last step
    # ux is normalized momentum (gamma * beta_x) ~ gamma for high energy
    ux, ids = ts.get_particle(var_list=["ux", "id"], 
                              species=species, 
                              iteration=last_iter)
    
    # 1. Find a TRAPPED particle (High Energy)
    # We take the particle with max energy
    trapped_idx = np.argmax(ux)
    trapped_id = ids[trapped_idx]
    
    # 2. Find an UNTRAPPED particle
    # We want a particle that was born but didn't get accelerated.
    # We look for particles with low ux, but valid IDs.
    # To make it scientifically relevant, we often want an ID "close" to the trapped one
    # implying they were born near the same time/location.
    
    # Sort IDs to find neighbors
    sorted_indices = np.argsort(ids)
    sorted_ids = ids[sorted_indices]
    sorted_ux = ux[sorted_indices]
    
    # Find the index of our trapped particle in the sorted list
    loc_in_sorted = np.where(sorted_ids == trapped_id)[0][0]
    
    # Look at neighbors (born nearby) and pick one with low momentum
    untrapped_id = None
    
    # Search nearby IDs for a low energy particle
    search_range = 1000 # Look at 1000 particles born before/after
    start = max(0, loc_in_sorted - search_range)
    end = min(len(sorted_ids), loc_in_sorted + search_range)
    
    # Filter for low energy (ux < 2, essentially just drifting)
    low_energy_candidates = sorted_ids[start:end][sorted_ux[start:end] < 5.0]
    
    if len(low_energy_candidates) > 0:
        untrapped_id = low_energy_candidates[0] # Pick the first one
    else:
        print("Could not find a neighbor with low energy, picking random low energy particle.")
        untrapped_id = ids[np.argmin(ux)]

    print(f"Trapped ID: {trapped_id} (ux={ux[trapped_idx]:.1f})")
    print(f"Untrapped ID: {untrapped_id}")
    
    return trapped_id, untrapped_id

def get_trajectories(ts, species, id_list, skip_step):
    """
    Loops over iterations with a 'skip_step' to save time.
    skip_step=n means we only look at 1 file out of every n steps.
    """
    history = {id_val: {'z': [], 'r': [], 't': []} for id_val in id_list}
    
    print(f"Extracting trajectories (Sampling every {skip_step}th file)...")
    
    # === OPTIMIZATION HERE ===
    # We slice the iterations list [::skip_step]
    reduced_iterations = ts.iterations[::skip_step]
    
    total_steps = len(reduced_iterations)
    
    for i, it in enumerate(reduced_iterations):
        # Print progress every 10 steps so you know it's not frozen
        if i % 10 == 0:
            print(f"Processing step {i}/{total_steps} (Iteration {it})")

        try:
            # We assume the particle exists. If not, openPMD might raise an error 
            # or return empty arrays depending on the version.
            # The 'select' argument is crucial for speed.
            z, r, pid = ts.get_particle(var_list=["z", "r", "id"], 
                                      species=species, 
                                      iteration=it,
                                      select={"id": id_list})
            
            t_current = ts.t[it]
            
            # Map the data to the correct ID
            for k, p_id in enumerate(pid):
                if p_id in history:
                    # Calculate Comoving Coordinate (Xi)
                    xi = z[k] - c * t_current
                    
                    history[p_id]['z'].append(xi)
                    history[p_id]['r'].append(r[k])
                    history[p_id]['t'].append(t_current)
                
        except Exception as e:
            # Usually happens if the particle hasn't been injected yet
            # or has left the window. Safe to ignore in this loop.
            pass
            
    return history

# --- EXECUTION ---

# 1. Find the IDs
try:
    id_trap, id_untrap = find_candidate_ids(ts, species_name)
    ids_to_track = [id_trap, id_untrap]

    # 2. Get the history
    trajectories = get_trajectories(ts, species_name, ids_to_track, skip_step=100)

    # 3. Plotting
    plt.figure(figsize=(10, 5))

    # A. Plot the Background Field (Electron Density)
    # We use 'rho' (charge density) or 'E' (field). 
    # visualising 'rho' allows us to see the bubble.
    # vmin/vmax controls contrast
    ts.get_field(field='rho', iteration=background_iteration, plot=True, vmin=-1e5, vmax=1e5, cmap='gray')

    # B. Overlay Trajectories
    # Note: openpmd-viewer plots usually have z (xi) on horizontal axis
    
    # Plot Trapped (Blue)
    t_data = trajectories[id_trap]
    plt.plot(np.array(t_data['z'])*1e6, np.array(t_data['r'])*1e6, 
             color='cyan', linewidth=2, label='Trapped')

    # Plot Untrapped (Red Dashed)f
    u_data = trajectories[id_untrap]
    plt.plot(np.array(u_data['z'])*1e6, np.array(u_data['r'])*1e6, 
             color='red', linestyle='--', linewidth=2, label='Untrapped')

    plt.title(f"Ionization Injection Dynamics ({species_name})")
    plt.xlabel(r'$\xi = z - ct$ ($\mu m$)')
    plt.ylabel(r'$r$ ($\mu m$)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure your 'path_to_data' is correct and species name matches your script.")