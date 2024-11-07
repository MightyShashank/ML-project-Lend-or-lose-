import numpy as np
import matplotlib.pyplot as plt

# Parameters
track_length = 10  # arbitrary length of the track
theta_frequency = 8  # Hz (theta frequency)
theta_cycle_time = 1 / theta_frequency  # Duration of one theta cycle
total_time = 60  # total time in seconds
time_steps = np.linspace(0, total_time, 1000)  # simulation time steps

# Generate rat's position as it moves along a track
def simulate_position(total_time, track_length):
    speed = 0.1  # speed of the rat in arbitrary units
    position = (speed * time_steps) % track_length
    return position

# Generate a place field for the CA1 pyramidal cell
def place_field(position, center=5, width=1):
    return np.exp(-((position - center) ** 2) / (2 * width ** 2))

# Phase precession: advance phase as rat moves through the place field
def phase_precession(position, place_center, max_phase_shift=360):
    relative_position = (position - place_center) / track_length
    phase_shift = max_phase_shift * (0.5 - relative_position)
    return phase_shift % 360  # wrap phase to stay within [0, 360]

# Simulate rat's position and neuron firing
position = simulate_position(total_time, track_length)
place_center = track_length / 2  # center of the place field
firing_rate = place_field(position, center=place_center)

# Compute the phase precession for each position
phases = phase_precession(position, place_center)

# Plot the trajectory and phase precession
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Spatial pattern of firing
axs[0].scatter(position, np.zeros_like(position), c=firing_rate, cmap='hot', s=2)
axs[0].set_title("Panel A: Spatial Firing Pattern")
axs[0].set_xlabel("Position on Track")
axs[0].set_yticks([])

# Panel B: Phase precession vs. position
axs[1].scatter(position, phases, c=firing_rate, cmap='hot', s=2)
axs[1].set_title("Panel B: Theta Phase vs. Position")
axs[1].set_xlabel("Position on Track")
axs[1].set_ylabel("Theta Phase (degrees)")

# Panel C: Firing rate over time in theta cycles
# Bin data into theta cycles and plot firing histogram
theta_cycle_indices = (time_steps % theta_cycle_time) / theta_cycle_time * 360
axs[2].hist(theta_cycle_indices, bins=30, weights=firing_rate, color='black')
axs[2].set_title("Panel C: Firing Rate Over Theta Cycles")
axs[2].set_xlabel("Theta Phase (degrees)")
axs[2].set_ylabel("Firing Rate")

plt.tight_layout()
plt.show()
