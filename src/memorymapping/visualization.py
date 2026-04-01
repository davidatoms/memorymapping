import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import os

# Create output directory
os.makedirs("graphs", exist_ok=True)

def generate_neural_spike_field(X, Y, T, memory_retrieval=False):
    """
    Generate 3D neural spike activity patterns
    X, Y: spatial coordinates (brain region)
    T: time dimension
    """
    
    # Initialize spike field
    spikes = np.zeros((len(T), len(Y), len(X)))
    
    if memory_retrieval:
        # Simulate memory retrieval: coordinated wave patterns
        
        # Memory pattern 1: Traveling wave from left to right
        wave_speed = 2.0
        for i, t in enumerate(T):
            wave_front = wave_speed * t
            # Create a traveling wave with some randomness
            for j, y in enumerate(Y):
                for k, x in enumerate(X):
                    # Distance from wave front
                    dist_from_front = abs(x - wave_front)
                    
                    # High activity near the wave front
                    if dist_from_front < 1.5:
                        # Synchronized firing with some noise
                        base_activity = 3.0 * np.exp(-dist_from_front**2 / 0.5)
                        # Add theta rhythm (6-8 Hz) and gamma (40 Hz) modulation
                        theta_mod = 1 + 0.3 * np.sin(2 * np.pi * 7 * t)
                        gamma_mod = 1 + 0.2 * np.sin(2 * np.pi * 40 * t + x + y)
                        
                        spike_prob = base_activity * theta_mod * gamma_mod
                        spikes[i, j, k] = max(0, spike_prob + np.random.normal(0, 0.5))
        
        # Memory pattern 2: Circular spreading activation
        center_x, center_y = 5, 5
        for i, t in enumerate(T):
            spread_radius = 1.5 * t
            for j, y in enumerate(Y):
                for k, x in enumerate(X):
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Ring of activation spreading outward
                    if abs(dist_from_center - spread_radius) < 1.0 and dist_from_center < 8:
                        ring_activity = 2.5 * np.exp(-abs(dist_from_center - spread_radius)**2 / 0.3)
                        # Add neural oscillations
                        alpha_mod = 1 + 0.4 * np.sin(2 * np.pi * 10 * t)
                        spikes[i, j, k] += max(0, ring_activity * alpha_mod + np.random.normal(0, 0.3))
    
    else:
        # Random background activity
        for i, t in enumerate(T):
            # Low-level random firing
            background = np.random.exponential(0.2, (len(Y), len(X)))
            # Add some spatial correlation
            for j in range(1, len(Y)-1):
                for k in range(1, len(X)-1):
                    # Nearby neurons influence each other
                    local_avg = np.mean(background[j-1:j+2, k-1:k+2])
                    background[j, k] = 0.7 * background[j, k] + 0.3 * local_avg
            
            spikes[i, :, :] = background
    
    return spikes

# Set up spatial and temporal grids
x = np.linspace(0, 10, 40)  # Brain region X (e.g., 10mm across)
y = np.linspace(0, 10, 40)  # Brain region Y 
t = np.linspace(0, 2, 60)   # Time (2 seconds, 30Hz sampling)

X, Y = np.meshgrid(x, y)

# Generate neural activity
print("Generating neural spike patterns...")
spike_activity = generate_neural_spike_field(x, y, t, memory_retrieval=True)

# Create 3D visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: 3D surface showing activity at a specific time point
ax1 = fig.add_subplot(221, projection='3d')
time_idx = 30  # Middle of the time series
surface = ax1.plot_surface(X, Y, spike_activity[time_idx], 
                          cmap='hot', alpha=0.8, linewidth=0)
ax1.set_title(f'Neural Activity at t={t[time_idx]:.2f}s')
ax1.set_xlabel('X Position (mm)')
ax1.set_ylabel('Y Position (mm)')
ax1.set_zlabel('Spike Rate (Hz)')

# Plot 2: Time series of activity at specific locations
ax2 = fig.add_subplot(222)
# Pick a few neurons to show
neuron_locs = [(10, 10), (20, 20), (30, 15), (15, 30)]
colors = ['red', 'blue', 'green', 'orange']

for (i, j), color in zip(neuron_locs, colors):
    neuron_activity = spike_activity[:, j, i]
    ax2.plot(t, neuron_activity, color=color, linewidth=2, 
            label=f'Neuron at ({x[i]:.1f}, {y[j]:.1f})')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Spike Rate (Hz)')
ax2.set_title('Individual Neuron Spike Trains')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Heatmap showing spatial activity over time
ax3 = fig.add_subplot(223)
# Average activity across Y dimension to show X vs Time
activity_profile = np.mean(spike_activity, axis=1)
im = ax3.imshow(activity_profile.T, aspect='auto', cmap='hot', 
               extent=[t[0], t[-1], x[0], x[-1]], origin='lower')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('X Position (mm)')
ax3.set_title('Spatiotemporal Activity Pattern')
plt.colorbar(im, ax=ax3, label='Spike Rate (Hz)')

# Plot 4: Network connectivity inference
ax4 = fig.add_subplot(224)
# Show cross-correlation between different spatial locations
correlation_matrix = np.zeros((len(neuron_locs), len(neuron_locs)))
for i, (x1, y1) in enumerate(neuron_locs):
    for j, (x2, y2) in enumerate(neuron_locs):
        signal1 = spike_activity[:, y1, x1]
        signal2 = spike_activity[:, y2, x2]
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        correlation_matrix[i, j] = correlation

im2 = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax4.set_title('Neural Synchronization Matrix')
ax4.set_xlabel('Neuron Index')
ax4.set_ylabel('Neuron Index')
plt.colorbar(im2, ax=ax4, label='Correlation')

# Add neuron labels
neuron_labels = [f'N{i+1}' for i in range(len(neuron_locs))]
ax4.set_xticks(range(len(neuron_locs)))
ax4.set_yticks(range(len(neuron_locs)))
ax4.set_xticklabels(neuron_labels)
ax4.set_yticklabels(neuron_labels)

plt.tight_layout()

# Save the plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join("graphs", f"neural_spikes_{timestamp}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Neural spike visualization saved to {save_path}")
print("\nThis visualization shows:")
print("1. Top-left: 3D spatial pattern of neural activity")
print("2. Top-right: Individual neuron spike trains over time") 
print("3. Bottom-left: Spatiotemporal waves of activity")
print("4. Bottom-right: Correlation between neurons (functional connectivity)") 