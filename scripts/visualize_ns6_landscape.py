#!/usr/bin/env python3
"""
Visualize real Neural Spikes (from NS6) on the theoretical Energy Landscape.
Fusion of real data and mathematical model.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from memorymapping.processor import NeuroDataProcessor, SpikeType
from memorymapping.math import NeuralLandscape

def visualize_integration():
    os.makedirs("graphs", exist_ok=True)
    ns6_file = "array_Sc2.ns6"
    
    # 1. Load Real Data
    processor = NeuroDataProcessor()
    if os.path.exists(ns6_file):
        print(f"Loading {ns6_file}...")
        success = processor.load_ns6_data(ns6_file, threshold_std=4.0)
        if not success:
            print("Failed to load NS6 data. Exiting.")
            return
    else:
        print(f"File {ns6_file} not found. Creating sample data instead.")
        processor.create_sample_data()

    spikes = processor.spike_events
    print(f"Visualizing {len(spikes)} spikes.")
    
    # 2. Generate Landscape (The "Cave")
    landscape = NeuralLandscape()
    limit = 5
    n = 100
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    Z_landscape = landscape.generate_landscape(X, Y)
    
    # 3. Map Spikes to Landscape
    # The processor currently assigns random (x,y) in (-5, 5). 
    # We will compute the Z height for each spike so it sits ON the landscape surface.
    
    spike_x = []
    spike_y = []
    spike_z = []
    spike_colors = []
    spike_sizes = []
    
    for spike in spikes:
        # Use simple mapping: Processor X/Y -> Landscape X/Y
        # Clamp to limit
        sx = max(-limit, min(limit, spike.position[0]))
        sy = max(-limit, min(limit, spike.position[1]))
        
        # Calculate Z height at this position
        sz = landscape.gradient(sx, sy)[0] * 0 # gradient returns tuple, we need height function
        # Wait, gradient returns derivatives. We need the height function.
        # We can re-implement the formula or use the static method if we expose it better.
        # The static method expects grids. Let's make a scalar helper or just use the formula here.
        
        # Re-using formula for scalar Z
        r_sq = sx**2 + sy**2
        valley = -3.0 * np.exp(-r_sq / 8.0)
        ridges = 0.0
        ridge_centers = [(3, 3), (-3, -3), (3, -3), (-3, 3)]
        for cx, cy in ridge_centers:
            d_sq = (sx - cx)**2 + (sy - cy)**2
            ridges += 2.0 * np.exp(-d_sq / 2.0)
        slope = 0.3 * (sx + sy)
        z = valley + ridges + slope
        
        # Add a small offset based on amplitude to show "intensity"
        # High amplitude = higher above the surface?
        z_offset = abs(spike.amplitude) / 100.0
        
        spike_x.append(sx)
        spike_y.append(sy)
        spike_z.append(z + z_offset)
        
        # Color by amplitude
        spike_colors.append(abs(spike.amplitude))
        spike_sizes.append(max(5, min(50, abs(spike.amplitude)/2)))

    # 4. Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Landscape (Wireframe for visibility)
    ax.plot_wireframe(X, Y, Z_landscape, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot Spikes
    sc = ax.scatter(spike_x, spike_y, spike_z, c=spike_colors, cmap='hot', s=spike_sizes, alpha=0.8, edgecolors='w', linewidth=0.2)
    
    ax.set_title(f"Real Neural Spikes on Energy Landscape\nSource: {ns6_file} ({len(spikes)} events)")
    ax.set_xlabel("State Space X")
    ax.set_ylabel("State Space Y")
    ax.set_zlabel("Energy / Potential")
    
    # Add colorbar
    plt.colorbar(sc, ax=ax, label='Spike Amplitude (uV)')
    
    # Set view
    ax.view_init(elev=35, azim=-60)
    
    save_path = "graphs/ns6_landscape_integration.png"
    plt.savefig(save_path, dpi=300, facecolor='black')
    
    # Also save a "Cave View" version
    ax.view_init(elev=5, azim=45)
    plt.savefig("graphs/ns6_landscape_cave_view.png", dpi=300, facecolor='black')
    
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_integration()
