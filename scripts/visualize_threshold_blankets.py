#!/usr/bin/env python3
"""
Visualize the Neural Landscape with "Blankets" (Planes) representing
activation thresholds. Regions poking through the blankets are active.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from memorymapping.math import NeuralLandscape

def visualize_blankets():
    os.makedirs("graphs", exist_ok=True)
    landscape = NeuralLandscape()
    
    # 1. Setup Grid
    limit = 5
    n = 100
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    Z = landscape.generate_landscape(X, Y)
    
    # 2. Define Thresholds (Blankets)
    # Let's pick levels relative to the data range
    z_min, z_max = np.min(Z), np.max(Z)
    z_range = z_max - z_min
    
    # Blanket 1: Low (Noise Floor) - Blue/Cyan
    t1 = z_min + 0.3 * z_range
    # Blanket 2: Mid (Salient) - Magenta
    t2 = z_min + 0.6 * z_range
    # Blanket 3: High (Burst) - Yellow
    t3 = z_min + 0.85 * z_range
    
    # Create plane data (same shape as X,Y)
    Z_t1 = np.full_like(Z, t1)
    Z_t2 = np.full_like(Z, t2)
    Z_t3 = np.full_like(Z, t3)
    
    # 3. Visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Landscape (Solid)
    # Using 'terrain' or 'gist_earth' for land-feel
    surf = ax.plot_surface(X, Y, Z, cmap='gist_earth', alpha=0.9, antialiased=True, shade=True)
    
    # Plot Blankets (Transparent)
    # We use shade=False to make them look like glowing laser planes
    ax.plot_surface(X, Y, Z_t1, color='cyan', alpha=0.2, shade=False, linewidth=0)
    ax.plot_surface(X, Y, Z_t2, color='magenta', alpha=0.2, shade=False, linewidth=0)
    ax.plot_surface(X, Y, Z_t3, color='yellow', alpha=0.2, shade=False, linewidth=0)
    
    # Text Labels
    # Place text in 3D space
    ax.text(limit, limit, t1, "  Low Threshold", color='blue')
    ax.text(limit, limit, t2, "  Mid Threshold", color='purple')
    ax.text(limit, limit, t3, "  High Threshold", color='orange')
    
    ax.set_title("Neural Landscape with Threshold 'Blankets'\n(Peaks piercing blankets represent active memories)")
    ax.set_xlabel("State X")
    ax.set_ylabel("State Y")
    ax.set_zlabel("Energy Potential")
    
    ax.view_init(elev=25, azim=-45)
    
    # Dark background for sci-fi feel?
    # plt.style.use('dark_background') # Global style change might affect subsequent plots, let's keep it manual
    
    path = "graphs/neural_threshold_blankets.png"
    plt.savefig(path, dpi=300) # Removed facecolor='black' to keep standard axes visibility unless requested
    print(f"Visualization saved to {path}")

if __name__ == "__main__":
    visualize_blankets()
