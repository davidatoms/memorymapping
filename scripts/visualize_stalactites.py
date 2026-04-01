#!/usr/bin/env python3
"""
Visualize the Neural Landscape as a dual-surface cave system.
Floor: Stalagmites (Energy Potentials)
Ceiling: Stalactites (inverted potentials)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from memorymapping.math import NeuralLandscape

def visualize_stalactites():
    os.makedirs("graphs", exist_ok=True)
    landscape = NeuralLandscape()
    
    # 1. Setup Grid
    limit = 5
    n = 100
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    
    # 2. Generate Floor (The standard energy landscape)
    Z_floor = landscape.generate_landscape(X, Y)
    
    # 3. Generate Ceiling (Stalactites)
    # We want the ceiling to "drip" down towards the high points of the floor (ridges).
    # Stalactites often mirror stalagmites.
    # So if Floor is high (ridge), Ceiling should be low (hanging down).
    
    # Shift floor to positive relative scale for calculation
    z_min = np.min(Z_floor)
    z_rel = Z_floor - z_min
    
    cave_height = 12.0
    
    # Ceiling Shape:
    # Start at cave_height.
    # Subtract z_rel to make it "hang down" where floor "goes up".
    # This creates columns where they meet.
    Z_ceiling = cave_height + z_min - z_rel
    
    # 4. Visualization
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Floor (Stalagmites)
    # 'gist_earth' gives a nice geological look
    surf_floor = ax.plot_surface(X, Y, Z_floor, cmap='gist_earth', 
                               alpha=1.0, linewidth=0, antialiased=False)
    
    # Plot Ceiling (Stalactites)
    # 'bone' or 'copper' for the roof. High alpha to see "through" the cave?
    # Or wireframe to really see the structure
    surf_ceiling = ax.plot_surface(X, Y, Z_ceiling, cmap='copper', 
                                 alpha=0.4, linewidth=0, antialiased=True)
    
    # Formatting
    ax.set_title("Neural Cave System: Stalactites & Stalagmites\n(Converging Memories)", fontsize=14)
    ax.set_xlabel("State Space X")
    ax.set_ylabel("State Space Y")
    ax.set_zlabel("Energy Potential")
    
    # Set limits to include both floor and ceiling
    ax.set_zlim(np.min(Z_floor), np.max(Z_ceiling))
    
    # Camera View: Inside the cave
    ax.view_init(elev=5, azim=30)
    
    # Add lighting effects / shading? (Standard plot_surface has some)
    
    path = "graphs/neural_stalactites.png"
    plt.savefig(path, dpi=300, facecolor='black')
    
    # Save a side profile too
    ax.view_init(elev=0, azim=0)
    plt.savefig("graphs/neural_stalactites_profile.png", dpi=300, facecolor='black')
    
    print(f"Stalactite visualization saved to {path}")

if __name__ == "__main__":
    visualize_stalactites()
