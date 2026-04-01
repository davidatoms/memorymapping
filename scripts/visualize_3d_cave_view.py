#!/usr/bin/env python3
"""
Visualize the Neural Landscape as a 3D "Cave" structure.
Includes "stalagmites" (ridges) and "caves" (valleys).
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from memorymapping.math import generate_valley_landscape
import os

def visualize_cave():
    os.makedirs("graphs", exist_ok=True)
    
    # 1. Generate Data
    X, Y, Z = generate_valley_landscape(n=100, limit=5)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(15, 10))
    
    # View 1: Top-down / Bird's eye
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
    ax1.set_title("Neural Energy Landscape (Top View)")
    ax1.set_xlabel("State X")
    ax1.set_ylabel("State Y")
    ax1.set_zlabel("Energy (V)")
    ax1.view_init(elev=60, azim=45)
    
    # View 2: "Through the Middle" (Cave View)
    ax2 = fig.add_subplot(122, projection='3d')
    # Use wireframe or transparent surface to see "inside"
    surf2 = ax2.plot_surface(X, Y, Z, cmap='magma', alpha=0.6, linewidth=0.1, edgecolors='k')
    
    # Emphasize the "Stalagmites" (High Z)
    ax2.set_title("Cave View (Through the Valley)")
    ax2.set_zlim(np.min(Z), np.max(Z) + 1)
    # Low elevation angle to look "through" the valley
    ax2.view_init(elev=10, azim=-45)
    
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Energy potential')
    
    path = "graphs/kakeya_3d_cave_view.png"
    plt.savefig(path, dpi=300)
    print(f"3D Cave visualization saved to {path}")
    plt.show()

if __name__ == "__main__":
    visualize_cave()
