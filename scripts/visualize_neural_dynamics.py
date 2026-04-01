#!/usr/bin/env python3
"""
Visualize Hamiltonian Dynamics on the Neural Energy Landscape.
Shows the exchange between Kinetic and Potential energy ("Energy Spikes").
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from memorymapping.math import NeuralLandscape, HamiltonianSystem

def visualize_physics():
    os.makedirs("graphs", exist_ok=True)
    
    # 1. Setup System
    landscape = NeuralLandscape()
    system = HamiltonianSystem(landscape)
    
    # 2. Add Particle
    # Start high on a ridge, with some velocity
    system.add_particle(x=3.0, y=3.0, vx=-0.5, vy=-0.5, mass=1.0)
    
    # 3. Reference Landscape for plotting
    limit = 5
    n = 100
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    Z = landscape.generate_landscape(X, Y)
    
    # 4. Run Simulation
    steps = 500
    for _ in range(steps):
        system.step(dt=0.05)
        
    p = system.particles[0]
    
    # 5. Visualize Trajectory
    fig = plt.figure(figsize=(15, 6))
    
    # Left: Contours + Trajectory
    ax1 = fig.add_subplot(121)
    cont = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax1.plot(p.trajectory_x, p.trajectory_y, 'w-', linewidth=2, label='Particle Path')
    ax1.plot(p.trajectory_x[0], p.trajectory_y[0], 'ro', label='Start')
    ax1.plot(p.trajectory_x[-1], p.trajectory_y[-1], 'rx', label='End')
    ax1.set_title("Particle Trajectory on Energy Landscape")
    ax1.set_xlabel("State X")
    ax1.set_ylabel("State Y")
    ax1.legend()
    plt.colorbar(cont, ax=ax1, label='Potential Energy')
    
    # Right: Energy Spikes (Kinetic vs Potential)
    ax2 = fig.add_subplot(122)
    ke = [e[0] for e in p.energy_history]
    pe = [e[1] for e in p.energy_history]
    total = [e[2] for e in p.energy_history]
    time = np.arange(len(ke)) * 0.05
    
    ax2.plot(time, ke, 'b-', label='Kinetic Energy (Motion)', alpha=0.7)
    ax2.plot(time, pe, 'r-', label='Potential Energy (Landscape)', alpha=0.7)
    ax2.plot(time, total, 'k--', label='Total Energy', linewidth=2)
    
    ax2.set_title("Hamiltonian Dynamics: Energy Exchange")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    path = "graphs/neural_dynamics.png"
    plt.savefig(path, dpi=300)
    print(f"Physics visualization saved to {path}")
    plt.show()

if __name__ == "__main__":
    visualize_physics()
