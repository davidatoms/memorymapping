#!/usr/bin/env python3
"""
Simulate ALS Progression (The Spreading Silence).
Models the spread of misfolded proteins (Prions) across the electrode array
and the resulting decay in neural firing rates.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from memorymapping.processor import NeuroDataProcessor
from scipy.ndimage import gaussian_filter

def simulate_als():
    os.makedirs("graphs", exist_ok=True)
    ns6_file = "array_Sc2.ns6"
    processor = NeuroDataProcessor()
    
    if not os.path.exists(ns6_file):
        print(f"File {ns6_file} not found.")
        return

    # 1. Get Baseline Activity (NEJM Features)
    # We use the Spike Band Power (energy) as our measure of "Health"
    result = processor.extract_features(ns6_file, bin_size_ms=50.0) # Larger bins for overview
    if result is None:
        return
        
    features = result['features'] # (Time x 2*Channels)
    num_feats = features.shape[1]
    num_channels = num_feats // 2
    sbp_indices = np.arange(num_channels, num_feats)
    
    # Extract just the Power (Energy)
    baseline_energy = features[:, sbp_indices] # (Time x Channels)
    mean_energy_map = np.mean(baseline_energy, axis=0) # (Channels,)
    
    # 2. Setup Spatial Grid (Virtual Brain)
    # Assume 10x10 grid (approx 96 channels)
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    
    # Map channels to grid (random mapping for now, or sequental)
    # We'll create a Prion Load Map (0 = Healthy, 1 = Dead)
    prion_load = np.zeros((grid_size, grid_size))
    
    # Seed Infection
    # Start at a random location (e.g., center)
    seed_x, seed_y = 5, 5
    prion_load[seed_x, seed_y] = 0.1
    
    # Simulation Parameters
    timesteps = 20
    diffusion_rate = 0.5
    replication_rate = 0.2
    
    # Store history for visualization
    history = []
    
    print("Simulating Prion Spread...")
    
    for t in range(timesteps):
        # 3. Simulate Prion Dynamics
        # A. Diffusion (Spread to neighbors)
        # Simple Gaussian blur approximates diffusion
        spread = gaussian_filter(prion_load, sigma=0.5) * diffusion_rate
        
        # B. Replication (Self-amplification)
        # dP/dt = r * P * (1-P) (Logistic growth locally)
        growth = replication_rate * prion_load * (1 - prion_load)
        
        prion_load = prion_load + growth + spread
        
        # Clip to max death (1.0)
        prion_load = np.clip(prion_load, 0, 1.0)
        
        # C. Apply to Neural Activity
        # Health = 1 - Prion Load
        # We model this by picking a representative frame of data and darkening it
        # Channel i maps to some x,y.
        # For viz, we'll just visualize the Prion Map effectively acting as a mask.
        
        history.append(prion_load.copy())
    
    # 4. Visualization
    # Series of 4 snapshots (Early, Mid, Late, Terminal)
    indices = [0, int(timesteps*0.33), int(timesteps*0.66), timesteps-1]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for ax, idx in zip(axes, indices):
        stage_load = history[idx]
        health_map = 1.0 - stage_load
        
        # Plot "Neural Health"
        # Green = Healthy, Black = Dead
        im = ax.imshow(health_map, cmap='YlGn_r', vmin=0, vmax=1)
        
        ax.set_title(f"Time Step {idx}\nAvg Health: {np.mean(health_map):.2f}")
        ax.axis('off')
        
    plt.suptitle("The Spreading Silence: ALS/Prion Progression Simulation", fontsize=16)
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Neural Function Remaining')
    
    save_path = "graphs/als_prion_simulation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Simulation saved to {save_path}")

if __name__ == "__main__":
    simulate_als()
