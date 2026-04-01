#!/usr/bin/env python3
"""
Map the "Health" (Neural Activity) of the Electrode Array.
Visualizes which channels are active (Healthy) and which are silent (Dead/Background).
Helps identify targets for simulated ALS progression.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from memorymapping.processor import NeuroDataProcessor

def map_health():
    os.makedirs("graphs", exist_ok=True)
    ns6_file = "array_Sc2.ns6"
    processor = NeuroDataProcessor()
    
    if not os.path.exists(ns6_file):
        print(f"File {ns6_file} not found.")
        return

    print("Extracting features to assess health...")
    # Use a large bin size just to get the total energy efficiently
    result = processor.extract_features(ns6_file, bin_size_ms=100.0) 
    if result is None:
        return
        
    features = result['features'] # (Time x 2*Channels)
    num_feats = features.shape[1]
    num_channels = num_feats // 2
    
    # 1. Metric: Total Spike Band Power (Use the second half of features)
    sbp_indices = np.arange(num_channels, num_feats)
    sbp_data = features[:, sbp_indices]
    
    # "Health" = Mean Power log-scaled (to handle dynamic range)
    # Adding small epsilon to avoid log(0)
    channel_health = np.log1p(np.mean(sbp_data, axis=0))
    
    print(f"Analyzed {num_channels} channels.")
    print(f"Max Health: {np.max(channel_health):.2f}, Min Health: {np.min(channel_health):.2f}")
    
    # 2. Map to 10x10 Grid (Virtual Layout)
    # Standard Utah Array often has 96 channels in 10x10 with corners missing.
    grid = np.nan * np.ones((10, 10)) # Initialize with NaNs
    
    # Mapping Strategy: Fill row by row, skipping corners
    # Corners are at indices: 0, 9, 90, 99 (in 0-99 flattened)
    # (0,0), (0,9), (9,0), (9,9)
    
    ch_idx = 0
    for r in range(10):
        for c in range(10):
            # Check for corners
            if (r==0 and c==0) or (r==0 and c==9) or (r==9 and c==0) or (r==9 and c==9):
                continue
            
            if ch_idx < num_channels:
                grid[r, c] = channel_health[ch_idx]
                ch_idx += 1
                
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Heatmap
    # cmap 'inferno' or 'magma' is good for "Energy/Life"
    current_cmap = plt.cm.get_cmap('magma').copy()
    current_cmap.set_bad(color='black') # Corners are black (Void)
    
    im = ax.imshow(grid, cmap=current_cmap)
    
    # Annotations
    # Label each active square with its channel number? Maybe too crowded.
    # Label grid coords.
    
    plt.colorbar(im, label='Neural Health (Log Mean Power)')
    ax.set_title(f"Electrode Array Health Map\n(Bright = Healthy Target for ALS, Dark = Silent/Dead)", fontsize=14)
    ax.set_xlabel("Electrode Column")
    ax.set_ylabel("Electrode Row")
    
    save_path = "graphs/array_health_map.png"
    plt.savefig(save_path, dpi=300)
    print(f"Health Map saved to {save_path}")

if __name__ == "__main__":
    map_health()
