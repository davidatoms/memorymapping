#!/usr/bin/env python3
"""
Extract and Visualize Neural Features mimicking NEJM Brain-to-Text.
Features: 
1. Threshold Crossings (TC)
2. Spike Band Power (SBP)
Binned at 20ms.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from memorymapping.processor import NeuroDataProcessor

def run_extraction():
    os.makedirs("graphs", exist_ok=True)
    ns6_file = "array_Sc2.ns6"
    
    processor = NeuroDataProcessor()
    
    if not os.path.exists(ns6_file):
        print(f"File {ns6_file} not found. Cannot extract features.")
        return

    # Extract Features
    # Result is a dict with 'features' (Time x Channels*2)
    result = processor.extract_features(ns6_file, bin_size_ms=20.0)
    
    if result is None:
        print("Extraction failed.")
        return
        
    features = result['features']
    times = result['bin_times']
    
    print(f"Feature Matrix Shape: {features.shape}")
    
    # Visualization: Heatmap of Neural Activity
    # Combine TC and SBP or show separately?
    # Let's show them side-by-side or stacked.
    
    num_bins, num_feats = features.shape
    num_channels = num_feats // 2
    
    tc_feats = features[:, :num_channels]
    sbp_feats = features[:, num_channels:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 1. Threshold Crossings (The "Spikes")
    # Transpose for (Channels x Time)
    im1 = ax1.imshow(tc_feats.T, aspect='auto', cmap='Greys', interpolation='nearest', origin='lower')
    ax1.set_title("Feature 1: Threshold Crossings (TC)")
    ax1.set_ylabel("Channel Index")
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # 2. Spike Band Power (The "Energy")
    # Log scale is often better for power
    im2 = ax2.imshow(np.log1p(sbp_feats.T), aspect='auto', cmap='inferno', interpolation='nearest', origin='lower')
    ax2.set_title("Feature 2: Spike Band Power (SBP) [Log Scale]")
    ax2.set_xlabel("Time Bin (20ms)")
    ax2.set_ylabel("Channel Index")
    plt.colorbar(im2, ax=ax2, label='Log Power')
    
    plt.tight_layout()
    save_path = "graphs/nejm_features.png"
    plt.savefig(save_path, dpi=300)
    print(f"Feature visualization saved to {save_path}")
    
    # Also save the raw matrix for inspections
    np.save("graphs/features.npy", features)
    print("Saved raw features to graphs/features.npy")

if __name__ == "__main__":
    run_extraction()
