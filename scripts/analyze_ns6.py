#!/usr/bin/env python3
"""
Analyze Blackrock NS6 neural data (raw analog to spikes).
"""

import sys
import os
from memorymapping.processor import NeuroDataProcessor

def analyze_ns6():
    # File path
    ns6_file = "array_Sc2.ns6"
    
    if not os.path.exists(ns6_file):
        print(f"File {ns6_file} not found.")
        return
        
    print(f"Analyzing {ns6_file}...")
    
    processor = NeuroDataProcessor()
    
    # Load and Detect Spikes
    # Using 4.0 std dev threshold
    success = processor.load_ns6_data(ns6_file, threshold_std=4.0)
    
    if success:
        print("Analysis complete. Generating plots...")
        output = processor.analyze_spike_patterns()
        print(f"Visualization saved to {output}")
    else:
        print("Failed to process data.")

if __name__ == "__main__":
    analyze_ns6()
