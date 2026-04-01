#!/usr/bin/env python3
"""
Visualize energy spikes and minimization points from Octopus Brain system
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys

def load_energy_data(json_file):
    """Load energy data exported from the Elixir system"""
    with open(json_file, 'r') as f:
        return json.load(f)

def visualize_energy_patterns(energy_data):
    """Create comprehensive visualization of energy patterns"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Octopus Brain Energy Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    history = energy_data.get('energy_history', [])
    spikes = energy_data.get('spike_events', [])
    minimizations = energy_data.get('minimization_events', [])
    
    if not history:
        print("No energy history data found!")
        return
    
    timestamps = [h['timestamp'] for h in history]
    energies = [h['energy'] for h in history]
    
    # Normalize timestamps to seconds from start
    start_time = min(timestamps)
    times_sec = [(t - start_time) / 1000.0 for t in timestamps]
    
    # 1. Energy Over Time with Spikes and Minimization highlighted
    ax1 = axes[0, 0]
    ax1.plot(times_sec, energies, 'b-', linewidth=2, label='Energy Level')
    
    # Highlight spikes
    if spikes:
        spike_times = [(s['timestamp'] - start_time) / 1000.0 for s in spikes]
        spike_energies = [s['energy_level'] for s in spikes]
        ax1.scatter(spike_times, spike_energies, color='red', s=100, 
                   marker='^', label='Energy Spikes', zorder=5)
    
    # Highlight minimization periods
    if minimizations:
        min_times = [(m['timestamp'] - start_time) / 1000.0 for m in minimizations]
        min_energies = [m['energy_level'] for m in minimizations]
        ax1.scatter(min_times, min_energies, color='green', s=100, 
                   marker='v', label='Minimization Points', zorder=5)
    
    ax1.axhline(y=energy_data.get('spike_threshold', 100), 
               color='r', linestyle='--', alpha=0.5, label='Spike Threshold')
    ax1.axhline(y=energy_data.get('min_threshold', 10), 
               color='g', linestyle='--', alpha=0.5, label='Min Threshold')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Energy Level')
    ax1.set_title('Energy Timeline with Spike Detection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy Distribution Histogram
    ax2 = axes[0, 1]
    ax2.hist(energies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(energies), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(energies):.1f}')
    ax2.axvline(x=np.median(energies), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(energies):.1f}')
    ax2.set_xlabel('Energy Level')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Energy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Event Type Breakdown
    ax3 = axes[1, 0]
    event_breakdown = energy_data.get('metrics', {}).get('events_by_type', {})
    if event_breakdown:
        event_types = list(event_breakdown.keys())
        event_counts = list(event_breakdown.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(event_types)))
        bars = ax3.bar(range(len(event_types)), event_counts, color=colors)
        ax3.set_xticks(range(len(event_types)))
        ax3.set_xticklabels(event_types, rotation=45, ha='right')
        ax3.set_ylabel('Count')
        ax3.set_title('Event Type Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 4. Efficiency Metrics
    ax4 = axes[1, 1]
    metrics = energy_data.get('metrics', {})
    
    metric_names = ['Total Energy', 'Avg Energy', 'Spike Count', 'Min Count']
    metric_values = [
        metrics.get('total_energy_used', 0) / 100,  # Scale down for visibility
        metrics.get('average_energy', 0),
        len(spikes),
        len(minimizations)
    ]
    
    bars = ax4.barh(metric_names, metric_values, color=['orange', 'blue', 'red', 'green'])
    ax4.set_xlabel('Value')
    ax4.set_title('System Metrics')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax4.text(val, i, f' {val:.1f}', va='center')
    
    # 5. Energy Rate of Change (derivative)
    ax5 = axes[2, 0]
    if len(energies) > 1:
        energy_diff = np.diff(energies)
        time_diff = np.diff(times_sec)
        energy_rate = energy_diff / time_diff
        
        ax5.plot(times_sec[1:], energy_rate, 'purple', linewidth=1.5)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.fill_between(times_sec[1:], 0, energy_rate, 
                        where=(energy_rate > 0), alpha=0.3, color='red', 
                        label='Energy Increasing')
        ax5.fill_between(times_sec[1:], 0, energy_rate, 
                        where=(energy_rate <= 0), alpha=0.3, color='green',
                        label='Energy Decreasing')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Energy Rate of Change')
        ax5.set_title('Energy Dynamics (Rate of Change)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Efficiency Score Visualization
    ax6 = axes[2, 1]
    
    # Calculate efficiency score
    spike_count = len(spikes)
    min_count = len(minimizations)
    total_events = spike_count + min_count
    
    if total_events > 0:
        efficiency_score = (min_count / total_events) * 100
    else:
        efficiency_score = 50.0
    
    # Create pie chart
    if total_events > 0:
        sizes = [spike_count, min_count]
        labels = [f'Spikes\n({spike_count})', f'Minimizations\n({min_count})']
        colors_pie = ['#ff6b6b', '#51cf66']
        explode = (0.1, 0)  # explode spike slice
        
        ax6.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax6.set_title(f'System Efficiency Score: {efficiency_score:.1f}%')
    else:
        ax6.text(0.5, 0.5, 'No Events Detected', 
                ha='center', va='center', fontsize=14)
        ax6.set_title('System Efficiency')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"graphs/energy_analysis_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    # Print analysis summary
    print("\n" + "="*60)
    print("ENERGY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Events Recorded: {metrics.get('event_count', 0)}")
    print(f"Total Energy Used: {metrics.get('total_energy_used', 0):.2f} units")
    print(f"Average Energy Level: {metrics.get('average_energy', 0):.2f} units")
    print(f"Energy Range: {min(energies):.2f} - {max(energies):.2f} units")
    print(f"\nSpikes Detected: {spike_count}")
    print(f"Minimization Periods: {min_count}")
    print(f"Efficiency Score: {efficiency_score:.2f}%")
    print(f"\nMonitoring Duration: {max(times_sec):.2f} seconds")
    print("="*60)
    
    return efficiency_score

def generate_sample_data():
    """Generate sample energy data for testing"""
    import random
    
    start_time = 1000000
    history = []
    current_energy = 5.0
    
    # Simulate 100 events over time
    for i in range(100):
        timestamp = start_time + (i * 1000)  # Events every second
        
        # Random walk with occasional spikes
        if random.random() < 0.1:  # 10% chance of spike
            current_energy += random.uniform(30, 70)
        elif random.random() < 0.2:  # 20% chance of minimization
            current_energy = max(2, current_energy - random.uniform(10, 20))
        else:
            current_energy += random.uniform(-5, 10)
        
        current_energy = max(0, current_energy)  # Keep non-negative
        
        history.append({
            'timestamp': timestamp,
            'energy': current_energy,
            'event_type': random.choice(['reflex', 'autonomous_action', 'coordination']),
            'details': {}
        })
    
    # Detect spikes and minimizations
    spikes = [
        {
            'timestamp': h['timestamp'],
            'energy_level': h['energy'],
            'event_type': h['event_type']
        }
        for h in history if h['energy'] > 100
    ]
    
    minimizations = [
        {
            'timestamp': h['timestamp'],
            'energy_level': h['energy'],
            'efficiency': 'high'
        }
        for h in history if h['energy'] < 10
    ]
    
    return {
        'energy_history': history,
        'spike_events': spikes,
        'minimization_events': minimizations,
        'spike_threshold': 100,
        'min_threshold': 10,
        'metrics': {
            'total_energy_used': sum(h['energy'] for h in history),
            'average_energy': np.mean([h['energy'] for h in history]),
            'event_count': len(history),
            'events_by_type': {
                'reflex': 30,
                'autonomous_action': 45,
                'coordination': 20,
                'synchronization': 5
            }
        }
    }

if __name__ == "__main__":
    import os
    os.makedirs("graphs", exist_ok=True)
    
    if len(sys.argv) > 1:
        # Load from file
        data_file = sys.argv[1]
        print(f"Loading energy data from: {data_file}")
        energy_data = load_energy_data(data_file)
    else:
        # Use sample data
        print("No data file provided, generating sample data...")
        energy_data = generate_sample_data()
        
        # Save sample data for reference
        with open('data/sample_energy_data.json', 'w') as f:
            json.dump(energy_data, f, indent=2)
        print("Sample data saved to: data/sample_energy_data.json")
    
    visualize_energy_patterns(energy_data)
