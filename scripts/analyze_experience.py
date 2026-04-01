#!/usr/bin/env python3
"""
Analyze the experiential context captured during neural simulation
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_experience():
    print("ANALYZING SIMULATED EXPERIENCE")
    print("=" * 50)
    
    # Load the neural simulation data
    # Load the neural simulation data
    with open('data/sample_neuro_data.json', 'r') as f:
        data = json.load(f)
    
    contexts = data['environmental_context']
    spike_events = data['spike_events']
    
    print(f"Total experience samples: {len(contexts)}")
    print(f"Total neural spikes: {len(spike_events)}")
    
    # Analyze the complete sensory experience
    print("\nEXPERIENCE TIMELINE:")
    print("-" * 30)
    
    experiences = {}
    for i, context in enumerate(contexts):
        stimulus = context['stimulus_type']
        if stimulus not in experiences:
            experiences[stimulus] = []
        experiences[stimulus].append(context)
    
    for stimulus_type, contexts_list in experiences.items():
        print(f"\n{stimulus_type.upper()}:")
        print(f"  Duration: {len(contexts_list)} time samples")
        
        # Calculate averages for this experience
        avg_attention = np.mean([c['attention_level'] for c in contexts_list])
        avg_arousal = np.mean([c['arousal_level'] for c in contexts_list])
        avg_difficulty = np.mean([c['task_difficulty'] for c in contexts_list])
        
        print(f"  Average attention level: {avg_attention:.2f}")
        print(f"  Average arousal level: {avg_arousal:.2f}")
        print(f"  Average task difficulty: {avg_difficulty:.2f}")
        
        # Show sensory input profile
        sensory_profile = {}
        for context in contexts_list:
            for sense, value in context['sensory_inputs'].items():
                if sense not in sensory_profile:
                    sensory_profile[sense] = []
                sensory_profile[sense].append(value)
        
        print(f"  Sensory experience:")
        for sense, values in sensory_profile.items():
            avg_val = np.mean(values)
            print(f"    {sense}: {avg_val:.3f} (range: {min(values):.3f}-{max(values):.3f})")
        
        # Show motor response
        motor_profile = {}
        for context in contexts_list:
            for motor, value in context['motor_outputs'].items():
                if motor not in motor_profile:
                    motor_profile[motor] = []
                motor_profile[motor].append(value)
        
        print(f"  Motor responses:")
        for motor, values in motor_profile.items():
            avg_val = np.mean(values)
            print(f"    {motor}: {avg_val:.3f} (range: {min(values):.3f}-{max(values):.3f})")
    
    # Detailed moment-by-moment experience analysis
    print(f"\nMOMENT-BY-MOMENT EXPERIENCE SAMPLES:")
    print("-" * 40)
    
    sample_moments = [0, 25, 50, 75, 100, 125]  # Sample different time points
    
    for moment in sample_moments:
        if moment < len(contexts):
            ctx = contexts[moment]
            print(f"\nTime {ctx['timestamp_ms']:.1f}ms - {ctx['stimulus_type']}:")
            print(f"  Person is {ctx['brain_state']} with {ctx['attention_level']:.1%} attention")
            print(f"  Arousal: {ctx['arousal_level']:.1%}, Task difficulty: {ctx['task_difficulty']:.1%}")
            print(f"  What they sense:")
            for sense, intensity in ctx['sensory_inputs'].items():
                if intensity > 0.1:  # Only show significant inputs
                    print(f"    {sense}: {intensity:.1%} intensity")
            print(f"  How they respond:")
            for motor, value in ctx['motor_outputs'].items():
                if abs(value) > 0.1:  # Only show significant movements
                    direction = "right" if value > 0 else "left" if motor.endswith('_x') else ("up" if value > 0 else "down")
                    print(f"    {motor}: {abs(value):.1%} {direction}")
    
    # Create visualization
    create_experience_visualization(contexts)
    
    # Neural response analysis
    print(f"\nNEURAL RESPONSE TO EXPERIENCE:")
    print("-" * 35)
    
    # Group spikes by time windows to correlate with experiences
    spike_times = [spike['timestamp_ms'] for spike in spike_events]
    spike_types = [spike['spike_type'] for spike in spike_events]
    
    for stimulus_type, contexts_list in experiences.items():
        # Find spikes that occurred during this stimulus
        stimulus_start = contexts_list[0]['timestamp_ms']
        stimulus_end = contexts_list[-1]['timestamp_ms']
        
        stimulus_spikes = [spike for spike in spike_events 
                          if stimulus_start <= spike['timestamp_ms'] <= stimulus_end]
        
        if stimulus_spikes:
            spike_type_counts = {}
            for spike in stimulus_spikes:
                spike_type = spike['spike_type']
                spike_type_counts[spike_type] = spike_type_counts.get(spike_type, 0) + 1
            
            print(f"\n{stimulus_type}: {len(stimulus_spikes)} neural spikes")
            for spike_type, count in sorted(spike_type_counts.items()):
                percentage = (count / len(stimulus_spikes)) * 100
                print(f"  {spike_type}: {count} ({percentage:.1f}%)")

def create_experience_visualization(contexts):
    """Create visualizations of the experiential data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    times = [ctx['timestamp_ms'] for ctx in contexts]
    attention = [ctx['attention_level'] for ctx in contexts]
    arousal = [ctx['arousal_level'] for ctx in contexts]
    difficulty = [ctx['task_difficulty'] for ctx in contexts]
    
    # Plot 1: Mental state over time
    axes[0, 0].plot(times, attention, 'b-', label='Attention', linewidth=2)
    axes[0, 0].plot(times, arousal, 'r-', label='Arousal', linewidth=2)
    axes[0, 0].plot(times, difficulty, 'g-', label='Task Difficulty', linewidth=2)
    axes[0, 0].set_title('Mental State Over Time')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Level (0-1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sensory experience heatmap
    sensory_types = ['visual', 'auditory', 'tactile', 'olfactory', 'gustatory']
    sensory_matrix = []
    
    for ctx in contexts:
        sensory_row = []
        for sense in sensory_types:
            sensory_row.append(ctx['sensory_inputs'].get(sense, 0))
        sensory_matrix.append(sensory_row)
    
    sensory_matrix = np.array(sensory_matrix).T
    im1 = axes[0, 1].imshow(sensory_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('Sensory Experience Over Time')
    axes[0, 1].set_xlabel('Time Samples')
    axes[0, 1].set_ylabel('Sensory Modality')
    axes[0, 1].set_yticks(range(len(sensory_types)))
    axes[0, 1].set_yticklabels(sensory_types)
    plt.colorbar(im1, ax=axes[0, 1], label='Intensity')
    
    # Plot 3: Motor responses
    movement_x = [ctx['motor_outputs'].get('movement_x', 0) for ctx in contexts]
    movement_y = [ctx['motor_outputs'].get('movement_y', 0) for ctx in contexts]
    head_turn = [ctx['motor_outputs'].get('head_turn', 0) for ctx in contexts]
    
    axes[1, 0].plot(times, movement_x, 'b-', label='Movement X', linewidth=2)
    axes[1, 0].plot(times, movement_y, 'r-', label='Movement Y', linewidth=2)
    axes[1, 0].plot(times, head_turn, 'g-', label='Head Turn', linewidth=2)
    axes[1, 0].set_title('Motor Responses Over Time')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Movement Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Experience intensity distribution
    stimulus_types = list(set([ctx['stimulus_type'] for ctx in contexts]))
    intensity_data = []
    labels = []
    
    for stimulus in stimulus_types:
        intensities = [ctx['stimulus_intensity'] for ctx in contexts 
                      if ctx['stimulus_type'] == stimulus]
        if intensities:
            intensity_data.append(intensities)
            labels.append(stimulus.replace('_', '\n'))
    
    axes[1, 1].boxplot(intensity_data, labels=labels)
    axes[1, 1].set_title('Stimulus Intensity Distribution')
    axes[1, 1].set_ylabel('Intensity Level')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"graphs/experience_analysis_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nExperience visualization saved to: {save_path}")

if __name__ == "__main__":
    analyze_experience() 