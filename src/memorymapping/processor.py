import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
try:
    import neo
except ImportError:
    neo = None

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

# Create output directory
os.makedirs("graphs", exist_ok=True)

class SpikeType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    SENSORY = "sensory"
    MOTOR = "motor"
    MEMORY = "memory"

class NeuronType(Enum):
    PYRAMIDAL = "pyramidal"
    INTERNEURON = "interneuron"
    PURKINJE = "purkinje"
    GRANULE = "granule"

@dataclass
class SpikeEvent:
    """Individual spike event from Rust simulator"""
    neuron_id: int
    timestamp: float  # ms
    spike_type: SpikeType
    amplitude: float  # mV
    position: Tuple[float, float, float]  # x, y, z coordinates
    neuron_type: NeuronType
    
@dataclass
class EnvironmentalContext:
    """Environmental/stimulus context when spike occurred"""
    stimulus_type: str
    stimulus_intensity: float
    sensory_input: Dict[str, float]  # e.g., {"visual": 0.8, "auditory": 0.2}
    motor_output: Dict[str, float]   # e.g., {"movement_x": 0.1, "movement_y": -0.3}
    brain_state: str  # e.g., "awake", "rem_sleep", "learning"
    attention_level: float

class NeuroDataProcessor:
    """Process neural spike data from Rust neuro-sim"""
    
    def __init__(self):
        self.spike_events: List[SpikeEvent] = []
        self.environmental_context: List[EnvironmentalContext] = []
        self.spatial_bounds = {"x": (-10, 10), "y": (-10, 10), "z": (-5, 5)}
        
    def load_rust_spike_data(self, json_file: str) -> bool:
        """
        Load spike data from Rust simulator JSON format
        Expected format from neuro-sim-rust:
        {
            "simulation_metadata": {
                "duration_ms": 5000,
                "timestep_ms": 0.1,
                "brain_region": "hippocampus"
            },
            "spike_events": [
                {
                    "neuron_id": 12345,
                    "timestamp_ms": 1234.5,
                    "spike_type": "excitatory",
                    "amplitude_mv": 70.2,
                    "position_xyz": [2.1, -1.4, 0.8],
                    "neuron_type": "pyramidal",
                    "synaptic_weights": [0.5, 0.8, 0.3],
                    "membrane_potential": -55.0
                }
            ],
            "environmental_context": [
                {
                    "timestamp_ms": 1234.0,
                    "stimulus_type": "visual_pattern",
                    "stimulus_intensity": 0.8,
                    "sensory_inputs": {"visual": 0.9, "auditory": 0.1},
                    "motor_outputs": {"movement_x": 0.0, "movement_y": 0.2},
                    "brain_state": "learning",
                    "attention_level": 0.7
                }
            ]
        }
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Load spike events
            for spike_data in data.get("spike_events", []):
                spike = SpikeEvent(
                    neuron_id=spike_data["neuron_id"],
                    timestamp=spike_data["timestamp_ms"],
                    spike_type=SpikeType(spike_data["spike_type"]),
                    amplitude=spike_data["amplitude_mv"],
                    position=tuple(spike_data["position_xyz"]),
                    neuron_type=NeuronType(spike_data["neuron_type"])
                )
                self.spike_events.append(spike)
            
            # Load environmental context
            for context_data in data.get("environmental_context", []):
                context = EnvironmentalContext(
                    stimulus_type=context_data["stimulus_type"],
                    stimulus_intensity=context_data["stimulus_intensity"],
                    sensory_input=context_data["sensory_inputs"],
                    motor_output=context_data["motor_outputs"],
                    brain_state=context_data["brain_state"],
                    attention_level=context_data["attention_level"]
                )
                self.environmental_context.append(context)
            
            print(f"Loaded {len(self.spike_events)} spike events")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def load_ns6_data(self, file_path: str, threshold_std: float = 4.0) -> bool:
        """
        Load Blackrock NS6 file (raw analog data) and detect spikes.
        
        Args:
            file_path: Path to .ns6 file
            threshold_std: Threshold for spike detection (std devs below mean)
        """
        if neo is None:
            print("Error: 'neo' library is not installed. Please install it to read .ns6 files.")
            return False
            
        print(f"Loading {file_path}...")
        try:
            reader = neo.io.BlackrockIO(filename=file_path)
            block = reader.read_block()
            
            print(f"Found {len(block.segments)} segments in file.")
            
            # Process discrete segments
            for seg in block.segments:
                print(f"Processing segment with {len(seg.analogsignals)} analog signals.")
                
                # Iterate through analog signals (electrodes)
                for i, signal in enumerate(seg.analogsignals):
                    # Signal is (Time x Channels)
                    sig_data = signal.magnitude
                    sampling_rate = float(signal.sampling_rate) # Hz
                    
                    # If multiple channels, iterate
                    num_channels = sig_data.shape[1]
                    for ch in range(num_channels):
                        channel_data = sig_data[:, ch]
                        
                        # Thresholding
                        mean = np.mean(channel_data)
                        std = np.std(channel_data)
                        threshold = mean - (threshold_std * std)
                        
                        # Find cross-threshold events (simple detection)
                        # We look for falling edges crossing threshold
                        crossings = np.where((channel_data[:-1] > threshold) & (channel_data[1:] <= threshold))[0]
                        
                        # Filter duplicates (refractory period ~1ms)
                        refractory_samples = int(1e-3 * sampling_rate)
                        valid_crossings = []
                        last = -refractory_samples
                        
                        for c in crossings:
                            if c - last > refractory_samples:
                                valid_crossings.append(c)
                                last = c
                        
                        # Create SpikeEvents
                        for c in valid_crossings:
                            # Convert sample index to ms
                            t_sec = float(signal.t_start) + (c / sampling_rate)
                            t_ms = t_sec * 1000.0
                            
                            # Amplitude (uV)
                            amp = channel_data[c]
                            
                            spike = SpikeEvent(
                                neuron_id=f"ch_{ch}", # Identify by channel
                                timestamp=t_ms,
                                spike_type=SpikeType.EXCITATORY, # Assumption
                                amplitude=amp,
                                position=(np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)), # Random spatial mapping for now
                                neuron_type=NeuronType.PYRAMIDAL
                            )
                            self.spike_events.append(spike)
                            
            print(f"Loaded {len(self.spike_events)} extracted spikes from NS6.")
            return True
            
        except Exception as e:
            print(f"Error loading NS6 file: {e}")
            return False

    def extract_features(self, file_path: str, bin_size_ms: float = 20.0) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract neural features (Threshold Crossings & Spike Band Power) 
        mimicking the NEJM Brain-to-Text pipeline.
        
        Args:
            file_path: Path to .ns6 file
            bin_size_ms: Size of time bins in milliseconds (default 20ms)
            
        Returns:
            Dictionary containing:
                'features': Matrix of shape (Num_Bins, Num_Channels * 2)
                'bin_times': Array of time centers for each bin
                'channel_names': List of channel names
        """
        if neo is None:
            print("Error: 'neo' library is not installed.")
            return None
        if scipy_signal is None:
            print("Error: 'scipy' library is not installed.")
            return None
            
        print(f"Extracting features from {file_path}...")
        try:
            reader = neo.io.BlackrockIO(filename=file_path)
            block = reader.read_block()
            
            # Use the first segment (assuming continuous recording)
            seg = block.segments[0]
            if len(seg.analogsignals) == 0:
                print("No analog signals found.")
                return None
                
            analog_signal = seg.analogsignals[0] # (Time x Channels)
            sampling_rate = float(analog_signal.sampling_rate)
            raw_data = analog_signal.magnitude # (Samples x Channels) unit: uV
            num_samples, num_channels = raw_data.shape
            
            print(f"Data shape: {num_samples} samples x {num_channels} channels @ {sampling_rate} Hz")
            
            # Design Filter: Bandpass 250 Hz - 5000 Hz (4th order Butterworth)
            # This isolates the spike band, removing LFP and high-frequency noise.
            nyquist = 0.5 * sampling_rate
            low = 250.0 / nyquist
            high = 5000.0 / nyquist
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            
            # Prepare Output Containers
            samples_per_bin = int((bin_size_ms / 1000.0) * sampling_rate)
            num_bins = num_samples // samples_per_bin
            
            # Matrix: (Bins, Channels*2). 
            # Ordering: [Ch1_TC, Ch2_TC, ..., Ch1_SBP, Ch2_SBP, ...] OR [Ch1_TC, Ch1_SBP, Ch2_TC, ...]
            # NEJM repo order: [All TCs] then [All SBPs]
            # Features 0-63: TC, 64-127: SBP (for 64 ch)
            
            tc_matrix = np.zeros((num_bins, num_channels))
            sbp_matrix = np.zeros((num_bins, num_channels))
            
            print(f"Processing channels (Filter -> Bin -> Extract)...")
            
            for ch in range(num_channels):
                # 1. Filter
                # Use filtfilt for zero-phase filtering
                filtered_signal = scipy_signal.filtfilt(b, a, raw_data[:, ch])
                
                # 2. Calculate Threshold
                # NEJM uses -4.5 * RMS of the filtered signal
                rms = np.sqrt(np.mean(filtered_signal**2))
                threshold = -4.5 * rms
                
                # 3. Binning Loop (Vectorized where possible)
                # Reshape signal into (Num_Bins, Samples_Per_Bin)
                # Truncate leftovers
                trunc_len = num_bins * samples_per_bin
                sig_binned = filtered_signal[:trunc_len].reshape(num_bins, samples_per_bin)
                
                # Feature 1: Threshold Crossings (TC)
                # Count falling edge crossings of threshold
                # Simple approximation: count points below threshold? No, strict crossing.
                # Vectorized crossing check within bins is tricky. 
                # Let's count samples <= threshold. For spikes this is roughly proportional to crossings.
                # OR do strict crossing check on full array then bin sum.
                
                # Strict Crossing on full array
                is_below = filtered_signal[:trunc_len] < threshold
                # Falling edges: was above, now below
                # shift right
                was_above = np.roll(filtered_signal[:trunc_len], 1) >= threshold
                crossings = (is_below & was_above).astype(int)
                
                # Sum crossings per bin
                tc_matrix[:, ch] = crossings.reshape(num_bins, samples_per_bin).sum(axis=1)
                
                # Feature 2: Spike Band Power (SBP)
                # Mean of squared filtered signal in bin
                sbp_matrix[:, ch] = np.mean(sig_binned**2, axis=1)
                
            # Concatenate features: [TCs, SBPs]
            features = np.hstack([tc_matrix, sbp_matrix])
            
            bin_times = np.arange(num_bins) * (bin_size_ms / 1000.0) + float(analog_signal.t_start)
            
            print(f"Extraction complete. Feature Matrix: {features.shape}")
            return {
                'features': features,
                'bin_times': bin_times,
                'channel_names': [f"channel_{i+1}" for i in range(num_channels)]
            }
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        print("Creating sample neural data...")
        
        # Generate sample spike events
        np.random.seed(42)
        for i in range(1000):
            spike = SpikeEvent(
                neuron_id=np.random.randint(1, 500),
                timestamp=np.random.uniform(0, 5000),  # 5 seconds
                spike_type=np.random.choice(list(SpikeType)),
                amplitude=np.random.normal(70, 10),
                position=(
                    np.random.uniform(-8, 8),
                    np.random.uniform(-8, 8),
                    np.random.uniform(-3, 3)
                ),
                neuron_type=np.random.choice(list(NeuronType))
            )
            self.spike_events.append(spike)
        
        # Generate environmental context
        for i in range(100):
            context = EnvironmentalContext(
                stimulus_type=np.random.choice(["visual_pattern", "auditory_tone", "tactile_touch"]),
                stimulus_intensity=np.random.uniform(0.1, 1.0),
                sensory_input={
                    "visual": np.random.uniform(0, 1),
                    "auditory": np.random.uniform(0, 1),
                    "tactile": np.random.uniform(0, 1)
                },
                motor_output={
                    "movement_x": np.random.uniform(-1, 1),
                    "movement_y": np.random.uniform(-1, 1)
                },
                brain_state=np.random.choice(["awake", "learning", "rem_sleep"]),
                attention_level=np.random.uniform(0.2, 1.0)
            )
            self.environmental_context.append(context)
    
    def analyze_spike_patterns(self):
        """Analyze different types of spike patterns"""
        if not self.spike_events:
            print("No spike data loaded!")
            return
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D Spatial Distribution by Spike Type
        ax1 = fig.add_subplot(331, projection='3d')
        spike_types = list(set(spike.spike_type for spike in self.spike_events))
        colors = plt.cm.tab10(np.linspace(0, 1, len(spike_types)))
        
        for spike_type, color in zip(spike_types, colors):
            spikes_of_type = [s for s in self.spike_events if s.spike_type == spike_type]
            if spikes_of_type:
                x = [s.position[0] for s in spikes_of_type]
                y = [s.position[1] for s in spikes_of_type]
                z = [s.position[2] for s in spikes_of_type]
                ax1.scatter(x, y, z, c=[color], label=spike_type.value, alpha=0.6, s=20)
        
        ax1.set_xlabel('X Position (mm)')
        ax1.set_ylabel('Y Position (mm)')
        ax1.set_zlabel('Z Position (mm)')
        ax1.set_title('3D Spatial Distribution by Spike Type')
        ax1.legend()
        
        # 2. Temporal Spike Patterns
        ax2 = fig.add_subplot(332)
        timestamps = [s.timestamp for s in self.spike_events]
        spike_type_nums = [list(SpikeType).index(s.spike_type) for s in self.spike_events]
        
        scatter = ax2.scatter(timestamps, spike_type_nums, 
                            c=[s.amplitude for s in self.spike_events],
                            cmap='viridis', alpha=0.6, s=10)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Spike Type')
        ax2.set_title('Temporal Spike Pattern (colored by amplitude)')
        ax2.set_yticks(range(len(SpikeType)))
        ax2.set_yticklabels([st.value for st in SpikeType], rotation=45)
        plt.colorbar(scatter, ax=ax2, label='Amplitude (mV)')
        
        # 3. Spike Rate Over Time
        ax3 = fig.add_subplot(333)
        time_bins = np.linspace(0, max(timestamps), 50)
        spike_counts, _ = np.histogram(timestamps, bins=time_bins)
        spike_rate = spike_counts / (time_bins[1] - time_bins[0]) * 1000  # Hz
        
        ax3.plot(time_bins[:-1], spike_rate, linewidth=2)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Spike Rate (Hz)')
        ax3.set_title('Population Spike Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. Neuron Type Distribution
        ax4 = fig.add_subplot(334)
        neuron_types = [s.neuron_type.value for s in self.spike_events]
        unique_types, counts = np.unique(neuron_types, return_counts=True)
        bars = ax4.bar(unique_types, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_types))))
        ax4.set_xlabel('Neuron Type')
        ax4.set_ylabel('Spike Count')
        ax4.set_title('Spike Distribution by Neuron Type')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # 5. Amplitude Distribution by Type
        ax5 = fig.add_subplot(335)
        for spike_type in spike_types:
            amplitudes = [s.amplitude for s in self.spike_events if s.spike_type == spike_type]
            ax5.hist(amplitudes, alpha=0.6, label=spike_type.value, bins=20)
        ax5.set_xlabel('Spike Amplitude (mV)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Amplitude Distribution by Spike Type')
        ax5.legend()
        
        # 6. Spatial Clustering Analysis
        ax6 = fig.add_subplot(336)
        positions = np.array([s.position[:2] for s in self.spike_events])  # X, Y only
        
        # Create 2D histogram of spike locations
        hist, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax6.imshow(hist.T, extent=extent, origin='lower', cmap='hot', alpha=0.8)
        ax6.set_xlabel('X Position (mm)')
        ax6.set_ylabel('Y Position (mm)')
        ax6.set_title('Spatial Spike Density')
        plt.colorbar(im, ax=ax6, label='Spike Count')
        
        # 7. Environmental Context Analysis
        ax7 = fig.add_subplot(337)
        if self.environmental_context:
            context_times = np.arange(len(self.environmental_context))
            attention_levels = [c.attention_level for c in self.environmental_context]
            stimulus_intensities = [c.stimulus_intensity for c in self.environmental_context]
            
            ax7.plot(context_times, attention_levels, 'b-', label='Attention Level', linewidth=2)
            ax7.plot(context_times, stimulus_intensities, 'r-', label='Stimulus Intensity', linewidth=2)
            ax7.set_xlabel('Context Sample')
            ax7.set_ylabel('Level (0-1)')
            ax7.set_title('Environmental Context Over Time')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Cross-correlation between brain regions
        ax8 = fig.add_subplot(338)
        # Divide space into regions and calculate cross-correlation
        regions = {
            'anterior': [(x, y, z) for x, y, z in [s.position for s in self.spike_events] if x > 0],
            'posterior': [(x, y, z) for x, y, z in [s.position for s in self.spike_events] if x <= 0]
        }
        
        # Calculate spike counts in time bins for each region
        for region_name, positions in regions.items():
            if positions:
                region_spikes = [s for s in self.spike_events if s.position in positions[:50]]  # Limit for demo
                region_times = [s.timestamp for s in region_spikes]
                if region_times:
                    counts, _ = np.histogram(region_times, bins=time_bins)
                    ax8.plot(time_bins[:-1], counts, label=f'{region_name} region', linewidth=2)
        
        ax8.set_xlabel('Time (ms)')
        ax8.set_ylabel('Spike Count')
        ax8.set_title('Regional Activity Comparison')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Spike Type Transitions
        ax9 = fig.add_subplot(339)
        # Analyze sequences of spike types
        sorted_spikes = sorted(self.spike_events, key=lambda x: x.timestamp)
        transitions = {}
        for i in range(len(sorted_spikes) - 1):
            current_type = sorted_spikes[i].spike_type.value
            next_type = sorted_spikes[i + 1].spike_type.value
            transition = f"{current_type} → {next_type}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Show top 10 transitions
        top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_transitions:
            labels, counts = zip(*top_transitions)
            y_pos = np.arange(len(labels))
            ax9.barh(y_pos, counts)
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(labels, fontsize=8)
            ax9.set_xlabel('Transition Count')
            ax9.set_title('Most Common Spike Type Transitions')
        
        plt.tight_layout()
        
        # Save the comprehensive analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("graphs", f"neural_analysis_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Neural analysis saved to {save_path}")
        return save_path

# Example usage and data format specification
def create_sample_rust_data_file():
    """Create a sample JSON file that shows the expected format from neuro-sim-rust"""
    
    sample_data = {
        "simulation_metadata": {
            "duration_ms": 5000.0,
            "timestep_ms": 0.1,
            "brain_region": "hippocampus",
            "neuron_count": 500,
            "simulation_type": "memory_formation"
        },
        "spike_events": [
            {
                "neuron_id": 12345,
                "timestamp_ms": 1234.5,
                "spike_type": "excitatory",
                "amplitude_mv": 70.2,
                "position_xyz": [2.1, -1.4, 0.8],
                "neuron_type": "pyramidal",
                "synaptic_weights": [0.5, 0.8, 0.3],
                "membrane_potential": -55.0,
                "refractory_period_ms": 2.0,
                "input_current": 15.2
            },
            {
                "neuron_id": 12346,
                "timestamp_ms": 1235.1,
                "spike_type": "inhibitory",
                "amplitude_mv": -65.8,
                "position_xyz": [1.8, -1.1, 0.9],
                "neuron_type": "interneuron",
                "synaptic_weights": [0.3, 0.6],
                "membrane_potential": -70.0,
                "refractory_period_ms": 1.5,
                "input_current": -8.7
            }
        ],
        "environmental_context": [
            {
                "timestamp_ms": 1234.0,
                "stimulus_type": "visual_pattern",
                "stimulus_intensity": 0.8,
                "sensory_inputs": {
                    "visual": 0.9,
                    "auditory": 0.1,
                    "tactile": 0.0,
                    "olfactory": 0.0
                },
                "motor_outputs": {
                    "movement_x": 0.0,
                    "movement_y": 0.2,
                    "head_turn": 0.1
                },
                "brain_state": "learning",
                "attention_level": 0.7,
                "arousal_level": 0.6,
                "task_difficulty": 0.5
            }
        ],
        "connectivity_matrix": {
            "description": "Adjacency matrix showing synaptic connections",
            "format": "sparse",
            "connections": [
                {"from_neuron": 12345, "to_neuron": 12346, "weight": 0.75, "delay_ms": 1.2},
                {"from_neuron": 12346, "to_neuron": 12345, "weight": -0.45, "delay_ms": 0.8}
            ]
        }
    }
    
    # Save sample data file
    with open("data/sample_neuro_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created data/sample_neuro_data.json showing expected Rust data format")
    return "data/sample_neuro_data.json"

if __name__ == "__main__":
    # Create processor instance
    processor = NeuroDataProcessor()
    
    # Option 1: Load from Rust simulator (if file exists)
    # processor.load_rust_spike_data("path_to_rust_output.json")
    
    # Option 2: Create sample data for demonstration
    processor.create_sample_data()
    
    # Analyze and visualize
    processor.analyze_spike_patterns()
    
    # Create sample data format specification
    create_sample_rust_data_file()
    
    print("\nData Format Requirements for neuro-sim-rust:")
    print("=" * 50)
    print("1. JSON format with spike events and environmental context")
    print("2. Each spike event needs: neuron_id, timestamp, type, position, amplitude")
    print("3. Environmental context for stimulus-response mapping")
    print("4. Optional: connectivity matrix for network analysis")
    print("5. Spatial coordinates should be in mm relative to brain region center")
    print("6. Timestamps in milliseconds with sub-millisecond precision") 