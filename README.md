# MemoryMapping

A Python library for processing and visualizing neural spike data, designed to interface with Rust-based neural simulators.

## Features

- **Data Processing**: Load and parse neural spike event data from JSON.
- **Visualization**: comprehensive plotting of neural activity:
    - 3D Spatial Distribution
    - Temporal Spike Trains
    - Spike Rate Analysis
    - Neuron Type & Amplitude Distribution
    - Spatio-temporal Activity Maps
    - Environmental Context Correlation

## Installation

```bash
pip install -e .
```

For development dependencies (testing, linting):

```bash
pip install -e ".[dev]"
```

## Usage

### Analyzing Neural Data

```python
from memorymapping import NeuroDataProcessor

processor = NeuroDataProcessor()
processor.load_rust_spike_data("data/simulation_output.json")
processor.analyze_spike_patterns()
```

### Running Scripts

```bash
# Generate analysis from sample data
python3 scripts/analyze_experience.py
```

## Testing

Run the test suite:

```bash
pytest
```
