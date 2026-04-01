import pytest
from memorymapping.processor import NeuroDataProcessor, SpikeEvent, SpikeType

def test_processor_initialization():
    processor = NeuroDataProcessor()
    assert processor.spike_events == []
    assert processor.environmental_context == []

def test_create_sample_data():
    processor = NeuroDataProcessor()
    processor.create_sample_data()
    assert len(processor.spike_events) > 0
    assert len(processor.environmental_context) > 0

def test_spike_event_structure():
    processor = NeuroDataProcessor()
    processor.create_sample_data()
    event = processor.spike_events[0]
    assert isinstance(event.neuron_id, int)
    assert isinstance(event.timestamp, float)
    assert isinstance(event.spike_type, SpikeType)
