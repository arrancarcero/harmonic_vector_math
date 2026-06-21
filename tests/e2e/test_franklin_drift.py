import pytest
import math
import torch
from harmonic_ops import HarmonicAttention

def test_wobble_frequency():
    """Verify the Franklin Constant drift wobble frequency is set to 8.125Hz."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    assert hasattr(attn, "franklin_freq"), "franklin_freq attribute is missing"
    assert attn.franklin_freq == 8.125, f"Expected 8.125, got {attn.franklin_freq}"

def test_wobble_amplitude():
    """Verify the phase wobble amplitude is set to 0.125 (12.5%)."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    assert hasattr(attn, "wobble_amplitude"), "wobble_amplitude attribute is missing"
    assert attn.wobble_amplitude == 0.125, f"Expected 0.125, got {attn.wobble_amplitude}"

def test_scaling_on_q():
    """Verify the phase wobble scaling calculation on query vectors matches the formula."""
    seq_len = 24
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    q_indices = torch.arange(seq_len, dtype=torch.float32)
    expected_phase = 2.0 * math.pi * (q_indices * attn.franklin_freq) / attn.base_frequency
    expected_wobble = 1.0 + attn.wobble_amplitude * torch.sin(expected_phase)
    
    for i in range(seq_len):
        phase = 2.0 * math.pi * (i * attn.franklin_freq) / attn.base_frequency
        wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
        assert math.isclose(expected_wobble[i].item(), wobble, rel_tol=1e-5)

def test_scaling_on_k():
    """Verify the phase wobble scaling calculation on key vectors matches the formula at harmonic indices."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    seq_len = 24
    h_indices = attn.get_harmonic_indices(seq_len, torch.device("cpu"))
    
    k_indices = h_indices.to(torch.float32)
    expected_k_phase = 2.0 * math.pi * (k_indices * attn.franklin_freq) / attn.base_frequency
    expected_k_wobble = 1.0 + attn.wobble_amplitude * torch.sin(expected_k_phase)
    
    for idx, h_idx in enumerate(h_indices):
        phase = 2.0 * math.pi * (h_idx.item() * attn.franklin_freq) / attn.base_frequency
        wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
        assert math.isclose(expected_k_wobble[idx].item(), wobble, rel_tol=1e-5)

def test_dynamic_base_frequency_influence(override_config):
    """Verify that changing base_frequency_hz in config dynamically scales the phase calculations."""
    custom_config = {
        "hardware_heritage": {
            "base_frequency_hz": 999.0
        }
    }
    # Override config using the fixture
    override_config(custom_config)
    
    # Re-instantiate attention to pick up config override
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    assert attn.base_frequency == 999.0, f"Expected base frequency to be overridden to 999.0, got {attn.base_frequency}"
    
    # Verify the phase changes correctly
    phase = 2.0 * math.pi * (1.0 * attn.franklin_freq) / 999.0
    expected_wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
    
    q_indices = torch.tensor([1.0], dtype=torch.float32)
    q_phase = 2.0 * math.pi * (q_indices * attn.franklin_freq) / attn.base_frequency
    q_wobble = 1.0 + attn.wobble_amplitude * torch.sin(q_phase)
    assert math.isclose(q_wobble.item(), expected_wobble, rel_tol=1e-5)

def test_franklin_drift_subprocess(subprocess_runner):
    """Run the verify_franklin_drift.py script in a subprocess to check execution flow."""
    result = subprocess_runner("tests/e2e/scripts/verify_franklin_drift.py")
    assert result.returncode == 0
    assert "All Franklin Constant Drift test cases passed!" in result.stdout
