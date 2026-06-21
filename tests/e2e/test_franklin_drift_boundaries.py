import pytest
import math
import torch
from harmonic_ops import HarmonicAttention

def test_seq_len_0():
    """Verify sequence length 0 handles empty input gracefully without runtime crash."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.zeros(2, 0, 8)
    out = attn(x)
    assert out.shape == (2, 0, 8)

def test_seq_len_1():
    """Verify sequence length 1 processes correctly."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.randn(2, 1, 8)
    out = attn(x)
    assert out.shape == (2, 1, 8)

def test_seq_len_2048():
    """Verify sequence length 2048 handles large sequences without numerical overflow/instability."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.randn(1, 2048, 8)
    out = attn(x)
    assert out.shape == (1, 2048, 8)
    assert not torch.isnan(out).any()

def test_extreme_base_frequencies(override_config):
    """Verify behaviour with extreme base frequencies: extremely small (approaching 0) and extremely large."""
    # Case A: Extremely large base frequency
    override_config({"hardware_heritage": {"base_frequency_hz": 1e12}})
    attn_large = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.randn(1, 24, 8)
    out_large = attn_large(x)
    assert not torch.isnan(out_large).any()
    
    # When base frequency is extremely large, phase approaches 0, sin(phase) -> 0, wobble -> 1.0
    q_indices = torch.arange(24, dtype=torch.float32)
    q_phase = 2.0 * math.pi * (q_indices * attn_large.franklin_freq) / attn_large.base_frequency
    q_wobble = 1.0 + attn_large.wobble_amplitude * torch.sin(q_phase)
    assert torch.allclose(q_wobble, torch.ones_like(q_wobble), atol=1e-5)

    # Case B: Extremely small base frequency (near zero, but not zero to avoid ZeroDivisionError unless expected)
    override_config({"hardware_heritage": {"base_frequency_hz": 1e-6}})
    attn_small = HarmonicAttention(embed_dim=8, num_heads=1)
    out_small = attn_small(x)
    assert not torch.isnan(out_small).any()

def test_extreme_sine_phase_indices():
    """Verify numerical stability when phase indices or sine inputs are extremely large or small."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    # Simulate a phase calculation with extremely large virtual index (e.g. 1e9)
    large_index = torch.tensor([1e9], dtype=torch.float32)
    phase = 2.0 * math.pi * (large_index * attn.franklin_freq) / attn.base_frequency
    wobble = 1.0 + attn.wobble_amplitude * torch.sin(phase)
    assert not torch.isnan(wobble)
    assert (wobble >= 0.875).all() and (wobble <= 1.125).all()

def test_franklin_drift_boundaries_subprocess(subprocess_runner):
    """Run verification script in a subprocess."""
    result = subprocess_runner("tests/e2e/scripts/verify_franklin_drift_boundaries.py")
    assert result.returncode == 0
