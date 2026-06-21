import sys
import torch
import math
from harmonic_ops import HarmonicAttention

def verify_seq_len_0():
    print("Verifying seq_len=0...")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.zeros(2, 0, 8)
    out = attn(x)
    assert out.shape == (2, 0, 8)
    print("PASS: seq_len=0")

def verify_seq_len_1():
    print("Verifying seq_len=1...")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.randn(2, 1, 8)
    out = attn(x)
    assert out.shape == (2, 1, 8)
    print("PASS: seq_len=1")

def verify_seq_len_2048():
    print("Verifying seq_len=2048...")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    x = torch.randn(1, 2048, 8)
    out = attn(x)
    assert out.shape == (1, 2048, 8)
    assert not torch.isnan(out).any()
    print("PASS: seq_len=2048")

def verify_extreme_base_frequencies():
    print("Verifying extreme base frequencies...")
    # Very large base frequency
    attn_large = HarmonicAttention(embed_dim=8, num_heads=1)
    attn_large.base_frequency = 1e12
    x = torch.randn(1, 24, 8)
    out_large = attn_large(x)
    assert not torch.isnan(out_large).any()
    
    # Very small base frequency
    attn_small = HarmonicAttention(embed_dim=8, num_heads=1)
    attn_small.base_frequency = 1e-6
    out_small = attn_small(x)
    assert not torch.isnan(out_small).any()
    print("PASS: extreme base frequencies")

def verify_extreme_sine_phase_indices():
    print("Verifying extreme sine phase indices...")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    large_index = torch.tensor([1e9], dtype=torch.float32)
    phase = 2.0 * math.pi * (large_index * attn.franklin_freq) / attn.base_frequency
    wobble = 1.0 + attn.wobble_amplitude * torch.sin(phase)
    assert not torch.isnan(wobble)
    assert (wobble >= 0.875).all() and (wobble <= 1.125).all()
    print("PASS: extreme sine phase indices")

if __name__ == "__main__":
    try:
        verify_seq_len_0()
        verify_seq_len_1()
        verify_seq_len_2048()
        verify_extreme_base_frequencies()
        verify_extreme_sine_phase_indices()
        print("All Franklin Constant Drift boundaries verified successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
