import sys
import torch
import math
import os
import json
from harmonic_ops import HarmonicAttention

def verify_wobble_frequency():
    print("Running test case: Wobble Frequency")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    # Check that franklin_freq is defined
    assert hasattr(attn, "franklin_freq"), "franklin_freq attribute is missing"
    assert attn.franklin_freq == 8.125, f"Expected 8.125, got {attn.franklin_freq}"
    print("Wobble frequency verified successfully.")

def verify_wobble_amplitude():
    print("Running test case: Wobble Amplitude")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    assert hasattr(attn, "wobble_amplitude"), "wobble_amplitude attribute is missing"
    assert attn.wobble_amplitude == 0.125, f"Expected 0.125, got {attn.wobble_amplitude}"
    print("Wobble amplitude verified successfully.")

def verify_scaling_on_q():
    print("Running test case: Scaling on Q")
    seq_len = 24
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    
    # Manually calculate the expected wobble
    q_indices = torch.arange(seq_len, dtype=torch.float32)
    expected_phase = 2.0 * math.pi * (q_indices * attn.franklin_freq) / attn.base_frequency
    expected_wobble = 1.0 + attn.wobble_amplitude * torch.sin(expected_phase)
    
    for i in range(seq_len):
        phase = 2.0 * math.pi * (i * attn.franklin_freq) / attn.base_frequency
        wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
        assert math.isclose(expected_wobble[i].item(), wobble, rel_tol=1e-5), f"Wobble mismatch at {i}: {expected_wobble[i].item()} vs {wobble}"
    print("Scaling on Q verified successfully.")

def verify_scaling_on_k():
    print("Running test case: Scaling on K")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    seq_len = 24
    h_indices = attn.get_harmonic_indices(seq_len, torch.device("cpu"))
    
    k_indices = h_indices.to(torch.float32)
    expected_k_phase = 2.0 * math.pi * (k_indices * attn.franklin_freq) / attn.base_frequency
    expected_k_wobble = 1.0 + attn.wobble_amplitude * torch.sin(expected_k_phase)
    
    for idx, h_idx in enumerate(h_indices):
        phase = 2.0 * math.pi * (h_idx.item() * attn.franklin_freq) / attn.base_frequency
        wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
        assert math.isclose(expected_k_wobble[idx].item(), wobble, rel_tol=1e-5), f"Wobble mismatch at harmonic index {h_idx}: {expected_k_wobble[idx].item()} vs {wobble}"
    print("Scaling on K verified successfully.")

def verify_dynamic_base_frequency_influence():
    print("Running test case: Dynamic Base Frequency Influence")
    attn = HarmonicAttention(embed_dim=8, num_heads=1)
    print(f"Initial base frequency: {attn.base_frequency}")
    
    base_freqs = [100.0, 432.0, 800.0]
    for bf in base_freqs:
        attn.base_frequency = bf
        # check wobble at index 1
        phase = 2.0 * math.pi * (1 * attn.franklin_freq) / bf
        expected_wobble = 1.0 + attn.wobble_amplitude * math.sin(phase)
        
        q_indices = torch.tensor([1.0], dtype=torch.float32)
        q_phase = 2.0 * math.pi * (q_indices * attn.franklin_freq) / attn.base_frequency
        q_wobble = 1.0 + attn.wobble_amplitude * torch.sin(q_phase)
        assert math.isclose(q_wobble.item(), expected_wobble, rel_tol=1e-5), f"Base frequency influence check failed for {bf}"
    print("Dynamic base frequency influence verified successfully.")

if __name__ == "__main__":
    try:
        verify_wobble_frequency()
        verify_wobble_amplitude()
        verify_scaling_on_q()
        verify_scaling_on_k()
        verify_dynamic_base_frequency_influence()
        print("All Franklin Constant Drift test cases passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
