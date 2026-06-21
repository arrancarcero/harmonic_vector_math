import pytest
import torch
import math
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_ground_stream_size_0():
    """Verify that when the ground stream has size 0 (all inputs are resonant), it does not crash and exhaust energy is 0."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    gemm.resonant_mask_idx = torch.arange(24)
    x = torch.randn(2, 24, 8)
    out = gemm(x)
    assert gemm.exhaust_energy == 0.0

def test_all_resonant_stream_inputs():
    """Verify behaviour when the inputs consist only of resonant stream timesteps. E.g. seq_len=1 with a resonant index (1)."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    gemm.resonant_mask_idx = torch.tensor([0])
    x = torch.randn(1, 1, 8)
    out = gemm(x)
    assert gemm.exhaust_energy == 0.0

def test_extreme_values_overflow():
    """Verify that extremely large values in the ground stream do not crash the module and result in infinity or very large value."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    x = torch.zeros(1, 24, 8)
    x[0, 0, 1] = 1e20
    out = gemm(x)
    assert math.isinf(gemm.exhaust_energy) or gemm.exhaust_energy >= 1e20

def test_near_zero_underflow():
    """Verify that extremely small values (near-zero) in the ground stream do not crash and handle underflow gracefully."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    x = torch.zeros(1, 24, 8)
    x[0, 0, 1] = 1e-45
    out = gemm(x)
    assert gemm.exhaust_energy == 0.0 or gemm.exhaust_energy < 1e-20

def test_empty_inputs():
    """Verify that empty inputs (such as batch size 0 or sequence length 0) do not crash and result in 0.0 exhaust energy."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    
    # Batch size 0
    x_b0 = torch.zeros(0, 24, 8)
    out_b0 = gemm(x_b0)
    assert gemm.exhaust_energy == 0.0
    assert out_b0.shape == (0, 24, 16)
    
    # Seq len 0
    x_s0 = torch.zeros(2, 0, 8)
    out_s0 = gemm(x_s0)
    assert gemm.exhaust_energy == 0.0
    assert out_s0.shape == (2, 0, 16)

def test_exhaust_stats_boundaries_subprocess(subprocess_runner):
    """Run verify_exhaust_stats_boundaries.py in a subprocess."""
    result = subprocess_runner("tests/e2e/scripts/verify_exhaust_stats_boundaries.py")
    assert result.returncode == 0
