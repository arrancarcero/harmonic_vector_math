import pytest
import torch
from harmonic_ops import FixedPointTensor

def test_scale_factor_extremes():
    """Verify behaviour with scale factor extremes (extremely small scale=1, and extremely large scale=1e8)."""
    fp_small = FixedPointTensor(1.7, scale=1)
    assert fp_small.tensor.item() == 2
    assert fp_small.to_float().item() == 2.0
    
    fp_large = FixedPointTensor(1.23456789, scale=100000000)
    assert fp_large.tensor.item() == 123456789
    assert abs(fp_large.to_float().item() - 1.23456789) < 1e-8

def test_rounding_boundary_0_5():
    """Verify rounding at the exact 0.5 boundary."""
    fp1 = FixedPointTensor(0.005, scale=100)
    assert fp1.tensor.item() == 0
    
    fp2 = FixedPointTensor(0.015, scale=100)
    assert fp2.tensor.item() == 2

def test_int64_overflow():
    """Verify behavior of inputs causing int64 overflow."""
    huge_val = 9e18
    fp = FixedPointTensor(huge_val, scale=1)
    assert fp.tensor.item() is not None

def test_zero_tensors():
    """Verify behaviour of zero tensors."""
    x = torch.zeros(2, 3)
    fp = FixedPointTensor(x, scale=100)
    assert torch.all(fp.tensor == 0)
    assert torch.all(fp.to_float() == 0.0)
    
    y = FixedPointTensor(torch.randn(2, 3), scale=100)
    res = fp * y
    assert torch.all(res.tensor == 0)

def test_near_zero_tensors():
    """Verify behaviour of near-zero tensors (values less than 1/scale)."""
    x = torch.tensor([0.004])
    fp = FixedPointTensor(x, scale=100)
    assert fp.tensor.item() == 0
    assert fp.to_float().item() == 0.0

def test_fixed_point_ops_boundaries_subprocess(subprocess_runner):
    """Run verify_fixed_point_ops_boundaries.py in a subprocess."""
    result = subprocess_runner("tests/e2e/scripts/verify_fixed_point_ops_boundaries.py")
    assert result.returncode == 0
