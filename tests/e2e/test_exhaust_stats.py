import pytest
import torch
import math
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_initialization_of_exhaust_energy():
    """Verify that exhaust_energy is initialized to 0.0."""
    gemm = ZeroSparseGEMM(16, 16)
    assert hasattr(gemm, "exhaust_energy"), "exhaust_energy attribute is missing"
    assert gemm.exhaust_energy == 0.0, f"Expected 0.0, got {gemm.exhaust_energy}"

def test_update_on_forward_pass():
    """Verify that a forward pass updates exhaust_energy to a positive value when non-resonant inputs exist."""
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.ones(1, 24, 16)
    out = gemm(x)
    assert gemm.exhaust_energy > 0.0, f"Expected positive exhaust energy, got {gemm.exhaust_energy}"

def test_l2_norm_correctness():
    """Verify that exhaust_energy correctly matches the L2-norm of the non-resonant ground stream."""
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16)
    out = gemm(x)
    
    indices = torch.arange(24) % 24
    resonant_mask = torch.any(indices.unsqueeze(1) == gemm.resonant_mask_idx, dim=1)
    
    x_masked = x * gemm.pin_mask.view(1, 1, -1)
    x_void = x_masked[:, ~resonant_mask, :]
    expected_norm = torch.norm(x_void).item()
    
    assert math.isclose(gemm.exhaust_energy, expected_norm, rel_tol=1e-5)

def test_no_grad_operation():
    """Verify that the exhaust energy calculation operates under no_grad and doesn't hold gradient nodes."""
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16, requires_grad=True)
    out = gemm(x)
    
    assert isinstance(gemm.exhaust_energy, float)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_behavior_with_zero_ground_stream():
    """Verify that passing an all-zeros input yields 0.0 exhaust energy."""
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.zeros(1, 24, 16)
    out = gemm(x)
    assert gemm.exhaust_energy == 0.0, f"Expected 0.0, got {gemm.exhaust_energy}"

def test_exhaust_stats_subprocess(subprocess_runner):
    """Run the verify_exhaust_stats.py script in a subprocess to check execution flow."""
    result = subprocess_runner("tests/e2e/scripts/verify_exhaust_stats.py")
    assert result.returncode == 0
    assert "All Dual-Stream Exhaust Stats test cases passed!" in result.stdout
