import sys
import torch
import math
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_initialization_of_exhaust_energy():
    print("Running test case: Initialization of Exhaust Energy")
    gemm = ZeroSparseGEMM(16, 16)
    assert hasattr(gemm, "exhaust_energy"), "exhaust_energy attribute is missing"
    assert gemm.exhaust_energy == 0.0, f"Expected 0.0, got {gemm.exhaust_energy}"
    print("Initialization of exhaust energy verified successfully.")

def test_update_on_forward_pass():
    print("Running test case: Update on Forward Pass")
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.ones(1, 24, 16)
    out = gemm(x)
    assert gemm.exhaust_energy > 0.0, f"Expected positive exhaust energy, got {gemm.exhaust_energy}"
    print("Update on forward pass verified successfully.")

def test_l2_norm_correctness():
    print("Running test case: L2-Norm Correctness")
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16)
    out = gemm(x)
    
    indices = torch.arange(24) % 24
    resonant_mask = torch.any(indices.unsqueeze(1) == gemm.resonant_mask_idx, dim=1)
    
    x_masked = x * gemm.pin_mask.view(1, 1, -1)
    x_void = x_masked[:, ~resonant_mask, :]
    expected_norm = torch.norm(x_void).item()
    
    assert math.isclose(gemm.exhaust_energy, expected_norm, rel_tol=1e-5), f"Norm mismatch: {gemm.exhaust_energy} vs {expected_norm}"
    print("L2-norm correctness verified successfully.")

def test_no_grad_operation():
    print("Running test case: No-Grad Operation")
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16, requires_grad=True)
    out = gemm(x)
    
    assert isinstance(gemm.exhaust_energy, float), f"Expected exhaust_energy to be float, got {type(gemm.exhaust_energy)}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    print("No-grad operation verified successfully.")

def test_behavior_with_zero_ground_stream():
    print("Running test case: Behavior with Zero Ground Stream")
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.zeros(1, 24, 16)
    out = gemm(x)
    assert gemm.exhaust_energy == 0.0, f"Expected 0.0, got {gemm.exhaust_energy}"
    print("Behavior with zero ground stream verified successfully.")

if __name__ == "__main__":
    try:
        test_initialization_of_exhaust_energy()
        test_update_on_forward_pass()
        test_l2_norm_correctness()
        test_no_grad_operation()
        test_behavior_with_zero_ground_stream()
        print("All Dual-Stream Exhaust Stats test cases passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
