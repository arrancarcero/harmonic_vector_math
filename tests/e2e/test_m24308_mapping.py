import pytest
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_pin_mask_generation():
    """Verify BZS-GEMM generates the pin_mask buffer with correct shape and type."""
    gemm = ZeroSparseGEMM(16, 16)
    assert hasattr(gemm, "pin_mask"), "pin_mask attribute is missing"
    assert isinstance(gemm.pin_mask, torch.Tensor), "pin_mask must be a torch.Tensor"
    assert gemm.pin_mask.dtype == torch.bool, "pin_mask must be of boolean type"
    assert gemm.pin_mask.shape == (16,), f"Expected shape (16,), got {gemm.pin_mask.shape}"

def test_modulo_16_index_zeroing():
    """Verify that indices congruent to 0 or 14 modulo 16 are masked (set to False)."""
    gemm = ZeroSparseGEMM(32, 16)
    mask = gemm.pin_mask
    for i in range(32):
        if i % 16 == 0 or i % 16 == 14:
            assert mask[i] == False, f"Expected False at index {i}, got {mask[i]}"
        else:
            assert mask[i] == True, f"Expected True at index {i}, got {mask[i]}"

def test_input_masking_verification():
    """Verify that inputs at the masked indices (0, 14 mod 16) do not affect the output of BZS-GEMM."""
    gemm = ZeroSparseGEMM(16, 16)
    x_with_vals = torch.ones(1, 24, 16)
    x_with_vals[0, :, 0] = 99.0
    x_with_vals[0, :, 14] = -99.0
    
    x_zeroed = torch.ones(1, 24, 16)
    x_zeroed[0, :, 0] = 0.0
    x_zeroed[0, :, 14] = 0.0
    
    out_vals = gemm(x_with_vals)
    out_zeroed = gemm(x_zeroed)
    
    assert torch.allclose(out_vals, out_zeroed, atol=1e-6)

def test_weight_shard_masking_verification():
    """Verify that weight shards are masked at indices 0 and 14 mod 16 before computation."""
    gemm = ZeroSparseGEMM(16, 16)
    for shard in gemm.gate_shards:
        masked = shard * gemm.pin_mask
        assert torch.all(masked[:, 0] == 0), "Column 0 of weight shard not masked"
        assert torch.all(masked[:, 14] == 0), "Column 14 of weight shard not masked"
        assert not torch.all(masked[:, 1] == 0)

def test_output_zero_zone_verification():
    """Verify that output at non-resonant gates (void gates) is zeroed out by BZS-GEMM."""
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16)
    out = gemm(x)
    
    for i in range(24):
        gate = i % 24
        if gate not in gemm.open_gates:
            assert torch.all(out[0, i, :] == 0.0), f"Expected zero output at position {i} (gate {gate})"

def test_m24308_mapping_subprocess(subprocess_runner):
    """Run the verify_m24308_mapping.py script in a subprocess to check execution flow."""
    result = subprocess_runner("tests/e2e/scripts/verify_m24308_mapping.py")
    assert result.returncode == 0
    assert "All M24308 Mapping test cases passed!" in result.stdout
