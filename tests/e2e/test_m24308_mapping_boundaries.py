import pytest
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_in_features_not_multiple_of_16():
    """Verify that in_features not multiple of 16 (e.g. 15 or 17) compiles and runs correctly."""
    # Case 15
    gemm_15 = ZeroSparseGEMM(in_features=15, out_features=16)
    assert gemm_15.pin_mask.shape == (15,)
    assert gemm_15.pin_mask[0] == False
    assert gemm_15.pin_mask[14] == False
    assert gemm_15.pin_mask[1] == True
    x_15 = torch.randn(2, 24, 15)
    out_15 = gemm_15(x_15)
    assert out_15.shape == (2, 24, 16)

    # Case 17
    gemm_17 = ZeroSparseGEMM(in_features=17, out_features=16)
    assert gemm_17.pin_mask.shape == (17,)
    assert gemm_17.pin_mask[0] == False
    assert gemm_17.pin_mask[14] == False
    assert gemm_17.pin_mask[16] == False
    assert gemm_17.pin_mask[15] == True
    x_17 = torch.randn(2, 24, 17)
    out_17 = gemm_17(x_17)
    assert out_17.shape == (2, 24, 16)

def test_weight_shape_not_multiple_of_16():
    """Verify that weight shards with dimensions not multiple of 16 broadcast correctly with the pin mask."""
    gemm = ZeroSparseGEMM(in_features=25, out_features=16)
    for shard in gemm.gate_shards:
        assert shard.shape == (2, 25)
        # Verify broadcasting shard * pin_mask works
        masked_shard = shard * gemm.pin_mask
        assert masked_shard.shape == (2, 25)

def test_in_features_1():
    """Verify corner case in_features=1."""
    gemm = ZeroSparseGEMM(in_features=1, out_features=8)
    assert gemm.pin_mask.shape == (1,)
    assert gemm.pin_mask[0] == False
    
    x = torch.randn(2, 24, 1)
    out = gemm(x)
    assert out.shape == (2, 24, 8)
    assert torch.all(out == 0)

def test_active_slot_bounds():
    """Verify that non-masked slots (active slots) propagate values while masked slots do not."""
    gemm = ZeroSparseGEMM(in_features=16, out_features=8)
    
    # Test index 0 (masked): should not affect output
    x_masked = torch.zeros(1, 24, 16)
    x_masked[0, :, 0] = 10.0 # slot 0 is masked
    out_masked = gemm(x_masked)
    assert torch.all(out_masked == 0)
    
    # Test index 1 (active): should affect output
    x_active = torch.zeros(1, 24, 16)
    x_active[0, :, 1] = 10.0 # slot 1 is active
    out_active = gemm(x_active)
    assert not torch.all(out_active == 0)

def test_pin_mask_no_crash_modes():
    """Verify that pin masks don't crash in standard vs fixed-point forward modes."""
    gemm_fp = ZeroSparseGEMM(in_features=16, out_features=8, fixed_point=True)
    x = torch.randn(2, 24, 16)
    out_fp = gemm_fp(x)
    assert out_fp.shape == (2, 24, 8)
    
    gemm_std = ZeroSparseGEMM(in_features=16, out_features=8, fixed_point=False)
    out_std = gemm_std(x)
    assert out_std.shape == (2, 24, 8)

def test_m24308_mapping_boundaries_subprocess(subprocess_runner):
    """Run the verify_m24308_mapping_boundaries.py script in a subprocess."""
    result = subprocess_runner("tests/e2e/scripts/verify_m24308_mapping_boundaries.py")
    assert result.returncode == 0
