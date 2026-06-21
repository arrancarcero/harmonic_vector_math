import sys
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_pin_mask_generation():
    print("Running test case: Pin Mask Generation")
    gemm = ZeroSparseGEMM(16, 16)
    assert hasattr(gemm, "pin_mask"), "pin_mask attribute is missing"
    assert isinstance(gemm.pin_mask, torch.Tensor), "pin_mask must be a torch.Tensor"
    assert gemm.pin_mask.dtype == torch.bool, "pin_mask must be of boolean type"
    assert gemm.pin_mask.shape == (16,), f"Expected shape (16,), got {gemm.pin_mask.shape}"
    print("Pin mask generation verified successfully.")

def test_modulo_16_index_zeroing():
    print("Running test case: Modulo 16 Index Zeroing")
    gemm = ZeroSparseGEMM(32, 16)
    mask = gemm.pin_mask
    for i in range(32):
        if i % 16 == 0 or i % 16 == 14:
            assert mask[i] == False, f"Expected False at index {i}, got {mask[i]}"
        else:
            assert mask[i] == True, f"Expected True at index {i}, got {mask[i]}"
    print("Modulo 16 index zeroing verified successfully.")

def test_input_masking_verification():
    print("Running test case: Input Masking Verification")
    gemm = ZeroSparseGEMM(16, 16)
    x_with_vals = torch.ones(1, 24, 16)
    x_with_vals[0, :, 0] = 99.0
    x_with_vals[0, :, 14] = -99.0
    
    x_zeroed = torch.ones(1, 24, 16)
    x_zeroed[0, :, 0] = 0.0
    x_zeroed[0, :, 14] = 0.0
    
    out_vals = gemm(x_with_vals)
    out_zeroed = gemm(x_zeroed)
    
    assert torch.allclose(out_vals, out_zeroed, atol=1e-6), "Input masking failed to isolate pin_mask indices"
    print("Input masking verification verified successfully.")

def test_weight_shard_masking_verification():
    print("Running test case: Weight Shard Masking Verification")
    gemm = ZeroSparseGEMM(16, 16)
    for shard in gemm.gate_shards:
        masked = shard * gemm.pin_mask
        assert torch.all(masked[:, 0] == 0), "Column 0 of weight shard not masked"
        assert torch.all(masked[:, 14] == 0), "Column 14 of weight shard not masked"
        assert not torch.all(masked[:, 1] == 0)
    print("Weight shard masking verification verified successfully.")

def test_output_zero_zone_verification():
    print("Running test case: Output Zero-Zone Verification")
    gemm = ZeroSparseGEMM(16, 16)
    x = torch.randn(1, 24, 16)
    out = gemm(x)
    
    for i in range(24):
        gate = i % 24
        if gate not in gemm.open_gates:
            assert torch.all(out[0, i, :] == 0.0), f"Expected zero output at non-resonant position {i} (gate {gate}), got {out[0, i, :]}"
    print("Output zero-zone verification verified successfully.")

if __name__ == "__main__":
    try:
        test_pin_mask_generation()
        test_modulo_16_index_zeroing()
        test_input_masking_verification()
        test_weight_shard_masking_verification()
        test_output_zero_zone_verification()
        print("All M24308 Mapping test cases passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
