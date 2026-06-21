import sys
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def verify_non_multiples():
    print("Verifying in_features not multiple of 16 (15 and 17)...")
    try:
        gemm_15 = ZeroSparseGEMM(15, 16)
        x_15 = torch.randn(2, 24, 15)
        out_15 = gemm_15(x_15)
        assert out_15.shape == (2, 24, 16)
        
        gemm_17 = ZeroSparseGEMM(17, 16)
        x_17 = torch.randn(2, 24, 17)
        out_17 = gemm_17(x_17)
        assert out_17.shape == (2, 24, 16)
        
        print("PASS: non-multiples of 16")
        return True
    except Exception as e:
        print(f"FAIL: non-multiples of 16 raised {e}")
        return False

def verify_weight_shape_broadcasting():
    print("Verifying weight shape broadcasting...")
    try:
        gemm = ZeroSparseGEMM(in_features=25, out_features=16)
        for shard in gemm.gate_shards:
            masked_shard = shard * gemm.pin_mask
            assert masked_shard.shape == (2, 25)
        print("PASS: weight shape broadcasting")
        return True
    except Exception as e:
        print(f"FAIL: weight shape broadcasting raised {e}")
        return False

def verify_in_features_1():
    print("Verifying in_features=1...")
    try:
        gemm = ZeroSparseGEMM(in_features=1, out_features=8)
        assert gemm.pin_mask.shape == (1,)
        assert gemm.pin_mask[0] == False
        x = torch.randn(2, 24, 1)
        out = gemm(x)
        assert torch.all(out == 0)
        print("PASS: in_features=1")
        return True
    except Exception as e:
        print(f"FAIL: in_features=1 raised {e}")
        return False

def verify_active_slot_bounds():
    print("Verifying active slot bounds...")
    try:
        gemm = ZeroSparseGEMM(in_features=16, out_features=8)
        
        # Slot 0 is masked
        x_masked = torch.zeros(1, 24, 16)
        x_masked[0, :, 0] = 10.0
        out_masked = gemm(x_masked)
        assert torch.all(out_masked == 0)
        
        # Slot 1 is active
        x_active = torch.zeros(1, 24, 16)
        x_active[0, :, 1] = 10.0
        out_active = gemm(x_active)
        assert not torch.all(out_active == 0)
        
        print("PASS: active slot bounds")
        return True
    except Exception as e:
        print(f"FAIL: active slot bounds raised {e}")
        return False

def verify_fixed_point_mode():
    print("Verifying fixed point mode support...")
    try:
        gemm = ZeroSparseGEMM(in_features=16, out_features=8, fixed_point=True)
        x = torch.randn(2, 24, 16)
        out = gemm(x)
        assert out.shape == (2, 24, 8)
        print("PASS: fixed point mode")
        return True
    except Exception as e:
        print(f"FAIL: fixed point mode raised {e}")
        return False

if __name__ == "__main__":
    results = {
        "non_multiples": verify_non_multiples(),
        "weight_broadcasting": verify_weight_shape_broadcasting(),
        "in_features_1": verify_in_features_1(),
        "active_slots": verify_active_slot_bounds(),
        "fixed_point": verify_fixed_point_mode(),
    }
    
    print("\n--- BZS-GEMM M24308 Mapping Boundaries Summary ---")
    all_passed = True
    for name, res in results.items():
        print(f"{name}: {'Passed' if res else 'Failed'}")
        if not res:
            all_passed = False
            
    if all_passed:
        print("All M24308 Mapping boundaries verified successfully!")
        sys.exit(0)
    else:
        sys.exit(1)
