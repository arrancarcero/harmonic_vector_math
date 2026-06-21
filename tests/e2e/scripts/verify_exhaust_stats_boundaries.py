import sys
import torch
import math
from harmonic_gemm_sparse import ZeroSparseGEMM

def verify_ground_stream_size_0():
    print("Verifying ground stream size 0...")
    try:
        gemm = ZeroSparseGEMM(in_features=8, out_features=16)
        gemm.resonant_mask_idx = torch.arange(24)
        x = torch.randn(2, 24, 8)
        gemm(x)
        assert gemm.exhaust_energy == 0.0
        print("PASS: ground stream size 0")
        return True
    except Exception as e:
        print(f"FAIL: ground stream size 0 raised {e}")
        return False

def verify_all_resonant_inputs():
    print("Verifying all resonant inputs...")
    try:
        gemm = ZeroSparseGEMM(in_features=8, out_features=16)
        gemm.resonant_mask_idx = torch.tensor([0])
        x = torch.randn(1, 1, 8)
        gemm(x)
        assert gemm.exhaust_energy == 0.0
        print("PASS: all resonant inputs")
        return True
    except Exception as e:
        print(f"FAIL: all resonant inputs raised {e}")
        return False

def verify_overflow():
    print("Verifying overflow...")
    try:
        gemm = ZeroSparseGEMM(in_features=8, out_features=16)
        x = torch.zeros(1, 24, 8)
        x[0, 0, 1] = 1e20
        gemm(x)
        assert math.isinf(gemm.exhaust_energy) or gemm.exhaust_energy >= 1e20
        print("PASS: overflow")
        return True
    except Exception as e:
        print(f"FAIL: overflow raised {e}")
        return False

def verify_underflow():
    print("Verifying underflow...")
    try:
        gemm = ZeroSparseGEMM(in_features=8, out_features=16)
        x = torch.zeros(1, 24, 8)
        x[0, 0, 1] = 1e-45
        gemm(x)
        assert gemm.exhaust_energy == 0.0 or gemm.exhaust_energy < 1e-20
        print("PASS: underflow")
        return True
    except Exception as e:
        print(f"FAIL: underflow raised {e}")
        return False

def verify_empty_inputs():
    print("Verifying empty inputs...")
    try:
        gemm = ZeroSparseGEMM(in_features=8, out_features=16)
        
        x_b0 = torch.zeros(0, 24, 8)
        gemm(x_b0)
        assert gemm.exhaust_energy == 0.0
        
        x_s0 = torch.zeros(2, 0, 8)
        gemm(x_s0)
        assert gemm.exhaust_energy == 0.0
        
        print("PASS: empty inputs")
        return True
    except Exception as e:
        print(f"FAIL: empty inputs raised {e}")
        return False

if __name__ == "__main__":
    results = {
        "ground_stream_0": verify_ground_stream_size_0(),
        "all_resonant": verify_all_resonant_inputs(),
        "overflow": verify_overflow(),
        "underflow": verify_underflow(),
        "empty_inputs": verify_empty_inputs(),
    }
    
    print("\n--- Dual-Stream Exhaust Stats Boundaries Summary ---")
    all_passed = True
    for name, res in results.items():
        print(f"{name}: {'Passed' if res else 'Failed'}")
        if not res:
            all_passed = False
            
    if all_passed:
        print("All Dual-Stream Exhaust Stats boundaries verified successfully!")
        sys.exit(0)
    else:
        sys.exit(1)
