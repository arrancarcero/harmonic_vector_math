import sys
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def verify_mismatched_shapes():
    print("Verifying mismatched shapes...")
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    x = torch.randn(2, 24, 9)
    try:
        gemm(x)
        print("FAIL: mismatched shapes did not raise an exception")
        return False
    except RuntimeError as e:
        print(f"PASS: mismatched shapes raised RuntimeError: {e}")
        return True

def verify_dimensions_of_size_1():
    print("Verifying dimensions of size 1...")
    try:
        gemm = ZeroSparseGEMM(in_features=1, out_features=8)
        x = torch.randn(2, 24, 1)
        out = gemm(x)
        assert out.shape == (2, 24, 8)
        print("PASS: dimensions of size 1")
        return True
    except Exception as e:
        print(f"FAIL: dimensions of size 1 raised {e}")
        return False

def verify_dynamic_config_failure():
    print("Verifying dynamic config / backend selector behavior...")
    try:
        gemm = ZeroSparseGEMM(8, 8, backend="M24308")
        assert gemm.backend == "M24308"
        print("PASS: backend parameter accepted successfully")
        return True
    except Exception as e:
        print(f"FAIL: backend selector failed: {e}")
        return False

def verify_extremely_large_shapes():
    print("Verifying extremely large shapes...")
    try:
        gemm = ZeroSparseGEMM(in_features=1024, out_features=1024)
        x = torch.randn(1, 24, 1024)
        out = gemm(x)
        assert out.shape == (1, 24, 1024)
        print("PASS: extremely large shapes")
        return True
    except Exception as e:
        print(f"FAIL: extremely large shapes raised {e}")
        return False

def verify_empty_config_behavior():
    print("Verifying empty config behavior...")
    try:
        gemm = ZeroSparseGEMM(8, 8)
        x = torch.randn(2, 24, 8)
        out = gemm(x)
        assert out.shape == (2, 24, 8)
        print("PASS: empty config behavior")
        return True
    except Exception as e:
        print(f"FAIL: empty config behavior raised {e}")
        return False

if __name__ == "__main__":
    results = {
        "mismatched_shapes": verify_mismatched_shapes(),
        "dimensions_of_size_1": verify_dimensions_of_size_1(),
        "dynamic_config_failure": verify_dynamic_config_failure(),
        "extremely_large_shapes": verify_extremely_large_shapes(),
        "empty_config": verify_empty_config_behavior(),
    }
    
    print("\n--- ZeroSparseGEMM Selector Boundaries Summary ---")
    all_passed = True
    for name, res in results.items():
        print(f"{name}: {'Passed' if res else 'Failed'}")
        if not res:
            all_passed = False
            
    if all_passed:
        print("All ZeroSparseGEMM Selector boundaries tests verified (with expected failures handled)!")
        sys.exit(0)
    else:
        sys.exit(1)
