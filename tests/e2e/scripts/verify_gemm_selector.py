import sys
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_backend_m24308():
    try:
        gemm = ZeroSparseGEMM(8, 8, backend="M24308")
        print("PASS: backend='M24308'")
        return True
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): backend='M24308' raised {e}")
        return False

def test_backend_auto():
    try:
        gemm = ZeroSparseGEMM(8, 8, backend="auto")
        print("PASS: backend='auto'")
        return True
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): backend='auto' raised {e}")
        return False

def test_backend_generic():
    try:
        gemm = ZeroSparseGEMM(8, 8, backend="generic")
        print("PASS: backend='generic'")
        return True
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): backend='generic' raised {e}")
        return False

def test_config_driven_default():
    try:
        gemm = ZeroSparseGEMM(8, 8)
        if hasattr(gemm, "backend"):
            print(f"PASS: config-driven default backend is {gemm.backend}")
            return True
        else:
            print("FAIL: backend attribute not found on ZeroSparseGEMM")
            return False
    except Exception as e:
        print(f"FAIL: config-driven default check raised {e}")
        return False

def test_invalid_backend():
    try:
        # If the parameter is implemented, this should raise ValueError for invalid backend
        gemm = ZeroSparseGEMM(8, 8, backend="invalid_backend")
        print("FAIL: invalid backend did not raise exception")
        return False
    except ValueError as e:
        print(f"PASS: invalid backend raised ValueError: {e}")
        return True
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): invalid backend raised {e}")
        return False

if __name__ == "__main__":
    results = {
        "M24308": test_backend_m24308(),
        "auto": test_backend_auto(),
        "generic": test_backend_generic(),
        "config_default": test_config_driven_default(),
        "invalid_backend": test_invalid_backend()
    }
    
    print("\n--- Summary ---")
    for name, res in results.items():
        print(f"{name}: {'Passed' if res else 'Failed'}")
        
    # We exit with 0 to allow the execution to complete and show which parts failed/passed
    sys.exit(0)
