import sys
import torch
from harmonic_ops import FixedPointTensor

def verify_scale_extremes():
    print("Verifying scale extremes...")
    try:
        fp_small = FixedPointTensor(1.7, scale=1)
        assert fp_small.tensor.item() == 2
        
        fp_large = FixedPointTensor(1.23456789, scale=100000000)
        assert fp_large.tensor.item() == 123456789
        print("PASS: scale extremes")
        return True
    except Exception as e:
        print(f"FAIL: scale extremes raised {e}")
        return False

def verify_rounding_boundary():
    print("Verifying rounding boundary...")
    try:
        fp1 = FixedPointTensor(0.005, scale=100)
        assert fp1.tensor.item() == 0
        
        fp2 = FixedPointTensor(0.015, scale=100)
        assert fp2.tensor.item() == 2
        print("PASS: rounding boundary")
        return True
    except Exception as e:
        print(f"FAIL: rounding boundary raised {e}")
        return False

def verify_int64_overflow():
    print("Verifying int64 overflow...")
    try:
        huge_val = 9e18
        fp = FixedPointTensor(huge_val, scale=1)
        assert fp.tensor.item() is not None
        print("PASS: int64 overflow")
        return True
    except Exception as e:
        print(f"FAIL: int64 overflow raised {e}")
        return False

def verify_zero_tensors():
    print("Verifying zero tensors...")
    try:
        x = torch.zeros(2, 3)
        fp = FixedPointTensor(x, scale=100)
        assert torch.all(fp.tensor == 0)
        assert torch.all(fp.to_float() == 0.0)
        print("PASS: zero tensors")
        return True
    except Exception as e:
        print(f"FAIL: zero tensors raised {e}")
        return False

def verify_near_zero_tensors():
    print("Verifying near zero tensors...")
    try:
        x = torch.tensor([0.004])
        fp = FixedPointTensor(x, scale=100)
        assert fp.tensor.item() == 0
        print("PASS: near zero tensors")
        return True
    except Exception as e:
        print(f"FAIL: near zero tensors raised {e}")
        return False

if __name__ == "__main__":
    results = {
        "scale_extremes": verify_scale_extremes(),
        "rounding_boundary": verify_rounding_boundary(),
        "int64_overflow": verify_int64_overflow(),
        "zero_tensors": verify_zero_tensors(),
        "near_zero_tensors": verify_near_zero_tensors(),
    }
    
    print("\n--- Fixed-Point Operations Boundaries Summary ---")
    all_passed = True
    for name, res in results.items():
        print(f"{name}: {'Passed' if res else 'Failed'}")
        if not res:
            all_passed = False
            
    if all_passed:
        print("All Fixed-Point Operations boundaries verified successfully!")
        sys.exit(0)
    else:
        sys.exit(1)
