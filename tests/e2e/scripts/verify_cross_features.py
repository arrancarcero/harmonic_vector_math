import sys
import torch
import math
import os
import json
from harmonic_ops import HarmonicAttention, FixedPointTensor
from harmonic_gemm_sparse import ZeroSparseGEMM

def verify_franklin_drift_fixed_point_attention():
    print("Running test case 1: Franklin Constant Drift + Fixed-Point in HarmonicAttention (F1, F5)")
    attn = HarmonicAttention(embed_dim=8, num_heads=1, fixed_point=True)
    
    assert hasattr(attn, "franklin_freq"), "franklin_freq attribute is missing"
    assert attn.franklin_freq == 8.125, f"Expected 8.125, got {attn.franklin_freq}"
    assert attn.wobble_amplitude == 0.125, f"Expected 0.125, got {attn.wobble_amplitude}"
    
    x = torch.randn(1, 24, 8)
    out1 = attn(x)
    assert out1.shape == (1, 24, 8), f"Expected shape (1, 24, 8), got {out1.shape}"
    
    # Check interaction: altering base frequency dynamically scales Franklin drift, which affects fixed-point scores
    attn.base_frequency = 100.0
    out2 = attn(x)
    
    diff = torch.max(torch.abs(out1 - out2)).item()
    assert diff > 0.0, "Changing base frequency did not change attention output"
    print(f"Franklin Drift influence verified under fixed-point. Output deviation: {diff:.6f}")
    return True

def verify_bzsgemm_m24308_exhaust_fixed_point():
    print("Running test case 2: BZS-GEMM M24308 Mapping + Exhaust Stats + Fixed-Point in ZeroSparseGEMM (F3, F4, F5)")
    
    try:
        gemm = ZeroSparseGEMM(16, 16, fixed_point=True, backend="M24308")
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): ZeroSparseGEMM raised {e}")
        return False
        
    assert gemm.fixed_point is True, "fixed_point should be True"
    assert gemm.backend == "M24308", f"Expected backend M24308, got {gemm.backend}"
    
    # Input has non-zero values on slots that should be masked by M24308 (mod 16: index 0 and 14)
    x = torch.randn(1, 24, 16)
    x[:, :, 0] = 99.0
    x[:, :, 14] = -99.0
    
    out1 = gemm(x)
    
    # Verify M24308 Mapping (F3): altering the masked input slots has no effect on output
    x_altered = x.clone()
    x_altered[:, :, 0] = 50.0
    x_altered[:, :, 14] = -50.0
    out2 = gemm(x_altered)
    
    diff = torch.max(torch.abs(out1 - out2)).item()
    assert diff == 0.0, f"M24308 mapping failed: altering masked input changed output by {diff}"
    
    # Verify Dual-Stream Exhaust Stats (F4): exhaust_energy is computed and non-zero
    assert gemm.exhaust_energy > 0.0, f"Exhaust energy should be positive, got {gemm.exhaust_energy}"
    print(f"Exhaust energy recorded: {gemm.exhaust_energy:.4f}")
    
    # Verify Fixed-Point Operations (F5): compare deviation with float baseline (<= 0.20)
    gemm_float = ZeroSparseGEMM(16, 16, fixed_point=False, backend="M24308")
    gemm_float.load_state_dict(gemm.state_dict())
    out_float = gemm_float(x)
    
    dev = torch.max(torch.abs(out1 - out_float)).item()
    assert dev <= 0.20, f"Fixed-point GEMM output deviated from float baseline by {dev:.4f}"
    print(f"Fixed-point vs float deviation verified: {dev:.4f}")
    return True

def verify_gemm_selector_m24308_mapping():
    print("Running test case 3: ZeroSparseGEMM Selector + M24308 Mapping (F2, F3)")
    
    try:
        gemm_generic = ZeroSparseGEMM(16, 16, backend="generic")
        gemm_m24308 = ZeroSparseGEMM(16, 16, backend="M24308")
    except TypeError as e:
        print(f"FAIL (Expected due to unimplemented selector): ZeroSparseGEMM raised {e}")
        return False
        
    assert gemm_generic.backend == "generic"
    assert gemm_m24308.backend == "M24308"
    
    x = torch.randn(1, 24, 16)
    x[:, :, 0] = 100.0
    
    out_generic1 = gemm_generic(x)
    out_m243081 = gemm_m24308(x)
    
    x_altered = x.clone()
    x_altered[:, :, 0] = 0.0
    
    out_generic2 = gemm_generic(x_altered)
    out_m243082 = gemm_m24308(x_altered)
    
    diff_generic = torch.max(torch.abs(out_generic1 - out_generic2)).item()
    diff_m24308 = torch.max(torch.abs(out_m243081 - out_m243082)).item()
    
    # Generic should not mask slot 0, so changing it changes output.
    # M24308 should mask slot 0, so changing it has no effect.
    assert diff_generic > 0.0, "Generic backend should not mask slot 0"
    assert diff_m24308 == 0.0, "M24308 backend failed to mask slot 0"
    
    # Generic does not compute exhaust energy (keeps at 0.0).
    # M24308 computes exhaust energy.
    assert gemm_generic.exhaust_energy == 0.0, f"Generic backend recorded non-zero exhaust energy: {gemm_generic.exhaust_energy}"
    assert gemm_m24308.exhaust_energy > 0.0, "M24308 backend failed to record exhaust energy"
    
    print("Selector successfully controls M24308 layout mapping application.")
    return True

def verify_drift_selector_config_sync():
    print("Running test case 4: Franklin Constant Drift + ZeroSparseGEMM Selector Config Sync (F1, F2)")
    
    config_path = "isf_optimization_config.json"
    backup_path = config_path + ".verify_backup"
    
    backup_created = False
    if os.path.exists(config_path):
        os.rename(config_path, backup_path)
        backup_created = True
        
    try:
        custom_config = {
            "hardware_heritage": {
                "base_frequency_hz": 999.0
            },
            "optimization_methods": {
                "6_specialized_hardware": {
                    "matrix_engine": "M24308"
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(custom_config, f, indent=2)
            
        attn = HarmonicAttention(embed_dim=8, num_heads=1)
        try:
            gemm = ZeroSparseGEMM(8, 8, backend="auto")
        except TypeError as e:
            print(f"FAIL (Expected due to unimplemented selector): ZeroSparseGEMM raised {e}")
            return False
            
        assert attn.base_frequency == 999.0, f"Expected base frequency to be 999.0, got {attn.base_frequency}"
        assert gemm.backend == "M24308", f"Expected backend to be M24308, got {gemm.backend}"
        
        x = torch.randn(1, 24, 8)
        attn_out = attn(x)
        gemm_out = gemm(attn_out)
        assert gemm_out.shape == (1, 24, 8), f"Expected shape (1, 24, 8), got {gemm_out.shape}"
        print("Config synchronization verified successfully.")
        return True
        
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        if backup_created:
            os.rename(backup_path, config_path)

def verify_sequential_flow():
    print("Running test case 5: Sequential Flow (F1, F2, F3, F4, F5)")
    
    config_path = "isf_optimization_config.json"
    backup_path = config_path + ".verify_backup"
    
    backup_created = False
    if os.path.exists(config_path):
        os.rename(config_path, backup_path)
        backup_created = True
        
    try:
        custom_config = {
            "hardware_heritage": {
                "base_frequency_hz": 500.0
            },
            "optimization_methods": {
                "6_specialized_hardware": {
                    "matrix_engine": "M24308"
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(custom_config, f, indent=2)
            
        attn = HarmonicAttention(embed_dim=16, num_heads=2, fixed_point=True)
        try:
            gemm = ZeroSparseGEMM(16, 16, fixed_point=True, backend="auto")
        except TypeError as e:
            print(f"FAIL (Expected due to unimplemented selector): ZeroSparseGEMM raised {e}")
            return False
            
        assert attn.fixed_point is True
        assert gemm.fixed_point is True
        assert attn.base_frequency == 500.0
        assert gemm.backend == "M24308"
        
        x = torch.randn(1, 24, 16)
        x[:, :, 0] = 50.0  # Put large value on M24308 masked slot
        
        # Pass through attention
        attn_out = x + attn(x)
        assert attn_out.shape == (1, 24, 16)
        
        # Pass through GEMM
        gemm_out = gemm(attn_out)
        assert gemm_out.shape == (1, 24, 16)
        
        # Verify internal metrics
        assert gemm.exhaust_energy > 0.0, f"Exhaust energy should be positive, got {gemm.exhaust_energy}"
        print(f"Sequential flow verified. Output shape: {gemm_out.shape}, GEMM exhaust energy: {gemm.exhaust_energy:.4f}")
        return True
        
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        if backup_created:
            os.rename(backup_path, config_path)

if __name__ == "__main__":
    results = []
    try:
        results.append(verify_franklin_drift_fixed_point_attention())
        results.append(verify_bzsgemm_m24308_exhaust_fixed_point())
        results.append(verify_gemm_selector_m24308_mapping())
        results.append(verify_drift_selector_config_sync())
        results.append(verify_sequential_flow())
        
        if all(results):
            print("All Cross-Feature Interaction test cases passed!")
            sys.exit(0)
        else:
            print("Some Cross-Feature Interaction test cases failed.")
            sys.exit(1)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Exception: {e}")
        sys.exit(1)
