import sys
import json
import torch

try:
    import harmonic_ops
    import harmonic_gemm_sparse
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def verify():
    # Instantiate HarmonicAttention
    # Using small dimensions to avoid overhead
    attn = harmonic_ops.HarmonicAttention(embed_dim=8, num_heads=1)
    
    # Print the relevant configuration values to stdout
    print(f"base_frequency: {attn.base_frequency}")
    print(f"strict_rules_enabled: {attn.strict_rules_enabled}")
    
    # Instantiate ZeroSparseGEMM
    gemm = harmonic_gemm_sparse.ZeroSparseGEMM(in_features=8, out_features=8)
    print(f"gemm_config_keys: {list(gemm.config.keys())}")
    
    # Check if a custom parameter we injected is visible in the config
    custom_value = gemm.config.get("custom_e2e_test_key", None)
    print(f"custom_e2e_test_key: {custom_value}")
    
    # Exit cleanly
    sys.exit(0)

if __name__ == "__main__":
    verify()
