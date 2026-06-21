import pytest
import torch
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_mismatched_shapes():
    """Verify that passing inputs with mismatched feature dimensions raises a RuntimeError."""
    gemm = ZeroSparseGEMM(in_features=8, out_features=16)
    x = torch.randn(2, 24, 9) # mismatched dimension 9 vs 8
    with pytest.raises(RuntimeError):
        gemm(x)

def test_dimensions_of_size_1():
    """Verify behaviour with dimensions of size 1. E.g. in_features=1, out_features=8."""
    gemm = ZeroSparseGEMM(in_features=1, out_features=8)
    x = torch.randn(2, 24, 1)
    out = gemm(x)
    assert out.shape == (2, 24, 8)
    assert not torch.isnan(out).any()

def test_dynamically_changed_config_during_runtime(override_config):
    """Verify behavior when config changes during runtime."""
    custom_config = {
        "optimization_methods": {
            "6_specialized_hardware": {
                "enabled": True,
                "matrix_engine": "M24308"
            }
        }
    }
    override_config(custom_config)
    
    gemm = ZeroSparseGEMM(8, 8, backend="M24308")
    assert gemm.backend == "M24308"

def test_extremely_large_shapes():
    """Verify that instantiating and running with extremely large dimensions works without crash."""
    gemm = ZeroSparseGEMM(in_features=1024, out_features=1024)
    x = torch.randn(1, 24, 1024)
    out = gemm(x)
    assert out.shape == (1, 24, 1024)

def test_empty_config_behavior(override_config):
    """Verify behaviour when optimization config is empty or missing."""
    override_config({})
    gemm = ZeroSparseGEMM(8, 8)
    assert gemm.config == {}
    x = torch.randn(2, 24, 8)
    out = gemm(x)
    assert out.shape == (2, 24, 8)

def test_gemm_selector_boundaries_subprocess(subprocess_runner):
    """Run the verify_gemm_selector_boundaries.py script in a subprocess."""
    result = subprocess_runner("tests/e2e/scripts/verify_gemm_selector_boundaries.py")
    assert result.returncode == 0
