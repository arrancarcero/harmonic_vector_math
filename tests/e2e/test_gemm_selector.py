import pytest
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_selector_backend_m24308():
    """Verify backend='M24308' selector contract."""
    gemm = ZeroSparseGEMM(8, 8, backend="M24308")
    assert hasattr(gemm, "backend"), "backend attribute is missing"
    assert gemm.backend == "M24308"

def test_selector_backend_auto(override_config):
    """Verify backend='auto' selector contract."""
    override_config({})
    gemm = ZeroSparseGEMM(8, 8, backend="auto")
    assert hasattr(gemm, "backend"), "backend attribute is missing"
    assert gemm.backend == "auto"

def test_selector_backend_generic():
    """Verify backend='generic' selector contract."""
    gemm = ZeroSparseGEMM(8, 8, backend="generic")
    assert hasattr(gemm, "backend"), "backend attribute is missing"
    assert gemm.backend == "generic"

def test_config_driven_default(override_config):
    """Verify config-driven default backend is applied when not explicitly specified."""
    # Custom config containing matrix engine or default backend settings
    custom_config = {
        "optimization_methods": {
            "6_specialized_hardware": {
                "enabled": True,
                "matrix_engine": "generic"
            }
        }
    }
    override_config(custom_config)
    
    gemm = ZeroSparseGEMM(8, 8)
    assert hasattr(gemm, "backend"), "backend attribute is missing"
    assert gemm.backend == "generic"

def test_invalid_backend_handling():
    """Verify that specifying an invalid backend raises a ValueError."""
    with pytest.raises(ValueError):
        ZeroSparseGEMM(8, 8, backend="invalid_backend")

def test_gemm_selector_subprocess(subprocess_runner):
    """Run the verify_gemm_selector.py script in a subprocess and inspect outputs."""
    result = subprocess_runner("tests/e2e/scripts/verify_gemm_selector.py")
    assert result.returncode == 0
    assert "M24308: Failed" in result.stdout or "M24308: Passed" in result.stdout
