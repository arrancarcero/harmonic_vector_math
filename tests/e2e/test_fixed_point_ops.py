import pytest
import torch
from harmonic_ops import FixedPointTensor, HarmonicAttention
from harmonic_gemm_sparse import ZeroSparseGEMM

def test_fixed_point_tensor_instantiation():
    """Verify FixedPointTensor instantiates correctly from lists, float tensors, and integer tensors."""
    fp1 = FixedPointTensor([0.1, 0.2, 0.3], scale=100)
    assert fp1.scale == 100
    assert torch.equal(fp1.tensor, torch.tensor([10, 20, 30], dtype=torch.int64))
    
    fp2 = FixedPointTensor(torch.tensor([1.2345, 2.3456]), scale=1000)
    assert fp2.scale == 1000
    assert torch.equal(fp2.tensor, torch.tensor([1235, 2346], dtype=torch.int64))
    
    t_int = torch.tensor([5, 6, 7], dtype=torch.int64)
    fp3 = FixedPointTensor(t_int, scale=50)
    assert fp3.scale == 50
    assert torch.equal(fp3.tensor, t_int)

def test_scaling_conversion_correctness():
    """Verify that converting FixedPointTensor back to float produces values within mathematical tolerance (1/scale)."""
    data = torch.randn(5, 5)
    scale = 10000
    fp = FixedPointTensor(data, scale=scale)
    data_back = fp.to_float()
    
    max_err = torch.max(torch.abs(data - data_back)).item()
    assert max_err <= 1.0 / scale

def test_attention_quantization_scaling():
    """Verify that HarmonicAttention performs quantized attention under fixed_point=True without runtime errors."""
    attn = HarmonicAttention(embed_dim=8, num_heads=1, fixed_point=True)
    x = torch.randn(1, 24, 8)
    out = attn(x)
    assert out.shape == (1, 24, 8)

def test_gemm_quantization_scaling():
    """Verify that ZeroSparseGEMM performs quantized matmul under fixed_point=True without runtime errors."""
    gemm = ZeroSparseGEMM(8, 8, fixed_point=True)
    x = torch.randn(1, 24, 8)
    out = gemm(x)
    assert out.shape == (1, 24, 8)

def test_deviation_limit_check():
    """Verify that fixed-point operations deviate from the float baseline by no more than 0.20."""
    attn_float = HarmonicAttention(embed_dim=16, num_heads=2, fixed_point=False)
    attn_fixed = HarmonicAttention(embed_dim=16, num_heads=2, fixed_point=True)
    attn_fixed.load_state_dict(attn_float.state_dict())
    
    x = torch.randn(1, 24, 16)
    out_float = attn_float(x)
    out_fixed = attn_fixed(x)
    
    dev_attn = torch.max(torch.abs(out_float - out_fixed)).item()
    assert dev_attn <= 0.20, f"Attention deviation {dev_attn} exceeds 0.20 limit!"
    
    gemm_float = ZeroSparseGEMM(16, 16, fixed_point=False)
    gemm_fixed = ZeroSparseGEMM(16, 16, fixed_point=True)
    gemm_fixed.load_state_dict(gemm_float.state_dict())
    
    x_gemm = torch.randn(1, 24, 16)
    out_gemm_float = gemm_float(x_gemm)
    out_gemm_fixed = gemm_fixed(x_gemm)
    
    dev_gemm = torch.max(torch.abs(out_gemm_float - out_gemm_fixed)).item()
    assert dev_gemm <= 0.20, f"GEMM deviation {dev_gemm} exceeds 0.20 limit!"

def test_fixed_point_ops_subprocess(subprocess_runner):
    """Run the verify_fixed_point_ops.py script in a subprocess to check execution flow."""
    result = subprocess_runner("tests/e2e/scripts/verify_fixed_point_ops.py")
    assert result.returncode == 0
    assert "All Fixed-Point Operations test cases passed!" in result.stdout
