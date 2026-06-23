import torch
import pytest

from harmonic_gemm_sparse import ZeroSparseGEMM, FullHarmonicTransformerLayer


def test_zero_sparse_gemm_padding():
    batch = 2
    seq_len = 5
    in_features = 6
    out_features = 10  # not divisible by 8

    module = ZeroSparseGEMM(in_features, out_features)
    x = torch.randn(batch, seq_len, in_features)
    out = module(x)
    assert out.shape == (batch, seq_len, out_features)


def test_full_layer_partial_blocks():
    batch = 1
    seq_len = 30  # not multiple of 24
    dim = 32
    heads = 4

    layer = FullHarmonicTransformerLayer(dim, heads)
    x = torch.randn(batch, seq_len, dim)
    out = layer(x)
    assert out.shape == (batch, seq_len, dim)
