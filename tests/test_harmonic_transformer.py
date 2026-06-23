import torch
import pytest
import torch.nn.functional as F

from harmonic_transformer import ResonantPositionalEncoding, HarmonicTransformer


def test_resonant_pe_requires_even_embed_dim():
    with pytest.raises(ValueError):
        ResonantPositionalEncoding(33, 24)


def test_model_forward_and_backward():
    torch.manual_seed(0)
    vocab_size = 16
    embed_dim = 32
    num_heads = 4
    num_layers = 1
    ff_dim = 64
    max_seq_len = 24

    model = HarmonicTransformer(vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len)
    x = torch.randint(0, vocab_size, (2, max_seq_len))
    logits = model(x)
    assert logits.shape == (2, max_seq_len, vocab_size)

    targets = (x + 1) % vocab_size
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    # Ensure at least one parameter received gradients
    assert any(p.grad is not None for p in model.parameters())
