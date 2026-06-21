import torch
import torch.nn as nn
import torch.nn.functional as F
from harmonic_ops import HarmonicAttention
import math
from typing import Tuple, Optional

# Load configuration once at module import to avoid repeated I/O in blocks
try:
    from harmonic_ops import load_config
    CONFIG = load_config()
except Exception:
    CONFIG = {}


class HarmonicTransformerBlock(nn.Module):
    """
    Single transformer block using HarmonicAttention and a feed-forward network.

    This block optionally uses a specialized sparse GEMM implementation when
    the repository optimization config enables specialized hardware.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, fixed_point: bool = False):
        super().__init__()
        self.fixed_point = fixed_point
        self.attn = HarmonicAttention(embed_dim, num_heads, fixed_point=fixed_point)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        specialized_hw = False
        try:
            opt_methods = CONFIG.get("optimization_methods", {})
            specialized_hw = opt_methods.get("6_specialized_hardware", {}).get("enabled", False)
        except Exception:
            specialized_hw = False

        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        if specialized_hw:
            from harmonic_gemm_sparse import ZeroSparseGEMM
            self.ff1 = ZeroSparseGEMM(embed_dim, ff_dim, fixed_point=fixed_point)
            self.ff2 = ZeroSparseGEMM(ff_dim, embed_dim, fixed_point=fixed_point)
            self.dropout_ff = nn.Dropout(dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer block.

        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor of same shape after attention + feed-forward with residual connections.
        """
        # Attention with Residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Feed Forward with Residual
        if hasattr(self, 'ff1'):
            x_ff = self.gelu(self.ff1(self.norm2(x)))
            x_ff = self.dropout_ff(x_ff)
            x_ff = self.ff2(x_ff)
            x = x + self.dropout(x_ff)
        else:
            x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class ResonantPositionalEncoding(nn.Module):
    """
    Positional Encoding locked to the 24-step Harmonic Lattice.
    Tokens on the same 'Gate' mod 24 share a fundamental phase alignment.

    The positional encoding is registered as a buffer so it will move with the
    model to the target device. We still ensure in forward that the buffer is
    used on the same device as the input tensor.
    """

    def __init__(self, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Create base sinusoidal positional encodings on CPU; register as buffer
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # The 'Harmonic Step' - ensures frequencies are multiples of the icositetragon cycle
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        # Apply sine and cosine to alternate embedding dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # The 'Resonance Factor': amplify positions that land on 'Open Gates'
        from harmonic_constants import IS_OPEN_GATE
        for i in range(max_seq_len):
            if IS_OPEN_GATE[i % 24]:
                pe[i, :] *= 1.5  # Boost the 'Radiation' signal for prime-aligned positions
            else:
                pe[i, :] *= 0.5  # Muffle the 'Void' signal

        # Register as buffer so it moves with .to(device) calls
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Ensures the positional encoding buffer is on the same device as x.
        """
        return x + self.pe[:, :x.size(1)].to(x.device)


class HarmonicTransformer(nn.Module):
    """
    A Transformer architecture where every attention layer is restricted
    to the 'Open Gates' of the Icositetragon, now with Resonant Positional Encoding.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int, max_seq_len: int, fixed_point: bool = False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.rpe = ResonantPositionalEncoding(embed_dim, max_seq_len)

        self.layers = nn.ModuleList([
            HarmonicTransformerBlock(embed_dim, num_heads, ff_dim, fixed_point=fixed_point)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Harmonic Transformer.

        Args:
            x: Long tensor of shape (batch, seq_len) containing token indices.

        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        b, t = x.shape
        x = self.token_emb(x)
        # Apply Resonant Positional Encoding
        x = self.rpe(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)


def generate_synthetic_data(num_samples: int, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates sequences where the target is based on modular arithmetic.

    Returns:
        data: Tensor[num_samples, seq_len]
        targets: Tensor[num_samples, seq_len]
    """
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Target: Predict (n + 1) mod vocab_size, but influenced by harmonic position
    targets = (data + 1) % vocab_size
    return data, targets


if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 100
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_DIM = 128
    MAX_SEQ_LEN = 24  # One full Grand Cycle

    model = HarmonicTransformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN)

    # Create a dummy batch
    x, y = generate_synthetic_data(4, MAX_SEQ_LEN, VOCAB_SIZE)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

    print("Harmonic Transformer Initialized.")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial Cross-Entropy Loss: {loss.item():.4f}")

    # Verify it can run a backward pass
    try:
        loss.backward()
        print("Backward pass successful. Gradients calculated through harmonic gates.")
    except Exception as e:
        print("Backward pass failed:", e)
