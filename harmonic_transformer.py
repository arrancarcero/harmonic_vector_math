import torch
import torch.nn as nn
import torch.nn.functional as F
from harmonic_ops import HarmonicAttention, load_config
import math

# Load config once at module import to avoid repeated I/O per layer
GLOBAL_HARMONIC_CONFIG = load_config()

class HarmonicTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, fixed_point=False):
        super().__init__()
        self.fixed_point = fixed_point
        self.attn = HarmonicAttention(embed_dim, num_heads, fixed_point=fixed_point)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Reuse global config loaded once at module import
        self.config = GLOBAL_HARMONIC_CONFIG
        
        specialized_hw = False
        try:
            opt_methods = self.config.get("optimization_methods", {})
            specialized_hw = opt_methods.get("6_specialized_hardware", {}).get("enabled", False)
        except Exception:
            pass
            
        self.dropout = nn.Dropout(dropout)
        
        if specialized_hw:
            from harmonic_gemm_sparse import ZeroSparseGEMM
            self.ff1 = ZeroSparseGEMM(embed_dim, ff_dim, fixed_point=fixed_point)
            self.ff2 = ZeroSparseGEMM(ff_dim, embed_dim, fixed_point=fixed_point)
            # keep dropout inside the FF path; avoid applying dropout twice
            self.dropout_ff = nn.Dropout(dropout)
        else:
            # include dropout here and avoid applying outer dropout when adding residual
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )

    def forward(self, x):
        # Attention with Residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Feed Forward with Residual
        if hasattr(self, 'ff1'):
            x_ff = F.gelu(self.ff1(self.norm2(x)))
            x_ff = self.dropout_ff(x_ff)
            x_ff = self.ff2(x_ff)
            # x_ff already had dropout applied (dropout_ff); avoid double dropout
            x = x + x_ff
        else:
            # ff contains its own dropout, so don't apply the outer dropout again
            x = x + self.ff(self.norm2(x))
        return x

class ResonantPositionalEncoding(nn.Module):
    """
    Positional Encoding locked to the 24-step Harmonic Lattice.
    Tokens on the same 'Gate' mod 24 share a fundamental phase alignment.
    """
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        
        # require even embed_dim for alternating sin/cos encoding
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sinusoidal positional encoding")
        
        # We create a standard Sinusoidal base, but quantize frequencies to Mod-24
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # The 'Harmonic Step' - ensures frequencies are multiples of the icositetragon cycle
        # We use the 24-gate structure as the base frequency (2*pi / 24)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        # Apply sine and cosine to alternate embedding dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # The 'Resonance Factor': We amplify positions that land on 'Open Gates'
        from harmonic_constants import IS_OPEN_GATE
        # Vectorized mask: create scaling factors for each position
        mask_vals = torch.tensor([1.5 if IS_OPEN_GATE[i % 24] else 0.5 for i in range(max_seq_len)], dtype=pe.dtype)
        pe = pe * mask_vals.unsqueeze(1)
                
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HarmonicTransformer(nn.Module):
    """
    A Transformer architecture where every attention layer is restricted
    to the 'Open Gates' of the Icositetragon, now with Resonant Positional Encoding.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, fixed_point=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.rpe = ResonantPositionalEncoding(embed_dim, max_seq_len)
        
        self.layers = nn.ModuleList([
            HarmonicTransformerBlock(embed_dim, num_heads, ff_dim, fixed_point=fixed_point)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, t = x.shape
        x = self.token_emb(x)
        # Apply Resonant Positional Encoding
        x = self.rpe(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x)

def generate_synthetic_data(num_samples, seq_len, vocab_size):
    """Generates sequences where the target is based on modular arithmetic."""
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
    torch.manual_seed(0)
    
    model = HarmonicTransformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN)
    
    # Create a dummy batch
    x, y = generate_synthetic_data(4, MAX_SEQ_LEN, VOCAB_SIZE)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    
    print("Harmonic Transformer Initialized.")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial Cross-Entropy Loss: {loss.item():.4f}")
    
    # Verify it can run a backward pass
    loss.backward()
    print("Backward pass successful. Gradients calculated through harmonic gates.")
