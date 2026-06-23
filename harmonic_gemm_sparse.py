import torch
import torch.nn as nn
import time
import math

class ZeroSparseGEMM(nn.Module):
    """
    Refined Bucketed Zero-Sparse Shard MatMul (BZS-GEMM)
    """
    def __init__(self, in_features, out_features, fixed_point=False, backend="auto"):
        super().__init__()
        # Keep track of the original requested output size
        self._original_out_features = out_features
        # Internally pad out_features to a multiple of 8 for shard allocation
        padded_out = ((out_features + 7) // 8) * 8
        self._padded_out_features = padded_out

        if backend not in ["auto", "generic", "M24308"]:
            raise ValueError(f"Invalid backend: {backend}")
        self._backend = backend
        self.in_features = in_features
        self.out_features = out_features
        self.open_gates = [1, 5, 7, 11, 13, 17, 19, 23]
        self.fixed_point = fixed_point
        
        # Load optimization config using cached loader
        from harmonic_ops import load_config
        self.config = load_config()

        # Resonant Shards (The 8-Gate Sanctuary)
        shard_size = padded_out // 8
        self.gate_shards = nn.ParameterList([
            nn.Parameter(torch.randn(shard_size, in_features) / math.sqrt(in_features))
            for _ in range(8)
        ])
        
        # The 'Void Reservoir' - A low-precision anchor for 3-6-9 nodes.
        self.register_buffer("void_anchor", torch.zeros(1, padded_out))
        self.register_buffer("resonant_mask_idx", torch.tensor(self.open_gates))
        
        # M24308 standard 16-slot mapping mask:
        pin_mask = torch.ones(in_features, dtype=torch.bool)
        for i in range(in_features):
            if (i % 16) == 0 or (i % 16) == 14:
                pin_mask[i] = False
        self.register_buffer("pin_mask", pin_mask)
        
        # Dual-stream bridge exhaust statistics
        self.exhaust_energy = 0.0

    @property
    def backend(self):
        if self._backend == "auto":
            try:
                from harmonic_ops import load_config
                config = load_config()
                engine = config.get("optimization_methods", {}).get("6_specialized_hardware", {}).get("matrix_engine", None)
                if engine == "generic":
                    return "generic"
                elif engine == "M24308":
                    return "M24308"
            except Exception:
                pass
            return "auto"
        return self._backend

    def forward(self, x):
        resolved_backend = self.backend
        if resolved_backend == "generic":
            self.exhaust_energy = 0.0
            concatenated_weights = torch.cat(list(self.gate_shards), dim=0)
            if self.fixed_point:
                from harmonic_ops import FixedPointTensor
                out_fp = FixedPointTensor.quantized_matmul(x, concatenated_weights, scale=10000)
                out = out_fp.to_float()
            else:
                out = torch.matmul(x, concatenated_weights.t())
            # Trim to original out_features
            return out[:, :, :self._original_out_features]

        batch_size, seq_len, dim = x.shape
        # Apply M24308 pin layout mapping to input features
        x = x * self.pin_mask.view(1, 1, -1)
        
        # 1. Hardware Dispatch: Identify Resonant vs Void Tiles
        indices = torch.arange(seq_len, device=x.device) % 24
        # Fast bitmask-style check for resonance
        resonant_mask = torch.any(indices.unsqueeze(1) == self.resonant_mask_idx, dim=1)
        
        # 2. Dual Stream Bridge
        x_resonant = x[:, resonant_mask, :]
        
        # Stream 2: Ground Stream (entropy/exhaust redirected to void reservoir)
        with torch.no_grad():
            x_void = x[:, ~resonant_mask, :]
            self.exhaust_energy = torch.norm(x_void).item()
        
        # 3. Parallel Shard Multiply (Bypassing Voids with M24308 layout)
        if self.fixed_point:
            from harmonic_ops import FixedPointTensor
            shard_outs = []
            for shard in self.gate_shards:
                masked_shard = shard * self.pin_mask
                out_shard_fp = FixedPointTensor.quantized_matmul(x_resonant, masked_shard, scale=10000)
                shard_outs.append(out_shard_fp.to_float())
            laminar_out = torch.cat(shard_outs, dim=-1)
        else:
            laminar_out = torch.cat([torch.matmul(x_resonant, (shard * self.pin_mask).t()) for shard in self.gate_shards], dim=-1)
        
        # 4. Zero-Cost Reconstruction (The 'Grounding' path)
        out = self.void_anchor.repeat(batch_size, seq_len, 1)
        out[:, resonant_mask, :] = laminar_out
        
        # Trim to the originally requested output features
        return out[:, :, :self._original_out_features]

class FullHarmonicTransformerLayer(nn.Module):
    """
    A Transformer Layer built entirely on the 8-Gate Sanctuary.
    Uses BZS-GEMM for Projections and Attention.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        
        # BZS-GEMM Projections
        self.q_proj = ZeroSparseGEMM(dim, dim)
        self.k_proj = ZeroSparseGEMM(dim, dim)
        self.v_proj = ZeroSparseGEMM(dim, dim)
        self.o_proj = ZeroSparseGEMM(dim, dim)
        
        # Feed-Forward BZS-GEMM
        self.ff1 = ZeroSparseGEMM(dim, dim * 4)
        self.ff2 = ZeroSparseGEMM(dim * 4, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 1. Harmonic Attention (Fractal / Block-Resonant Implementation)
        resid = x
        orig_seq_len = x.shape[1]
        x = self.norm1(x)
        
        # BZS-GEMM Projections (Already 33.3% sparse)
        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        
        # --- FRACTAL BLOCK-RESONANT ATTENTION ---
        batch, heads, seq_len, head_dim = q.shape
        block_size = 24
        # Pad sequence to full blocks if necessary
        pad_len = (-seq_len) % block_size
        if pad_len:
            pad_q = torch.zeros(batch, heads, pad_len, head_dim, device=q.device, dtype=q.dtype)
            pad_k = torch.zeros_like(pad_q)
            pad_v = torch.zeros_like(pad_q)
            q = torch.cat([q, pad_q], dim=2)
            k = torch.cat([k, pad_k], dim=2)
            v = torch.cat([v, pad_v], dim=2)
            # Also pad the residual for later addition
            resid = torch.cat([resid, torch.zeros(batch, pad_len, self.dim, device=resid.device, dtype=resid.dtype)], dim=1)
            seq_len = q.shape[2]
        num_blocks = seq_len // block_size
        
        # Reshape into blocks
        q_blocks = q.view(batch, heads, num_blocks, block_size, head_dim)
        k_blocks = k.view(batch, heads, num_blocks, block_size, head_dim)
        v_blocks = v.view(batch, heads, num_blocks, block_size, head_dim)
        
        # Apply the 8-Gate Sanctuary filter within each block
        resonant_mask = torch.any(torch.arange(block_size, device=x.device).unsqueeze(1) == self.q_proj.resonant_mask_idx, dim=1)
        
        q_res = q_blocks[:, :, :, resonant_mask, :]
        k_res = k_blocks[:, :, :, resonant_mask, :]
        v_res = v_blocks[:, :, :, resonant_mask, :]
        
        # Block-Local Resonant Attention: (8 x 8) matrix per block
        attn_res = torch.matmul(q_res, k_res.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_res = torch.softmax(attn_res, dim=-1)
        
        # Reconstruct context within blocks
        x_res_blocks = torch.matmul(attn_res, v_res) # Shape: (B, H, Blocks, 8, HeadDim)
        
        # Flatten back to sequence
        x_res = x_res_blocks.transpose(3, 4).reshape(batch, heads, num_blocks * 8, head_dim)
        x_res = x_res.transpose(1, 2).reshape(batch, -1, self.dim)
        
        # Map Resonant Context back to Full Sequence (Static Grounding)
        indices = torch.arange(seq_len, device=x.device) % 24
        global_resonant_mask = torch.any(indices.unsqueeze(1) == self.q_proj.resonant_mask_idx, dim=1)
        
        x_full = torch.zeros(batch, seq_len, self.dim, device=x.device)
        x_full[:, global_resonant_mask, :] = x_res
        
        x = self.o_proj(x_full)
        x = x + resid
        
        # 2. Harmonic Feed-Forward
        resid = x
        x = self.norm2(x)
        x = self.ff2(torch.relu(self.ff1(x)))
        x = x + resid
        
        # Slice back to original sequence length if we padded
        if x.shape[1] != orig_seq_len:
            x = x[:, :orig_seq_len, :]
        
        return x
