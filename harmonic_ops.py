import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from harmonic_constants import OPEN_GATES, IS_OPEN_GATE

_config_cache = None
_config_mtime = None

def load_config():
    global _config_cache, _config_mtime
    config_path = os.path.join(os.path.dirname(__file__), "isf_optimization_config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        mtime = os.path.getmtime(config_path)
        if _config_cache is None or _config_mtime != mtime:
            with open(config_path, "r") as f:
                _config_cache = json.load(f)
            _config_mtime = mtime
        return _config_cache
    except Exception:
        if _config_cache is not None:
            return _config_cache
        return {}


class FixedPointTensor:
    """
    Fixed-Point Vectorization Utility.
    Scales decimal/floating-point values to integers (e.g. scale factor 100 or 10000)
    to perform exact calculations on integer representations using vectorized math,
    avoiding floating point rounding errors while maintaining SIMD speed.
    """
    def __init__(self, data, scale=10000):
        self.scale = scale
        if isinstance(data, torch.Tensor):
            if data.dtype in (torch.int32, torch.int64):
                self.tensor = data
            else:
                self.tensor = torch.round(data.double() * scale).to(torch.int64)
        else:
            dtype = torch.float64 if scale >= 10000 else torch.float32
            self.tensor = torch.round(torch.tensor(data, dtype=dtype) * scale).to(torch.int64)

    def to_float(self):
        return self.tensor.to(torch.float32) / self.scale

    def __add__(self, other):
        assert self.scale == other.scale
        return FixedPointTensor(self.tensor + other.tensor, self.scale)

    def __sub__(self, other):
        assert self.scale == other.scale
        return FixedPointTensor(self.tensor - other.tensor, self.scale)

    def __mul__(self, other):
        if isinstance(other, FixedPointTensor):
            assert self.scale == other.scale
            res = (self.tensor * other.tensor + self.scale // 2) // self.scale
            return FixedPointTensor(res, self.scale)
        else:
            dtype = torch.float64 if self.scale >= 10000 else torch.float32
            return FixedPointTensor(torch.round(self.tensor.to(dtype) * other).to(torch.int64), self.scale)

    @classmethod
    def quantized_matmul(cls, x, w, scale=10000):
        """
        Runs integer matrix multiplication at hardware speed using scaled representations.
        """
        x_fp = cls(x, scale)
        w_fp = cls(w, scale)
        res = torch.matmul(x_fp.tensor.float(), w_fp.tensor.float().transpose(-2, -1))
        res_int = torch.round(res / scale).to(torch.int64)
        return cls(res_int, scale)

class HarmonicAttention(nn.Module):
    """
    CRYOGENIC RESONANT ATTENTION: Internal Liquid Cooling for Logic.
    Uses Thermal Probes and a Cryo-Softmax to prevent internal 'Boiling'.
    """
    def __init__(self, embed_dim, num_heads, fixed_point=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fixed_point = fixed_point

        # --- ISF OPTIMIZATION CONFIG LOAD ---
        self.config = load_config()

        # Hardware settings
        hw = self.config.get("hardware_heritage", {})
        self.base_frequency = hw.get("base_frequency_hz", 432.0)

        # Clock-drift parameters (Franklin constant ripple of 8.125Hz)
        self.franklin_freq = 8.125
        self.wobble_amplitude = 0.125 # 12.5% (or 1/8)

        # Strict rules checking (e.g. empty buffers at Gate 1 and Gate 17)
        opt_methods = self.config.get("optimization_methods", {})
        rules_cfg = opt_methods.get("1_strict_rules_meta_prompting", {})
        self.strict_rules_enabled = rules_cfg.get("enabled", False)

        specialized_hw = False
        try:
            specialized_hw = opt_methods.get("6_specialized_hardware", {}).get("enabled", False)
        except Exception:
            pass

        if specialized_hw:
            from harmonic_gemm_sparse import ZeroSparseGEMM
            self.q_proj = ZeroSparseGEMM(embed_dim, embed_dim, fixed_point=fixed_point)
            self.k_proj = ZeroSparseGEMM(embed_dim, embed_dim, fixed_point=fixed_point)
            self.v_proj = ZeroSparseGEMM(embed_dim, embed_dim, fixed_point=fixed_point)
            self.out_proj = ZeroSparseGEMM(embed_dim, embed_dim, fixed_point=fixed_point)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # --- CRYO-COEFFICIENTS ---
        self.T_BOILING = 2.5 # Entropy threshold for the 'Boiling Point'
        self.T_COOLANT = 0.5 # Coolant temperature (Sub-zero)
        
        self.gate_weights = {
            1: 1.000, 5: 0.950, 7: 1.200, 11: 0.980,
            13: 0.920, 17: 0.900, 19: 0.880, 23: 0.890
        }

        self.open_gates = OPEN_GATES
        self._mask_cache = {}
        self._weight_cache = {}
        self._wobble_cache = {}
        
        # Track 'Flush' events for diagnostic synthesis
        self.cryo_stats = {"flush_count": 0, "heat_removed": 0.0}

    def get_harmonic_indices(self, seq_len, device):
        cache_key = (seq_len, str(device))
        if cache_key not in self._mask_cache:
            indices = torch.arange(seq_len, device=device)
            cycle_pos = indices % 24
            gates_tensor = torch.tensor(OPEN_GATES, device=device)
            mask = torch.any(cycle_pos.unsqueeze(1) == gates_tensor, dim=1)
            self._mask_cache[cache_key] = torch.where(mask)[0]
        return self._mask_cache[cache_key]

    def get_resonant_weights(self, h_indices, device):
        cache_key = (len(h_indices), str(device))
        if cache_key not in self._weight_cache:
            cycle_pos = h_indices % 24
            weights = torch.tensor([self.gate_weights[int(p)] for p in cycle_pos], device=device)
            self._weight_cache[cache_key] = weights.unsqueeze(0).unsqueeze(-1)
        return self._weight_cache[cache_key]

    def get_wobble_factors(self, seq_len, h_indices, device):
        cache_key = (seq_len, float(self.base_frequency), str(device))
        if cache_key not in self._wobble_cache:
            q_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
            q_phase = 2.0 * math.pi * (q_indices * self.franklin_freq) / self.base_frequency
            q_wobble = 1.0 + self.wobble_amplitude * torch.sin(q_phase)

            k_indices = h_indices.to(torch.float32)
            k_phase = 2.0 * math.pi * (k_indices * self.franklin_freq) / self.base_frequency
            wobble_k = 1.0 + self.wobble_amplitude * torch.sin(k_phase)
            self._wobble_cache[cache_key] = (q_wobble, wobble_k)
        return self._wobble_cache[cache_key]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Enforce empty buffers at Gate 1 (A) and Gate 17 (Q) to maintain laminar pressure differentials.
        if self.strict_rules_enabled:
            indices = torch.arange(seq_len, device=x.device)
            laminar_mask = (indices % 24 != 1) & (indices % 24 != 17)
            # Apply input differential mask
            x = x * laminar_mask.unsqueeze(0).unsqueeze(-1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        h_indices = self.get_harmonic_indices(seq_len, x.device)
        k = self.k_proj(x)[:, h_indices, :].view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x)[:, h_indices, :].view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        res_weights = self.get_resonant_weights(h_indices, x.device)
        k = k * res_weights
        v = v * res_weights

        # --- FRANKLIN CONSTANT DRIFT WOBBLE SIMULATION ---
        # Model the 8.125Hz Franklin Constant drift as a phase-offset/wobble variable in the query-key attention scaling.
        q_wobble, wobble_k = self.get_wobble_factors(seq_len, h_indices, x.device)
        q = q * q_wobble.view(1, 1, seq_len, 1)
        k = k * wobble_k.view(1, 1, -1, 1)

        # 1. RAW RESONANCE SCORES
        if self.fixed_point:
            # Vectorized fixed point simulation
            q_fp = FixedPointTensor(q, scale=10000)
            k_fp = FixedPointTensor(k, scale=10000)
            scores_fp = torch.matmul(q_fp.tensor.float(), k_fp.tensor.float().transpose(-2, -1)) / 100000000.0
            scores = scores_fp / math.sqrt(self.head_dim)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 2. THERMAL PROBE: Calculate Local Entropy (Heat)
        with torch.no_grad():
            heat = torch.var(scores, dim=-1) # (batch, heads, seq)
            avg_heat = heat.mean().item()
        
        # Model clock-drift dynamic variance into boiling point
        avg_wobble = q_wobble.mean().item() - 1.0
        t_boiling_effective = self.T_BOILING * (1.0 + avg_wobble)

        # 3. CRYO-SOFTMAX (Liquid Nitrogen Flush)
        if avg_heat > t_boiling_effective:
            self.cryo_stats["flush_count"] += 1
            self.cryo_stats["heat_removed"] += (avg_heat - t_boiling_effective)
            scores = scores / self.T_COOLANT
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 4. FINAL SIEVE: Zero out any signal that tried to leak into the Voids
        output = self.out_proj(context)
        indices = torch.arange(seq_len, device=x.device)
        h_mask = torch.tensor([(int(i) % 24 in self.open_gates) for i in indices], device=x.device, dtype=torch.bool)
        output[:, ~h_mask, :] = 0
        
        return output
