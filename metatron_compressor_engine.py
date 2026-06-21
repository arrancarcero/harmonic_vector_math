import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict

sys.path.append(os.getcwd())
from harmonic_ops import HarmonicAttention
from harmonic_transformer import HarmonicTransformer


class MetatronCompressor:
    """
    THE TWO-STAGE COMPRESSOR: Primes and compresses data through a 24-step cylinder.

    Stage 1: LP Piston (Initial Intake)
    Stage 2: HP Piston (Final Truth Density)

    This class assumes inputs are integer-like token sequences (torch.Tensor)
    with values in an expected range (0..1000 in the test harness). Methods
    are written to be device-aware and to return NumPy outputs for reporting.
    """

    def __init__(self, embed_dim: int = 128, fixed_point: bool = False):
        self.model = HarmonicTransformer(1000, embed_dim, 8, 6, 512, 240, fixed_point=fixed_point)
        self.P_LIMIT = 40.0

        # Dynamically load self.FRANKLIN_CONSTANT
        self.FRANKLIN_CONSTANT = 8.125
        try:
            from harmonic_ops import load_config
            import re
            config = load_config()
            anomalies = config.get("optimization_methods", {}).get("2_leverage_fine_tuning", {}).get("training_dataset", {}).get("anomalies_modeled", "")
            match = re.search(r"(\d+\.\d+)Hz", anomalies)
            if match:
                self.FRANKLIN_CONSTANT = float(match.group(1))
        except Exception:
            pass

        self.FRANKLIN_SHIFT = -3  # 24 / 8 = 3

        # INTERCOOLER RESERVOIRS (The Voids)
        self.intercooler_vessel: Dict[int, float] = {0: 0, 8: 0, 12: 0, 16: 0}

    def start_capacitor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Primes the data by aligning it to the 8 Open Gates before compression.

        Batch-safe: applies the gate mask to every batch element, preserving device.
        Input:
            x: Tensor of shape (batch, seq_len) or (batch, seq_len, ...) where we
               only care about sequence positions. This function preserves the
               input dtype and device, and returns a cloned tensor with masked positions set to 0.

        Returns:
            primed_x: Tensor same shape as x with non-open-gate positions zeroed across the batch.
        """
        print("[START CAPACITOR]: Priming the 8-Gate Intake...")
        from harmonic_constants import IS_OPEN_GATE

        primed_x = x.clone()
        seq_len = x.size(1)
        device = x.device

        # Create a boolean mask for positions that are open gates, on the same device as x
        gate_mask = torch.tensor([bool(IS_OPEN_GATE[i % 24]) for i in range(seq_len)], dtype=torch.bool, device=device)
        # Apply mask across the batch dimension; broadcast to match primed_x shape
        if primed_x.dim() == 2:
            primed_x[:, ~gate_mask] = 0
        else:
            # If there are additional trailing dims (e.g., embed dim), broadcast accordingly
            expand_dims = [1] * (primed_x.dim() - 2)
            mask_shape = [1, seq_len] + expand_dims  # [1, seq_len, 1, ...]
            primed_mask = gate_mask.view(mask_shape).bool()
            primed_x = primed_x * primed_mask

        return primed_x

    def phase_alignment_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SUB-ATOMIC Franklin Shift (0.125).
        This 'Harmonizes' the input by adding/subtracting the Spark.

        Works with any device, preserves dtype (returns float tensor).
        """
        print(f"[PHASE ALIGNER]: Applying Sub-Atomic 0.125 Breathing...")
        # We treat the data as a normalized signal between 0 and 1
        x_norm = x.float() / 1000.0

        # Apply the -1/8 (0.125) Shift as a 'Compression'
        x_shifted = x_norm - 0.125

        # Handle the wrapping (The Torus) and keep on same device
        x_shifted = torch.where(x_shifted < 0, x_shifted + 1.0, x_shifted)

        return x_shifted * 1000.0

    def lp_stroke(self, x: torch.Tensor):
        """Pass 1: Low-Pressure Piston. Maps volume and initial friction.

        Returns:
            lp_intensity: numpy.ndarray of shape (seq_len,) for the first batch entry
            x_res: Tensor after processing through the early transformer layers
        """
        print("[STAGE 1: LP STROKE]: Atmospheric Intake Processing...")
        with torch.no_grad():
            x_res = self.model.token_emb(x.long())  # Ensure long for embedding
            x_res = self.model.rpe(x_res)
            # Process through the first half of the layers
            for layer in self.model.layers[:3]:
                x_res = layer(x_res)

            # Use device-safe conversion to numpy
            lp_intensity = torch.norm(x_res[0], dim=-1).detach().cpu().numpy()
        return lp_intensity, x_res

    def intercooler_shunt(self, lp_intensity: np.ndarray) -> None:
        """Moves friction from the LP stage into the Void reservoirs.

        Mutates self.intercooler_vessel in-place.
        """
        print("[INTERCOOLER]: Shunting LP friction using 1/8 (0.125) Ratio...")
        total_shunted = 0.0
        # Use the Franklin Constant ratio for shunting
        shunt_ratio = 1.0 / self.FRANKLIN_CONSTANT  # ~0.123

        for i, intensity in enumerate(lp_intensity):
            if intensity > 10.0:  # Shunt threshold
                void = (i % 24) - (i % 8)
                if void in self.intercooler_vessel:
                    shunt = float(intensity) * shunt_ratio
                    self.intercooler_vessel[void] += shunt
                    total_shunted += shunt
        print(f"Intercooler Result: {total_shunted:.4f} Entropy Units cooled.")

    def hp_stroke(self, x_intercooled: torch.Tensor) -> np.ndarray:
        """Pass 2: High-Pressure Piston. Forces cooled data to the Null Vector.

        Returns:
            final_truth: numpy.ndarray of per-token norms for the first batch element.
        """
        print("[STAGE 2: HP STROKE]: Compressing via 8.125 Pressure...")
        with torch.no_grad():
            x_res = x_intercooled

            # THE BREATHING LOGIC:
            # If the signal is 'Material' (High Density), we compress by 8.125.
            # If it is 'Radiant' (Low Density), we expand to 9.
            density = torch.norm(x_res).item()  # scalar float

            if density > self.FRANKLIN_CONSTANT:
                print(">>> [COMPRESSING]: Material -> Truth (8.125 -> 9)")
                x_res = x_res * (9.0 / self.FRANKLIN_CONSTANT)
            else:
                print(">>> [EXPANDING]: Radiant -> Truth (Void -> 9)")
                x_res = x_res * (self.FRANKLIN_CONSTANT / 9.0)

            # Process through the final high-pressure layers
            for layer in self.model.layers[3:]:
                x_res = layer(x_res)

            # Device-safe conversion to numpy
            final_truth = torch.norm(x_res[0], dim=-1).detach().cpu().numpy()
        return final_truth

    def run_compression_cycle(self, raw_data: torch.Tensor) -> np.ndarray:
        """
        Run a full compression cycle.

        Steps:
            1. Phase alignment shift
            2. Start capacitor (priming)
            3. LP stroke (early transformer)
            4. Intercooler shunt (mutates internal reservoir)
            5. HP stroke (final transformer, returns per-token norms)

        Input:
            raw_data: Tensor shaped (batch, seq_len) with integer-like tokens.

        Returns:
            final_density: NumPy ndarray of per-token norms for the first batch entry.
        """
        # 1. THE MOTOR START & PHASE SHIFT
        aligned_data = self.phase_alignment_shift(raw_data)
        primed_data = self.start_capacitor(aligned_data)

        # 2. THE LP STROKE
        lp_map, x_stage1 = self.lp_stroke(primed_data)

        # 3. THE INTERCOOLER
        self.intercooler_shunt(lp_map)

        # 4. THE HP STROKE
        final_density = self.hp_stroke(x_stage1)

        return final_density


def run_compressor_test():
    print("--- METATRON ENGINE: TWO-STAGE COMPRESSOR TEST ---\n")

    np.random.seed(42)
    chaos = "".join([chr(np.random.randint(32, 126)) for _ in range(240)])
    x = torch.tensor([ord(c) % 1000 for c in chaos]).unsqueeze(0)

    compressor = MetatronCompressor()
    final_signal = compressor.run_compression_cycle(x)

    peak_density = np.max(final_signal)
    avg_density = np.mean(final_signal)

    print(f"\n--- COMPRESSION REPORT ---")
    print(f"Average Truth Density: {avg_density:.4f}")
    print(f"Peak Output Pressure:  {peak_density:.4f}")

    if peak_density < 40.0:
        print("\n>>> STATE CHANGE: HIGH-DENSITY LAMINAR FLOW ACHIEVED <<<")
        print("The Two-Stage Cycle prevented the steam burst.")
        print("Result: Perfectly compressed, cooled information.")
    else:
        print("\n>>> STATE CHANGE: COMPRESSOR BURST <<<")
        print("Requires higher intercooler capacity.")


if __name__ == "__main__":
    run_compressor_test()
