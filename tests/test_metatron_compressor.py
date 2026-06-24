import pytest
import torch
import numpy as np

from metatron_compressor_engine import MetatronCompressor
from harmonic_constants import IS_OPEN_GATE

SEQ_LEN = 240
BATCH = 2

def test_run_compression_cycle_cpu_shape_and_type():
    # Create deterministic input on CPU
    chaos = torch.tensor([[i % 1000 for i in range(SEQ_LEN)] for _ in range(BATCH)], dtype=torch.long)
    compressor = MetatronCompressor()
    # Ensure model is on CPU
    compressor.model.to('cpu')
    out = compressor.run_compression_cycle(chaos)
    assert isinstance(out, np.ndarray)
    # Output corresponds to the first batch element's per-token norms -> length == seq_len
    assert out.shape[0] == SEQ_LEN

def test_start_capacitor_batch_masking():
    # Construct input filled with ones so masked positions clearly become zero
    x = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)
    compressor = MetatronCompressor()
    primed = compressor.start_capacitor(x)
    # Ensure masked columns (not open gates) are zero for all batch rows
    for pos in range(SEQ_LEN):
        is_open = IS_OPEN_GATE[pos % 24]
        col = primed[:, pos]
        if is_open:
            assert torch.all(col != 0), f"Open gate position {pos} should not be zeroed."
        else:
            assert torch.all(col == 0), f"Void gate position {pos} should be zeroed."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_safety_cuda_to_numpy():
    device = torch.device("cuda")
    chaos = torch.tensor([[i % 1000 for i in range(SEQ_LEN)]], dtype=torch.long, device=device)
    compressor = MetatronCompressor()
    # Move model to CUDA to emulate GPU inference
    compressor.model.to(device)
    out = compressor.run_compression_cycle(chaos)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == SEQ_LEN
