import os
import ctypes
import numpy as np
import pytest

def get_dr(n_val):
    if n_val == 0:
        return 0
    return 1 + (abs(n_val) - 1) % 9

def test_cuda_reduction_dll_execution():
    """Verify the compiled C++ CUDA reduction kernel via ctypes host wrapper."""
    dll_path = os.path.abspath("harmonic_reduction.dll")
    assert os.path.exists(dll_path), f"Reduction DLL not found at {dll_path}"
    
    # Load the library
    cuda_lib = ctypes.CDLL(dll_path)
    
    # Define prototype
    cuda_lib.run_harmonic_reduction_dll.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    cuda_lib.run_harmonic_reduction_dll.restype = None
    
    n = 512
    h_data = np.ones(n, dtype=np.float32)
    blocks_per_grid = (n + 31) // 32
    h_out = np.zeros(blocks_per_grid, dtype=np.float32)
    
    h_data_ptr = h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_out_ptr = h_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Run the kernel
    cuda_lib.run_harmonic_reduction_dll(h_data_ptr, h_out_ptr, n)
    
    # Each thread block has 32 threads.
    # In each thread block, get_harmonic_idx(tid) maps tid [0..31] to (cycle*24) + gate.
    # Specifically, for threads [0..31], cycles are [0, 1, 2, 3] (since cycle = tid / 8).
    # For each cycle (8 threads), they load into s_data[(cycle*24) + gate].
    # Then reduction sums them.
    # Expected sum per block is the number of active loaded gates in that block.
    # For block 0: elements loaded are up to index 31.
    # The active indices in s_data are:
    # cycle 0: 1, 5, 7, 11, 13, 17, 19, 23
    # cycle 1: 25, 29, 31, 35, 37, 41, 43, 47
    # cycle 2: 49, 53, 55, 59, 61, 65, 67, 71
    # cycle 3: 73, 77, 79, 83, 85, 89, 91, 95
    # All these indices map < n = 512.
    # So all 32 threads load 1.0f. The block reduction sums all 32 elements.
    # Hence, each block should output exactly 32.0f.
    # Total blocks = 512 / 32 = 16. Total sum = 16 * 32 = 512.0f.
    total_sum = float(np.sum(h_out))
    assert abs(total_sum - n) < 1e-5, f"CUDA reduction sum deviation: {total_sum} vs {n}"

def test_cuda_stride_dll_execution():
    """Verify the compiled C++ CUDA stride kernel via ctypes host wrapper."""
    dll_path = os.path.abspath("harmonic_stride.dll")
    assert os.path.exists(dll_path), f"Stride DLL not found at {dll_path}"
    
    cuda_lib = ctypes.CDLL(dll_path)
    
    cuda_lib.run_harmonic_stride_dll.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    cuda_lib.run_harmonic_stride_dll.restype = None
    
    n = 10000
    h_data = np.ones(n, dtype=np.float32)
    blocks_per_grid = 10
    h_out = np.zeros(blocks_per_grid, dtype=np.float32)
    
    h_data_ptr = h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_out_ptr = h_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Run stride kernel
    cuda_lib.run_harmonic_stride_dll(h_data_ptr, h_out_ptr, n, blocks_per_grid)
    
    # Verify mathematically matching Python reference
    OPEN_GATES = [1, 5, 7, 11, 13, 17, 19, 23]
    py_out = np.zeros(blocks_per_grid, dtype=np.float32)
    
    for b in range(blocks_per_grid):
        stride = get_dr(b + 1)
        s_tile = np.zeros(24, dtype=np.float32)
        for gate in OPEN_GATES:
            global_idx = (b * 24 * stride) + gate
            if global_idx < n:
                s_tile[gate] = h_data[global_idx]
        
        resonance_sum = 0.0
        for gate in OPEN_GATES:
            resonance_sum += s_tile[gate]
        py_out[b] = resonance_sum

    assert np.allclose(h_out, py_out), f"CUDA stride output {h_out} does not match Py reference {py_out}"
