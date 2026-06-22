import ctypes
import numpy as np
import os

# Load the DLL
dll_path = os.path.abspath("harmonic_stride.dll")
print(f"Loading DLL from {dll_path}...")
cuda_lib = ctypes.CDLL(dll_path)

# Define function prototype:
# void run_harmonic_stride_dll(float* h_data, float* h_out, int n, int blocksPerGrid)
cuda_lib.run_harmonic_stride_dll.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]
cuda_lib.run_harmonic_stride_dll.restype = None

# Set up test data
n = 10000
h_data = np.ones(n, dtype=np.float32)

# Calculate digital root
def get_dr(n_val):
    if n_val == 0:
        return 0
    return 1 + (abs(n_val) - 1) % 9

# Run with blocksPerGrid = 10
blocks_per_grid = 10
h_out = np.zeros(blocks_per_grid, dtype=np.float32)

# Pass pointers to ctypes
h_data_ptr = h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
h_out_ptr = h_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Call the CUDA kernel wrapper
print(f"Launching CUDA stride kernel with {blocks_per_grid} blocks...")
cuda_lib.run_harmonic_stride_dll(h_data_ptr, h_out_ptr, n, blocks_per_grid)

# Calculate Python reference
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

print("\n--- STRIDE KERNEL VERIFICATION ---")
for b in range(blocks_per_grid):
    stride = get_dr(b + 1)
    print(f"Block {b:02d} | Stride {stride} | CUDA Out: {h_out[b]:5.1f} | Py Out: {py_out[b]:5.1f}")

if np.allclose(h_out, py_out):
    print("\nVerification Succeeded! The CUDA stride kernel works perfectly and matches Python reference.")
else:
    print("\nVerification Failed.")
