import ctypes
import numpy as np
import os

# Load the DLL
dll_path = os.path.abspath("harmonic_reduction.dll")
print(f"Loading DLL from {dll_path}...")
cuda_lib = ctypes.CDLL(dll_path)

# Define function prototype
# void run_harmonic_reduction_dll(float* h_data, float* h_out, int n)
cuda_lib.run_harmonic_reduction_dll.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
cuda_lib.run_harmonic_reduction_dll.restype = None

# Set up test data
n = 1024
# Initialize host input array with 1s
h_data = np.ones(n, dtype=np.float32)
# Output array size: ceil(n / threadsPerBlock) = ceil(1024 / 32) = 32 elements
blocks_per_grid = (n + 31) // 32
h_out = np.zeros(blocks_per_grid, dtype=np.float32)

# Pass pointers to ctypes
h_data_ptr = h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
h_out_ptr = h_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Call the CUDA kernel wrapper
print("Launching CUDA kernel on the GPU...")
cuda_lib.run_harmonic_reduction_dll(h_data_ptr, h_out_ptr, n)

# Verify output
total_sum = float(np.sum(h_out))
print(f"Expected Sum: {n}")
print(f"Calculated Sum: {total_sum}")
if abs(total_sum - n) < 1e-5:
    print("Verification Succeeded! The CUDA kernel calculated the exact sum on the GPU via ctypes.")
else:
    print("Verification Failed.")
