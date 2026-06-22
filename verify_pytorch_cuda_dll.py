import torch
import ctypes
import sys
import os
import numpy as np

def run_integration_test():
    print("--- PyTorch-CUDA Custom DLL Direct Binding Test ---")
    
    # 1. Check PyTorch CUDA capability
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: PyTorch is currently CPU-only. We will simulate direct GPU integration using host wrappers.")

    # 2. Load Libraries
    suffix = ".dll" if sys.platform == "win32" else ".so"
    reduction_dll_path = os.path.abspath(f"harmonic_reduction{suffix}")
    stride_dll_path = os.path.abspath(f"harmonic_stride{suffix}")
    
    print(f"Loading reduction library from {reduction_dll_path}...")
    reduction_lib = ctypes.CDLL(reduction_dll_path)
    
    print(f"Loading stride library from {stride_dll_path}...")
    stride_lib = ctypes.CDLL(stride_dll_path)

    # Define function prototypes for direct GPU memory access
    # void run_harmonic_reduction_direct_gpu(float* d_data, float* d_out, int n)
    reduction_lib.run_harmonic_reduction_direct_gpu.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    reduction_lib.run_harmonic_reduction_direct_gpu.restype = None

    # void run_harmonic_stride_direct_gpu(float* d_data, float* d_out, int n, int blocksPerGrid)
    stride_lib.run_harmonic_stride_direct_gpu.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int
    ]
    stride_lib.run_harmonic_stride_direct_gpu.restype = None

    n = 1024
    
    if cuda_available:
        # --- PATH A: Native Direct GPU Integration ---
        print("\n[Executing Path A: Zero-Copy Direct GPU Pointers]")
        
        # Allocate tensors directly on GPU
        x_gpu = torch.ones(n, device="cuda", dtype=torch.float32)
        
        # Reduction outputs
        blocks_red = (n + 31) // 32
        y_red_gpu = torch.zeros(blocks_red, device="cuda", dtype=torch.float32)
        
        # Stride outputs
        blocks_stride = 10
        y_stride_gpu = torch.zeros(blocks_stride, device="cuda", dtype=torch.float32)
        
        # Invoke kernels directly using data pointers
        print("Launching direct-GPU reduction kernel...")
        reduction_lib.run_harmonic_reduction_direct_gpu(
            ctypes.c_void_p(x_gpu.data_ptr()),
            ctypes.c_void_p(y_red_gpu.data_ptr()),
            n
        )
        
        print("Launching direct-GPU stride kernel...")
        stride_lib.run_harmonic_stride_direct_gpu(
            ctypes.c_void_p(x_gpu.data_ptr()),
            ctypes.c_void_p(y_stride_gpu.data_ptr()),
            n,
            blocks_stride
        )
        
        # Fetch results to CPU for verification
        sum_red = float(y_red_gpu.cpu().sum())
        out_stride = y_stride_gpu.cpu().numpy()
        
        print(f"Direct Reduction Sum: {sum_red} (Expected: 1024.0)")
        print(f"Direct Stride Output: {out_stride}")
        
    else:
        # --- PATH B: CPU/Host Boundary Wrapper ---
        print("\n[Executing Path B: Host Boundary Copy Simulation]")
        
        # Setup host arrays
        h_data = np.ones(n, dtype=np.float32)
        
        # Define host wrappers prototypes
        reduction_lib.run_harmonic_reduction_dll.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        reduction_lib.run_harmonic_reduction_dll.restype = None
        
        stride_lib.run_harmonic_stride_dll.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        stride_lib.run_harmonic_stride_dll.restype = None
        
        # Reduction execution
        blocks_red = (n + 31) // 32
        h_red_out = np.zeros(blocks_red, dtype=np.float32)
        
        reduction_lib.run_harmonic_reduction_dll(
            h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h_red_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n
        )
        
        # Stride execution
        blocks_stride = 10
        h_stride_out = np.zeros(blocks_stride, dtype=np.float32)
        
        stride_lib.run_harmonic_stride_dll(
            h_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h_stride_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            blocks_stride
        )
        
        sum_red = float(np.sum(h_red_out))
        print(f"Host Reduction Sum: {sum_red} (Expected: 1024.0)")
        print(f"Host Stride Output: {h_stride_out}")

    print("\nDLL Integration Verification Complete!")

if __name__ == "__main__":
    run_integration_test()
