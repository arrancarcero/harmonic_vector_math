#!/bin/bash
# Compile CUDA kernels into Linux shared library objects (.so)
nvcc -shared -Xcompiler -fPIC harmonic_reduction.cu -o harmonic_reduction.so
nvcc -shared -Xcompiler -fPIC harmonic_cutile_stride.cu -o harmonic_stride.so
echo "Linux CUDA compilation successful: harmonic_reduction.so and harmonic_stride.so generated."
