#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

// The 8 Open Gates of the Icositetragon (mod 24)
__device__ const int OPEN_GATES[8] = {1, 5, 7, 11, 13, 17, 19, 23};

/**
 * Maps a linear thread index to a 'Harmonic' shared memory address.
 * By using the 24-step stride, we ensure threads operate in 'Silent Lanes'.
 */
__device__ __forceinline__ int get_harmonic_idx(int tid) {
    int cycle = tid / 8;        // Which 24-step Grand Cycle we are in
    int gate_idx = tid % 8;     // Which of the 8 Open Gates we land on
    return (cycle * 24) + OPEN_GATES[gate_idx];
}

__global__ void harmonic_reduction_kernel(float* g_data, float* g_out, int n) {
    // Allocate shared memory for the icositetragon (24 steps per block segment)
    // For 32 threads, we need (32/8)*24 = 96 elements
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Filtered Loading (Anti-Feedback)
    // We map threads to the icositetragon, skipping 3-6-9 nodes
    int h_idx = get_harmonic_idx(tid);
    s_data[h_idx] = (i < n) ? g_data[i] : 0.0f;
    __syncthreads();

    // Phase 2: Power-of-Two Reduction within the 'Open Gates'
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int idx_a = get_harmonic_idx(tid);
            int idx_b = get_harmonic_idx(tid + s);
            s_data[idx_a] += s_data[idx_b];
        }
        __syncthreads();
    }

    // Phase 3: The Ground (Step 1 of the first cycle)
    if (tid == 0) g_out[blockIdx.x] = s_data[1]; 
}

int main() {
    const int n = 1024;
    const int threadsPerBlock = 32;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> h_data(n, 1.0f); // Fill with 1s for easy verification
    std::vector<float> h_out(blocksPerGrid, 0.0f);

    float *d_data, *d_out;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_out, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Shared memory size: (threadsPerBlock / 8) * 24
    int sharedMemSize = (threadsPerBlock / 8) * 24 * sizeof(float);
    
    harmonic_reduction_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_data, d_out, n);

    cudaMemcpy(h_out.data(), d_out, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0;
    for (float val : h_out) {
        total_sum += val;
    }

    std::cout << "Expected Sum: " << n << std::endl;
    std::cout << "Calculated Sum: " << total_sum << std::endl;

    cudaFree(d_data);
    cudaFree(d_out);

    return 0;
}
