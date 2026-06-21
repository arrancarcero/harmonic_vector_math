import time
import numpy as np

def run_vector_benchmark():
    # Size of the dataset (1 million elements)
    n = 1000000
    print(f"--- Vectorized Math Benchmark (Array Size: {n:,}) ---")
    
    # Generate two random arrays using standard Python lists
    list_a = list(np.random.rand(n))
    list_b = list(np.random.rand(n))
    
    # Generate the same random arrays as NumPy vectors
    vector_a = np.array(list_a)
    vector_b = np.array(list_b)
    
    # 1. Sequential Calculation (Standard Python Loop)
    start_time = time.time()
    loop_result = []
    for i in range(n):
        loop_result.append(list_a[i] + list_b[i])
    loop_time = time.time() - start_time
    print(f"Sequential Loop Time: {loop_time:.6f} seconds")
    
    # 2. Vectorized Calculation (NumPy SIMD acceleration)
    start_time = time.time()
    vector_result = vector_a + vector_b
    vector_time = time.time() - start_time
    print(f"Vectorized Math Time: {vector_time:.6f} seconds")
    
    # Calculate performance difference
    speedup = loop_time / vector_time
    print(f"\nSpeedup Factor: {speedup:.2f}x faster using vectorized math")
    
    # Verify correctness of both results
    # Convert loop result to numpy array for comparison
    loop_result_np = np.array(loop_result)
    is_identical = np.allclose(loop_result_np, vector_result)
    print(f"Results match exactly: {is_identical}")

if __name__ == "__main__":
    run_vector_benchmark()
