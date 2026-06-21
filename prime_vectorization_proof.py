import numpy as np
import math

def is_prime(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

def prove_prime_vectorization(limit=2400):
    """
    Treats the number line as a 2D Vector Space with a width of 24.
    Demonstrates that Primes are not 'random' but follow 8 strictly 
    defined vectors (The Open Gates).
    """
    print(f"--- PRIME VECTORIZATION ANALYSIS (Width=24, N={limit}) ---")
    
    # 1. Reshape the number line into a 24-column matrix
    # Each row is one 'Grand Cycle' (24 steps)
    rows = limit // 24
    grid = np.arange(limit).reshape(rows, 24)
    
    # 2. Identify the Prime Vectors
    # We create a boolean mask of the same shape
    prime_mask = np.vectorize(is_prime)(grid)
    
    # 3. Sum the columns to see the density of each Vector
    vector_densities = np.sum(prime_mask, axis=0)
    
    print("\n[The 24-Column Vector Map]")
    print(f"{'Vector':<8} | {'Prime Count':<12} | {'Status'}")
    print("-" * 35)
    
    open_gates = {1, 5, 7, 11, 13, 17, 19, 23}
    
    for i in range(24):
        count = vector_densities[i]
        status = "OPEN (Radiating)" if i in open_gates else ("ANCHOR" if i in [2, 3] else "VOID (Interference)")
        print(f"V[{i:02d}]    | {count:<12} | {status}")

    # 4. Vectorized Filtering Simulation
    print("\n--- SIMD-STYLE FILTERING SIMULATION ---")
    print("In a traditional system, you check every number (O(N)).")
    print("In the ISF, we skip 16 out of 24 vectors entirely.")
    
    # The 'Compressed Prime Field'
    compressed_field = grid[:, sorted(list(open_gates))]
    print(f"Full Field Size:       {grid.shape}")
    print(f"Compressed Vector Set: {compressed_field.shape} (66.7% reduction)")
    
    # 5. The Digital Root Trajectory
    print("\n[Digital Root Vector Alignment]")
    # Primes on Vector 1 always have DRs in a specific sub-sequence
    v1_primes = [p for p in grid[:, 1] if is_prime(p)]
    v1_drs = [1 + (p-1)%9 for p in v1_primes[:10]]
    print(f"V[01] Digital Root Trajectory (First 10): {v1_drs}")
    
    v5_primes = [p for p in grid[:, 5] if is_prime(p)]
    v5_drs = [1 + (p-1)%9 for p in v5_primes[:10]]
    print(f"V[05] Digital Root Trajectory (First 10): {v5_drs}")

if __name__ == "__main__":
    prove_prime_vectorization()
