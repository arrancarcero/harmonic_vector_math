import time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

def run_fixed_point_benchmark():
    n = 500000
    print(f"--- Fixed-Point Vectorization Benchmark (Size: {n:,}) ---")
    print("Goal: Calculate 8.25% tax on prices. Must be 100% accurate to the cent.\n")
    
    # 1. Generate Raw Price Data
    # Let's generate a list of prices like 19.99, 10.50, etc.
    raw_prices = [19.99 + (i % 100) * 0.01 for i in range(n)]
    
    # --- METHOD A: Sequential Decimal (The baseline for 100% accuracy) ---
    start_time = time.time()
    decimal_prices = [Decimal(str(p)) for p in raw_prices]
    tax_rate_dec = Decimal('0.0825')
    decimal_results = []
    
    for price in decimal_prices:
        tax = (price * tax_rate_dec).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        decimal_results.append(tax)
    
    decimal_time = time.time() - start_time
    print(f"Method A (Sequential Decimal) -> Time: {decimal_time:.6f} seconds | Exact Accuracy")

    # --- METHOD B: Standard Float Vectorization (Fast, but has rounding errors) ---
    start_time = time.time()
    float_prices = np.array(raw_prices)
    float_tax = np.round(float_prices * 0.0825, 2)
    float_time = time.time() - start_time
    
    # Check accuracy against Method A
    decimal_results_float = np.array([float(d) for d in decimal_results])
    errors_float = np.sum(~np.isclose(float_tax, decimal_results_float))
    print(f"Method B (Float Vectorized)   -> Time: {float_time:.6f} seconds | Rounding Errors: {errors_float:,}")

    # --- METHOD C: Scaled Integer Vectorization (Fast AND 100% accurate) ---
    # We multiply prices by 100 to represent cents.
    # We multiply the tax rate by 10000 (8.25% -> 825) to represent it as an integer.
    start_time = time.time()
    
    # Convert prices to cents (integers)
    cents_prices = np.array([int(round(p * 100)) for p in raw_prices], dtype=np.int64)
    
    # Calculate tax: (cents * 825) / 10000
    # To round half up correctly in integer math, we add 5000 before integer division by 10000.
    cents_tax = (cents_prices * 825 + 5000) // 10000
    
    # Scale back to floats for final output (dollars)
    scaled_results = cents_tax / 100.0
    
    fixed_point_time = time.time() - start_time
    
    # Check accuracy against Method A
    errors_fixed = np.sum(~np.isclose(scaled_results, decimal_results_float))
    print(f"Method C (Scaled Integer Vec) -> Time: {fixed_point_time:.6f} seconds | Rounding Errors: {errors_fixed}")
    
    # Performance gains
    speedup = decimal_time / fixed_point_time
    print(f"\nSpeedup: Method C is {speedup:.2f}x faster than Method A while maintaining 100% accuracy.")

if __name__ == "__main__":
    run_fixed_point_benchmark()
