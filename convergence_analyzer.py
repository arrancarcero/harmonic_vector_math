import math

def sieve_of_eratosthenes(limit):
    """Generate all primes up to limit using a fast sieve."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [x for x, p in enumerate(is_prime) if p]

def analyze_convergence(thresholds):
    max_limit = max(thresholds)
    print(f"Generating primes up to {max_limit}...")
    all_primes = sieve_of_eratosthenes(max_limit)
    print(f"Generated {len(all_primes)} primes.\n")
    
    # Coprime residues
    coprime_mod9 = {1, 2, 4, 5, 7, 8}
    coprime_mod24 = {1, 5, 7, 11, 13, 17, 19, 23}
    
    for limit in thresholds:
        # Filter primes up to current limit (excluding 2 and 3 for residue analysis)
        primes_subset = [p for p in all_primes if 3 < p <= limit]
        total_subset = len(primes_subset)
        
        if total_subset == 0:
            continue
            
        print(f"=== ANALYSIS AT LIMIT: {limit:,} (Primes > 3: {total_subset:,}) ===")
        
        # 1. Modulo 9 Convergence Analysis (Target: 16.67% per lane)
        print("  --- Modulo 9 (Target: 1/6 ~= 16.67% per lane) ---")
        counts_mod9 = {r: 0 for r in coprime_mod9}
        for p in primes_subset:
            r = p % 9
            if r in counts_mod9:
                counts_mod9[r] += 1
                
        for r in sorted(counts_mod9.keys()):
            pct = (counts_mod9[r] / total_subset) * 100
            diff = pct - (100 / 6)
            sign = "+" if diff >= 0 else ""
            print(f"    Lane {r} -> Count: {counts_mod9[r]:6} | Percentage: {pct:6.3f}% | Dev: {sign}{diff:.3f}%")
            
        # 2. Modulo 24 Convergence Analysis (Target: 12.5% per lane)
        print("\n  --- Modulo 24 (Target: 1/8 = 12.50% per lane) ---")
        counts_mod24 = {r: 0 for r in coprime_mod24}
        for p in primes_subset:
            r = p % 24
            if r in counts_mod24:
                counts_mod24[r] += 1
                
        for r in sorted(counts_mod24.keys()):
            pct = (counts_mod24[r] / total_subset) * 100
            diff = pct - 12.5
            sign = "+" if diff >= 0 else ""
            print(f"    Lane {r:2} -> Count: {counts_mod24[r]:6} | Percentage: {pct:6.3f}% | Dev: {sign}{diff:.3f}%")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Test convergence at three distinct orders of magnitude
    analyze_convergence([10000, 100000, 1000000])
