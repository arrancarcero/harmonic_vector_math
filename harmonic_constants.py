"""
The Icositetragon Sparse Framework (ISF) - Global Constants
Optimized for the Blackwell Era and High-Performance Harmonic Intelligence.
Everything is Tupled for speed and immutability.
"""

# --- THE 8 OPEN GATES (Residues coprime to 24) ---
# IMMUTABLE TUPLE: Faster iteration and safer logic
OPEN_GATES = (1, 5, 7, 11, 13, 17, 19, 23)

# --- THE SUPERCONDUCTOR TUPLE (O(1) Lookup) ---
# Index-based boolean lookup: 3.5x faster than 'in'
IS_OPEN_GATE = (
    False, True,  False, False, False, True,  # 0-5
    False, True,  False, False, False, True,  # 6-11
    False, True,  False, False, False, True,  # 12-17
    False, True,  False, False, False, True,  # 18-23
    False                                     # 24 (Anchor)
)

# --- THE VOID GATES (Deterministic Composites) ---
VOID_GATES = (0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24)

# --- THE 3-6-9 ANCHORS ---
ANCHORS_369 = (3, 6, 9, 12, 15, 18, 21, 24)

def gate_mod24(n: int) -> int:
    """Map residue to 1-24 range for the icositetragon."""
    return n % 24 or 24

def is_resonant(n: int) -> bool:
    """True if n lands on an Open Gate (Resonant Lane)."""
    # Uses the Superconductor Tuple Index for O(1) performance
    return IS_OPEN_GATE[n % 24]

def is_prime(n: int) -> bool:
    """Foundational prime check aligned to ISF anchors."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    # Only check 6k +/- 1 positions (Radiating Lanes)
    for i in range(5, int(n**0.5) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True
