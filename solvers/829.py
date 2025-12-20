#!/usr/bin/env python3
"""Project Euler 829: Integral Fusion

We define a deterministic binary factor tree T(n): split n into a*b where
(a <= b) and (b-a) is minimal (i.e. choose the divisor a closest to sqrt(n)
from below), then recurse.

For each n, M(n) is the smallest integer whose factor tree has the same
(unlabelled) shape as T(n!!). This program computes sum_{n=2..31} M(n).

Constraints from the prompt:
- No external libraries.
- Assert any test values stated in the problem statement.
"""

from __future__ import annotations

import bisect
import heapq
import math
from typing import Dict, List, Optional, Tuple


# ---------------------------
# 64-bit primality + factoring
# ---------------------------

_primality_cache: Dict[int, bool] = {}


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 0 <= n < 2^64."""
    if n in _primality_cache:
        return _primality_cache[n]

    if n < 2:
        _primality_cache[n] = False
        return False

    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            _primality_cache[n] = True
            return True
        if n % p == 0:
            _primality_cache[n] = False
            return False

    # Write n-1 = d*2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic bases for 64-bit integers
    # See: https://miller-rabin.appspot.com/
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in bases:
        if a % n == 0:
            continue
        if not check(a):
            _primality_cache[n] = False
            return False

    _primality_cache[n] = True
    return True


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _pollard_rho_brent(n: int) -> int:
    """Return a non-trivial factor of composite odd n (64-bit)."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    # Deterministic parameter choices
    # Brent cycle detection with polynomial f(x) = x^2 + c mod n
    for c in (1, 3, 5, 7, 11, 13, 17, 19, 23):
        y = 2
        m = 128
        g = 1
        r = 1
        q = 1

        def f(x: int) -> int:
            return (x * x + c) % n

        while g == 1:
            x = y
            for _ in range(r):
                y = f(y)
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = f(y)
                    q = (q * abs(x - y)) % n
                g = _gcd(q, n)
                k += m
            r <<= 1

        if g == n:
            # fallback: standard gcd steps
            g = 1
            y = ys
            while g == 1:
                y = f(y)
                g = _gcd(abs(x - y), n)

        if 1 < g < n:
            return g

    # Should be extremely rare for 64-bit with the c values above; retry with another seed.
    # If it happens, we can still fall back to trial division up to a small limit.
    limit = int(math.isqrt(n))
    d = 5
    while d <= limit and d <= 1_000_000:
        if n % d == 0:
            return d
        d += 2
    return n  # give up (should not happen for our inputs)


_factor_cache: Dict[int, Dict[int, int]] = {}


def factorize(n: int) -> Dict[int, int]:
    """Prime factorization of n as a dict {prime: exponent}."""
    if n in _factor_cache:
        return dict(_factor_cache[n])

    original = n
    factors: Dict[int, int] = {}

    # Trial division by small primes first
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors[p] = factors.get(p, 0) + e

    def _recurse(x: int) -> None:
        if x == 1:
            return
        if is_probable_prime(x):
            factors[x] = factors.get(x, 0) + 1
            return
        d = _pollard_rho_brent(x)
        if d == x:
            # As a last resort, treat it as prime (should not occur)
            factors[x] = factors.get(x, 0) + 1
            return
        _recurse(d)
        _recurse(x // d)

    if n > 1:
        _recurse(n)

    # Normalize
    _factor_cache[original] = dict(factors)
    return dict(factors)


# ---------------------------
# Factor tree shape utilities
# ---------------------------

# We represent a shape as:
#   LEAF (None) for a prime node
#   (left_shape, right_shape) for a composite node
Shape = Optional[Tuple["Shape", "Shape"]]

# Sentinel value representing a leaf node in the factor-tree shape.
LEAF: Shape = None

_shape_cache: Dict[int, Shape] = {}
_best_divisor_cache: Dict[int, int] = {}


def _products_from_factors(items: List[Tuple[int, int]]) -> List[int]:
    """Generate all divisor-products from a list of (p, e) items."""
    res = [1]
    for p, e in items:
        powers = [1]
        v = 1
        for _ in range(e):
            v *= p
            powers.append(v)
        new_res = []
        for base in res:
            for pw in powers:
                new_res.append(base * pw)
        res = new_res
    return res


def best_divisor_le_sqrt(n: int) -> int:
    """Return the largest divisor d of n such that d <= sqrt(n)."""
    if n in _best_divisor_cache:
        return _best_divisor_cache[n]

    # n is expected composite when called
    fac = factorize(n)
    items = sorted(fac.items())
    half = len(items) // 2
    left_items = items[:half]
    right_items = items[half:]

    div_left = _products_from_factors(left_items)
    div_right = _products_from_factors(right_items)
    div_right.sort()

    root = int(math.isqrt(n))
    best = 1
    for a in div_left:
        if a > root:
            continue
        limit = root // a
        j = bisect.bisect_right(div_right, limit) - 1
        if j >= 0:
            cand = a * div_right[j]
            if cand > best:
                best = cand

    _best_divisor_cache[n] = best
    return best


def shape_of(n: int) -> Shape:
    """Compute the ordered factor-tree shape of n."""
    if n in _shape_cache:
        return _shape_cache[n]
    if is_probable_prime(n):
        _shape_cache[n] = LEAF
        return LEAF

    d = best_divisor_le_sqrt(n)
    a = d
    b = n // d
    # Ensure a <= b (should always hold)
    if a > b:
        a, b = b, a

    sh = (shape_of(a), shape_of(b))
    _shape_cache[n] = sh
    return sh


def double_factorial(n: int) -> int:
    res = 1
    for k in range(n, 1, -2):
        res *= k
    return res


def count_leaves(sh: Shape) -> int:
    if sh is LEAF:
        return 1
    return count_leaves(sh[0]) + count_leaves(sh[1])  # type: ignore[index]


# ---------------------------
# Enumerate numbers by shape
# ---------------------------


class ShapeSequence:
    """Lazy increasing sequence of all numbers <= MAXVAL that realize a given shape."""

    def __init__(self, sh: Shape, maxval: int):
        self.sh = sh
        self.maxval = maxval
        self.values: List[int] = []

        if sh is LEAF:
            self._next_prime = 2
            self._gen_kind = "leaf"
        else:
            self.left: Shape = sh[0]  # type: ignore[index]
            self.right: Shape = sh[1]  # type: ignore[index]
            self._gen_kind = "node"
            self._heap: List[Tuple[int, int, int]] = []  # (product, i, j)
            self._in_heap: set[Tuple[int, int]] = set()
            self._next_i_to_add = 0
            self._started = False

    def _yield_next_prime(self) -> int:
        p = self._next_prime
        if p > self.maxval:
            raise StopIteration
        if p == 2:
            self._next_prime = 3
            return 2

        cand = p
        while True:
            if cand > self.maxval:
                raise StopIteration
            if is_probable_prime(cand):
                self._next_prime = cand + 2
                return cand
            cand += 2

    def _ensure_child_value(self, sh: Shape, idx: int) -> int:
        return get_value(sh, idx)

    def _try_child_value(self, sh: Shape, idx: int) -> Optional[int]:
        """Get a child value if it exists (<= maxval), else return None."""
        try:
            return get_value(sh, idx)
        except StopIteration:
            return None

    def _ensure_right_ge(self, x: int) -> Optional[int]:
        """Return smallest j such that right[j] >= x, extending right as needed."""
        st = _seq(self.right)
        # ensure at least one element
        if not st.values:
            try:
                st.values.append(st._next_value())
            except StopIteration:
                return None

        while st.values[-1] < x:
            try:
                nxt = st._next_value()
            except StopIteration:
                break
            if nxt > self.maxval:
                break
            st.values.append(nxt)

        if st.values[-1] < x:
            return None
        return bisect.bisect_left(st.values, x)

    def _push_pair(self, i: int, j: int) -> None:
        if (i, j) in self._in_heap:
            return
        x = self._try_child_value(self.left, i)
        y = self._try_child_value(self.right, j)
        if x is None or y is None:
            return
        if x > y:
            return
        prod = x * y
        if prod > self.maxval:
            return
        self._in_heap.add((i, j))
        heapq.heappush(self._heap, (prod, i, j))

    def _start_node(self) -> None:
        if self._started:
            return
        self._started = True
        # Add i=0 list
        x0 = self._try_child_value(self.left, 0)
        if x0 is None:
            return
        j0 = self._ensure_right_ge(x0)
        if j0 is not None:
            self._push_pair(0, j0)
        self._next_i_to_add = 1

    def _maybe_add_more_i(self, current_min: int) -> None:
        """Add new i-lists whose lower bound left[i]^2 could affect ordering."""
        while True:
            x = self._try_child_value(self.left, self._next_i_to_add)
            if x is None:
                return
            # Lower bound for any pair with this x is x*x (since y >= x)
            if x * x > current_min and self._heap:
                return
            j0 = self._ensure_right_ge(x)
            if j0 is not None:
                self._push_pair(self._next_i_to_add, j0)
            self._next_i_to_add += 1

    def _next_candidate_product(self) -> int:
        self._start_node()

        while True:
            if not self._heap:
                # Try to add the next i; if that fails, stop.
                x = self._try_child_value(self.left, self._next_i_to_add)
                if x is None:
                    raise StopIteration
                j0 = self._ensure_right_ge(x)
                if j0 is None:
                    raise StopIteration
                self._push_pair(self._next_i_to_add, j0)
                self._next_i_to_add += 1
                continue

            prod, i, j = heapq.heappop(self._heap)
            # Ensure heap includes all i that could have first product <= this prod
            self._maybe_add_more_i(prod)

            # Advance along this i-list
            self._push_pair(i, j + 1)
            return prod

    def _next_value(self) -> int:
        if self._gen_kind == "leaf":
            return self._yield_next_prime()

        last = self.values[-1] if self.values else None
        while True:
            cand = self._next_candidate_product()
            if last is not None and cand == last:
                continue
            # Filter by actual factor-tree shape
            if shape_of(cand) == self.sh:
                return cand


_sequences: Dict[Shape, ShapeSequence] = {}


def _seq(sh: Shape) -> ShapeSequence:
    return _sequences[sh]


def get_value(sh: Shape, idx: int) -> int:
    """Get idx-th smallest value (0-indexed) that realizes shape sh."""
    st = _sequences[sh]
    while len(st.values) <= idx:
        st.values.append(st._next_value())
    return st.values[idx]


# ---------------------------
# Main computation
# ---------------------------


def main() -> None:
    max_n = 31
    maxval = double_factorial(max_n)

    # First, build all target shapes (and subshapes) from n!!
    targets: List[Shape] = []
    for n in range(2, max_n + 1):
        sh = shape_of(double_factorial(n))
        targets.append(sh)

    # Collect all unique subshapes appearing in the target shapes.
    all_shapes: set[Shape] = set()

    def collect(sh: Shape) -> None:
        if sh in all_shapes:
            return
        all_shapes.add(sh)
        if sh is LEAF:
            return
        collect(sh[0])  # type: ignore[index]
        collect(sh[1])  # type: ignore[index]

    for sh in targets:
        collect(sh)

    # Create sequences for all shapes, in increasing leaf-count order
    ordered = sorted(all_shapes, key=count_leaves)
    for sh in ordered:
        _sequences[sh] = ShapeSequence(sh, maxval)

    # Compute M(n) via the 0th element of each target shape's sequence
    m_values: Dict[int, int] = {}
    for n, sh in zip(range(2, max_n + 1), targets):
        m_values[n] = get_value(sh, 0)

    # Assert the example from the problem statement
    assert m_values[9] == 72

    total = sum(m_values[n] for n in range(2, max_n + 1))
    print(total)


if __name__ == "__main__":
    main()
