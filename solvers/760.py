#!/usr/bin/env python3
"""
Project Euler 760: Sum over Bitwise Operators

We need:
  g(m,n) = (m xor n) + (m or n) + (m and n)
  G(N)   = sum_{n=0..N} sum_{k=0..n} g(k, n-k)

Then print G(10^18) mod 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007


def _count_pairs_sum_leq(N, fixed_bit=None):
    """
    Count pairs (a,b) with a>=0, b>=0 and a+b <= N, modulo MOD.

    If fixed_bit is not None, additionally enforce that bit 'fixed_bit'
    of both a and b is 0.

    Implementation detail:
    We do a digit-DP over bits of the sum S=a+b to enforce S<=N, while
    reconstructing valid (a_bit, b_bit) choices. The carry in binary addition
    naturally flows from low bits to high bits, so we run the automaton "backwards":
      - process bits from MSB -> LSB
      - keep as state the carry into the already-processed higher bit (carry_next)
      - pick (a_bit,b_bit,carry_current) that produce that carry_next
      - propagate carry_current to the next (lower) bit.
    """
    if N < 0:
        return 0
    bits = max(1, N.bit_length())

    # dp[carry_next][less] for the prefix (higher bits already processed)
    # carry_next in {0,1}: carry into the next higher bit (already processed side)
    # less in {0,1}: whether the built sum prefix is already strictly less than N's prefix
    dp = [[0, 0], [0, 0]]
    dp[0][0] = 1

    for pos in range(bits - 1, -1, -1):
        nbit = (N >> pos) & 1
        ndp = [[0, 0], [0, 0]]

        for carry_next in (0, 1):
            for less in (0, 1):
                ways = dp[carry_next][less]
                if ways == 0:
                    continue

                for carry_cur in (0, 1):
                    for a_bit in (0, 1):
                        for b_bit in (0, 1):
                            if (
                                fixed_bit is not None
                                and pos == fixed_bit
                                and (a_bit | b_bit)
                            ):
                                continue

                            total = a_bit + b_bit + carry_cur
                            if (total >> 1) != carry_next:
                                continue

                            s_bit = total & 1
                            if less == 0 and s_bit > nbit:
                                continue

                            new_less = less or (s_bit < nbit)
                            ndp[carry_cur][new_less] = (
                                ndp[carry_cur][new_less] + ways
                            ) % MOD

        dp = ndp

    # After processing all bits, the carry into the (non-existent) bit below LSB must be 0.
    return (dp[0][0] + dp[0][1]) % MOD


def G_mod(N):
    """
    Compute G(N) modulo MOD.
    """
    inv2 = (MOD + 1) // 2  # modular inverse of 2 since MOD is prime

    # Total number of pairs (a,b) with a+b <= N equals (N+1)(N+2)/2
    total_pairs = ((N + 1) % MOD) * ((N + 2) % MOD) % MOD
    total_pairs = total_pairs * inv2 % MOD

    bits = max(1, N.bit_length())
    pow2 = 1
    sum_or = 0

    # Sum (a|b) over all pairs is sum over bits:
    #   2^i * (#pairs where OR-bit i is 1)
    # OR-bit i is 0 iff a_i=0 and b_i=0.
    for i in range(bits):
        both_zero = _count_pairs_sum_leq(N, fixed_bit=i)
        bit_is_one = (total_pairs - both_zero) % MOD
        sum_or = (sum_or + pow2 * bit_is_one) % MOD
        pow2 = (pow2 * 2) % MOD

    # g(m,n) simplifies to 2*(m|n), therefore G(N) = 2*sum_or
    return (2 * sum_or) % MOD


def _self_test():
    # Test values from the problem statement
    assert G_mod(10) == 754
    assert G_mod(10**2) == 583766


def main():
    _self_test()

    # Default target for Project Euler is 10^18.
    N = 10**18

    # Optional CLI: allow computing G(N) for a user-supplied N.
    # Usage: python3 main.py [N]
    import sys

    if len(sys.argv) >= 2:
        N = int(sys.argv[1])

    print(G_mod(N))


if __name__ == "__main__":
    main()
