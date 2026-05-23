#!/usr/bin/env python

from math import comb


MOD = 1_000_036_000_099
P1 = 1_000_003
P2 = 1_000_033


def build_ways(active: int, passive: int, mod: int | None = None) -> list[int]:
    ways = [0] * (active + passive + 1)
    for active_ones in range(0, active + 1, 2):
        active_choices = comb(active, active_ones)
        for passive_ones in range(passive + 1):
            ones = active_ones + passive_ones
            value = active_choices * comb(passive, passive_ones)
            if mod is None:
                ways[ones] += value
            else:
                ways[ones] = (ways[ones] + value) % mod
    return ways


def count_with_ways(total_sum: int, ways: list[int], mod: int | None = None) -> int:
    terms = [(ones, count) for ones, count in enumerate(ways) if count]
    dp = [1]

    for bit_index in range(total_sum.bit_length()):
        target_bit = (total_sum >> bit_index) & 1
        next_dp = [0] * (len(dp) + len(ways) // 2 + 2)

        for carry, value in enumerate(dp):
            if value == 0:
                continue
            for ones, count in terms:
                column_total = carry + ones
                if (column_total & 1) != target_bit:
                    continue

                next_carry = (column_total - target_bit) // 2
                if mod is None:
                    next_dp[next_carry] += value * count
                else:
                    next_dp[next_carry] = (next_dp[next_carry] + value * count) % mod

        dp = next_dp

    return dp[0]


def binomial_mod_prime_small_k(n: int, k: int, prime: int) -> int:
    if k < 0 or k > n:
        return 0

    k = min(k, n - k)
    numerator = 1
    denominator = 1
    offset = n - k
    for i in range(1, k + 1):
        numerator = numerator * (offset + i) % prime
        denominator = denominator * i % prime
    return numerator * pow(denominator, prime - 2, prime) % prime


def losing_count(n: int, c: int, mod: int | None = None) -> int:
    if c % 2 != 0:
        raise ValueError("This implementation handles the even-c cases used here.")

    coin_count = c + 1
    empty_squares = n - coin_count
    active = (coin_count + 1) // 2
    passive = coin_count - active + 1

    active_ways = build_ways(active, passive, mod)
    reduced_active_ways = build_ways(active - 1, passive, mod)

    second_coin = count_with_ways(empty_squares, active_ways, mod)
    other_coins = count_with_ways(empty_squares + 1, active_ways, mod)
    other_coins -= count_with_ways(empty_squares + 1, reduced_active_ways, mod)

    if mod is None:
        return second_coin + (coin_count - 2) * other_coins
    return (second_coin + (coin_count - 2) * other_coins) % mod


def W_exact_even_c(n: int, c: int) -> int:
    coin_count = c + 1
    return coin_count * comb(n, coin_count) - losing_count(n, c)


def W_mod_even_c(n: int, c: int) -> int:
    coin_count = c + 1

    def solve_prime(prime: int) -> int:
        total = coin_count * binomial_mod_prime_small_k(n, coin_count, prime)
        return (total - losing_count(n, c, prime)) % prime

    residue_1 = solve_prime(P1)
    residue_2 = solve_prime(P2)

    t = ((residue_2 - residue_1) % P2) * pow(P1, -1, P2) % P2
    return (residue_1 + P1 * t) % MOD


def main() -> None:
    assert W_exact_even_c(10, 2) == 324
    assert W_exact_even_c(100, 10) == 1_514_704_946_113_500

    print(W_mod_even_c(1_000_000, 100))


if __name__ == "__main__":
    main()
