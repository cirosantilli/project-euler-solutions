MOD = 987_898_789


def build_combinatorics(limit: int, mod: int):
    """Precompute factorials and inverse factorials modulo a prime mod."""
    fact = [1] * (limit + 1)
    for i in range(1, limit + 1):
        fact[i] = fact[i - 1] * i % mod

    inv_fact = [1] * (limit + 1)
    inv_fact[limit] = pow(fact[limit], mod - 2, mod)
    for i in range(limit, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod

    return fact, inv_fact


class Comb:
    def __init__(self, limit: int, mod: int):
        self.mod = mod
        self.fact, self.inv_fact = build_combinatorics(limit, mod)

    def __call__(self, n: int, r: int) -> int:
        if r < 0 or r > n:
            return 0
        return (
            self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod
        )


def endpoint_count(n: int, k: int, end: int, comb: Comb, mod: int) -> int:
    """
    Count valid journeys that end on a fixed stone `end`.

    Let right[i] be the number of traversals i -> i+1 across edge i.
    For i < n these satisfy:
        right[0] = k - [end = 0]
        right[1] = 2 - [end = 1]
        right[i] = 1 + right[i-2] - [end = i]   for i >= 2

    For each internal stone v (1 <= v < n), the walk is determined by the order of
    its left/right departures. The last departure is forced toward the endpoint, so
    the number of valid local orders is a binomial coefficient.
    """
    if n == 0:
        return 1

    right = [0] * n
    right[0] = k - (1 if end == 0 else 0)
    if n >= 2:
        right[1] = 2 - (1 if end == 1 else 0)
    for i in range(2, n):
        right[i] = 1 + right[i - 2] - (1 if end == i else 0)

    ways = 1
    for v in range(1, n):
        out_degree = k + v - (1 if end == v else 0)
        if v < end:
            # Last departure from v must be to the right.
            ways = ways * comb(out_degree - 1, right[v] - 1) % mod
        elif v == end:
            # No forced last departure at the endpoint.
            ways = ways * comb(out_degree, right[v]) % mod
        else:
            # Last departure from v must be to the left.
            ways = ways * comb(out_degree - 1, right[v]) % mod
    return ways


def journey_count(n: int, k: int, comb: Comb, mod: int = MOD) -> int:
    total = 0
    for end in range(n + 1):
        total = (total + endpoint_count(n, k, end, comb, mod)) % mod
    return total


def solve() -> int:
    n = 500
    ks = [1, 10, 100, 1000, 10000]

    # For these inputs, all needed binomials have top <= max(k) + n.
    comb = Comb(max(ks) + n, MOD)

    assert journey_count(3, 2, comb, MOD) == 17
    assert journey_count(6, 1, comb, MOD) == 1320
    assert journey_count(6, 5, comb, MOD) == 16_793_280

    answer = 0
    for k in ks:
        answer = (answer + journey_count(n, k, comb, MOD)) % MOD
    return answer


if __name__ == "__main__":
    print(solve())
