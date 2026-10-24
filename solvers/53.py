#!/usr/bin/env python
def solve(limit_n: int, threshold: int) -> int:
    """
    @formalsw: Returns
    @formalsw: Py from project_euler_statements.p53 import naive
    @formalsw: Equiv(naive)
    """
    if threshold < 1:
        return limit_n * (limit_n + 3) // 2 if limit_n > 0 else 0
    total = 0
    for n in range(1, limit_n + 1):
        c = 1  # C(n, 0)
        for r in range(1, n // 2 + 1):
            # C(n, r) = C(n, r-1) * (n-r+1) / r
            c = c * (n - r + 1) // r
            if c > threshold:
                # By symmetry and unimodality, all r in [r, n-r] exceed threshold
                total += n - 2 * r + 1
                break
    return total


if __name__ == "__main__":
    assert solve(1, 0) == 2
    assert solve(2, -1) == 5
    assert solve(-1, 0) == 0
    print(solve(100, 1_000_000))
