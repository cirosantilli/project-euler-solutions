# Nagini
from nagini_contracts.contracts import Decreases, Ensures, Ghost, Pure, Requires, Result
from project_euler_statements.p1 import naive


# Nagini
@Ghost
@Pure
def multiples_step(m: int, n: int) -> bool:
    """
    Advancing the upper bound adds n exactly when m divides n.
    """
    Requires((m == 3 or m == 5 or m == 15) and n > 0)
    Decreases(0)
    Ensures(Result())
    Ensures(
        (m * (n // m) * (n // m + 1) // 2)
        == (m * ((n - 1) // m) * ((n - 1) // m + 1) // 2) + (n if n % m == 0 else 0)
    )
    q = n // m
    if n % m == 0:
        return (
            (n - 1) // m == q - 1
            and n == m * q
            and m * q * (q + 1) == m * (q - 1) * q + 2 * n
            and m * q * (q + 1) // 2 == m * (q - 1) * q // 2 + n
        )
    return (n - 1) // m == q


# Nagini
@Ghost
@Pure
def equivalence(n: int) -> bool:
    """
    Induct on n using inclusion-exclusion for multiples of 3 and 5.
    """
    Requires(n >= 0)
    Decreases(n)
    Ensures(Result())
    Ensures(
        naive(n)
        == (3 * ((n - 1) // 3) * ((n - 1) // 3 + 1) // 2)
        + (5 * ((n - 1) // 5) * ((n - 1) // 5 + 1) // 2)
        - (15 * ((n - 1) // 15) * ((n - 1) // 15 + 1) // 2)
    )
    if n == 0:
        return naive(0) == 0
    if n == 1:
        return naive(0) == 0 and naive(1) == 0
    return (
        equivalence(n - 1)
        and multiples_step(3, n - 1)
        and multiples_step(5, n - 1)
        and multiples_step(15, n - 1)
        and (((n - 1) % 15 == 0) == ((n - 1) % 3 == 0 and (n - 1) % 5 == 0))
    )
