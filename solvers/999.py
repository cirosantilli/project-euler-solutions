MOD = 1_234_567_891
INV_TWO = (MOD + 1) // 2

# W_0 through W_8 for the rescaled elliptic divisibility sequence.
# Negative indices use W_{-n} = -W_n.
_SMALL_W = (0, 1, 2, -4, -32, -192, 3584, 77824, 262144)


def _small_w(index):
    """Return a small W-index modulo MOD, including negative indices."""
    if index < 0:
        return -_small_w(-index) % MOD
    return _SMALL_W[index] % MOD


def _eds_block(n):
    """Return (W_{n-3}, ..., W_{n+4}) modulo MOD in O(log n) time."""
    if n <= 4:
        return tuple(_small_w(index) for index in range(n - 3, n + 5))

    middle = n // 2
    source = _eds_block(middle)
    source_start = middle - 3

    def get(index):
        return source[index - source_start]

    def odd(index):
        """Return W_{2*index-1} from nearby terms around W_index."""
        return (
            get(index + 1) * pow(get(index - 1), 3, MOD)
            - get(index - 2) * pow(get(index), 3, MOD)
        ) % MOD

    def even(index):
        """Return W_{2*index}; division is only by the fixed W_2 = 2."""
        return (
            get(index)
            * INV_TWO
            * (
                get(index + 2) * pow(get(index - 1), 2, MOD)
                - get(index - 2) * pow(get(index + 1), 2, MOD)
            )
        ) % MOD

    if n % 2 == 0:
        return (
            odd(middle - 1),
            even(middle - 1),
            odd(middle),
            even(middle),
            odd(middle + 1),
            even(middle + 1),
            odd(middle + 2),
            even(middle + 2),
        )

    return (
        even(middle - 1),
        odd(middle),
        even(middle),
        odd(middle + 1),
        even(middle + 1),
        odd(middle + 2),
        even(middle + 2),
        odd(middle + 3),
    )


def _w_mod(n):
    """Return W_n modulo MOD."""
    if n < 0:
        return -_w_mod(-n) % MOD
    return _eds_block(n)[3]


def a_mod(n):
    """Return a_n modulo MOD for n >= 1."""
    assert n >= 1
    sign = 1 if n % 4 in (1, 2) else -1
    inverse_scale = pow(INV_TWO, n * n // 4, MOD)
    return sign * _w_mod(n) * inverse_scale % MOD


def main():
    assert a_mod(1) == 1
    assert a_mod(2) == 1
    assert a_mod(3) == 1
    assert a_mod(4) == 2
    assert a_mod(13) == 23321
    assert a_mod(1003) == 231906014

    print(a_mod(10**18 + 3))


if __name__ == "__main__":
    main()
