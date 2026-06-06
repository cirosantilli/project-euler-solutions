"""Solve the three sub-problems and print M(1000) modulo 1_000_000_007.

The implementation uses only the Python standard language features.
"""

MODULUS = 1_000_000_007


def _ones_in_bit_from_zero_to_n(n: int, bit: int) -> int:
    """Count integers in [0, n] whose selected bit is set."""
    if n < 0:
        return 0
    half_period = 1 << bit
    period = half_period << 1
    value_count = n + 1
    full_periods, remainder = divmod(value_count, period)
    return full_periods * half_period + max(0, remainder - half_period)


def max_and(n: int) -> int:
    """Return I(n), the maximum cross-group sum of bitwise AND values."""
    if n <= 0:
        return 0

    total = 0
    for bit in range(n.bit_length()):
        count = _ones_in_bit_from_zero_to_n(n, bit)
        smaller_half = count // 2
        larger_half = count - smaller_half
        total += (1 << bit) * smaller_half * larger_half
    return total


def max_xor_sum(n: int) -> int:
    """Return X(n) using dynamic programming over increasing edge weights."""
    if n <= 1:
        return 0

    vertex_bits = max(1, n.bit_length())
    payload_bits = 2 * vertex_bits
    vertex_mask = (1 << vertex_bits) - 1
    squares = [value * value for value in range(n + 1)]

    # Each packed integer sorts first by edge weight and then by endpoints.
    # Keeping one Python integer per edge is much smaller than storing tuples.
    packed_edges = []
    for left in range(1, n + 1):
        left_square = squares[left]
        for right in range(left + 1, n + 1):
            weight = left_square ^ squares[right]
            packed_edges.append(
                (weight << payload_bits) | (left << vertex_bits) | right
            )
    packed_edges.sort()

    # best[vertex] is the maximum sum of a valid increasing walk ending there
    # using only edge weights already committed by earlier groups.
    best = [0] * (n + 1)
    pending = [-1] * (n + 1)

    index = 0
    edge_count = len(packed_edges)
    while index < edge_count:
        weight = packed_edges[index] >> payload_bits
        touched = []
        next_index = index

        # Equal-weight edges must all read the old best array: strict
        # increase means that they cannot be chained together.
        while (
            next_index < edge_count
            and packed_edges[next_index] >> payload_bits == weight
        ):
            packed = packed_edges[next_index]
            left = (packed >> vertex_bits) & vertex_mask
            right = packed & vertex_mask

            candidate_left = best[right] + weight
            if candidate_left > pending[left]:
                if pending[left] < 0:
                    touched.append(left)
                pending[left] = candidate_left

            candidate_right = best[left] + weight
            if candidate_right > pending[right]:
                if pending[right] < 0:
                    touched.append(right)
                pending[right] = candidate_right

            next_index += 1

        for vertex in touched:
            if pending[vertex] > best[vertex]:
                best[vertex] = pending[vertex]
            pending[vertex] = -1

        index = next_index

    return max(best)


_EVEN_XOR_PATTERNS = (0b000, 0b011, 0b101, 0b110)
_ALL_THREE_BIT_PATTERNS = tuple(range(8))


def _count_unreachable_with_highest_xor_bit(limit: int, pivot: int) -> int:
    """Count unreachable triples <= limit whose XOR has highest bit pivot."""
    bit_count = max(1, limit.bit_length())

    # A set bit means that the corresponding pile prefix is still tight.
    counts_by_tight_mask = [0] * 8
    counts_by_tight_mask[0b111] = 1

    for bit in range(bit_count - 1, -1, -1):
        if bit > pivot:
            patterns = _EVEN_XOR_PATTERNS
        elif bit == pivot:
            patterns = (0b111,)
        else:
            patterns = _ALL_THREE_BIT_PATTERNS

        limit_bit = (limit >> bit) & 1
        next_counts = [0] * 8

        for tight_mask, ways in enumerate(counts_by_tight_mask):
            if not ways:
                continue

            for pattern in patterns:
                next_tight_mask = 0
                valid = True

                for pile in range(3):
                    pile_mask = 1 << pile
                    if tight_mask & pile_mask:
                        chosen_bit = (pattern >> pile) & 1
                        if chosen_bit > limit_bit:
                            valid = False
                            break
                        if chosen_bit == limit_bit:
                            next_tight_mask |= pile_mask

                if valid:
                    next_counts[next_tight_mask] += ways

        counts_by_tight_mask = next_counts

    return sum(counts_by_tight_mask)


def count_unreachable_nim(n: int) -> int:
    """Return C(n), the number of unreachable triples with piles in [0, n)."""
    if n <= 0:
        return 0

    limit = n - 1
    bit_count = max(1, limit.bit_length())
    return sum(
        _count_unreachable_with_highest_xor_bit(limit, pivot)
        for pivot in range(bit_count)
    )


def meta_values(last_index: int) -> list[int]:
    """Compute M(0) through M(last_index), reducing products modulo MODULUS."""
    values = [
        max_and(1000) % MODULUS,
        max_xor_sum(1000) % MODULUS,
        count_unreachable_nim(1000) % MODULUS,
    ]

    while len(values) <= last_index:
        values.append(values[-1] * values[-2] * values[-3] % MODULUS)

    return values[: last_index + 1]


def main() -> None:
    # Statement examples.
    assert max_and(10) == 50
    assert max_xor_sum(4) == 71
    assert max_xor_sum(10) == 702
    assert count_unreachable_nim(10) == 123

    values = meta_values(1000)

    # Additional statement check for the combined recurrence.
    assert values[4] == 457_587_170

    # Print the requested final value without embedding or asserting it.
    print(values[1000])


if __name__ == "__main__":
    main()
