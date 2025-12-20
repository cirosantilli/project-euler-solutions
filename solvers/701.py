#!/usr/bin/env python3
"""
Project Euler 701: Random Connected Area

We consider a W x H grid of cells. Each cell is independently black with
probability 1/2, otherwise white. Black cells sharing an edge are connected.
Let E(W,H) be the expected size (area) of the largest connected black component.

We compute E(7,7) rounded to 8 decimal places.

No external libraries are used (standard library only).
"""

from decimal import Decimal, getcontext, ROUND_HALF_UP


def _dp_expected_max_component(W: int, H: int):
    """
    Returns (numerator, denominator) for the exact expectation as a rational number:
        E(W,H) = numerator / denominator
    where denominator = 2^(W*H).

    Uses a sweep-line / frontier connectivity DP, processing cells in row-major order.
    The DP state keeps:
      - labels: tuple length W of component IDs along the frontier cut
      - sizes:  tuple of active component sizes (indexed by component ID-1)
      - a distribution over 'current maximum closed component size' for this frontier
    """
    N = W * H
    start_labels = (0,) * W
    start_sizes = ()
    start_conn = (start_labels, start_sizes)

    # dp maps (labels, sizes) -> dict{current_max: count_of_configurations}
    dp = {start_conn: {0: 1}}

    # Cache transitions: (labels, sizes, col, color) -> (labels2, sizes2, closed_max)
    # color: 0 = white, 1 = black
    trans_cache = {}

    for idx in range(N):
        c = idx % W
        new_dp = {}

        for (labels, sizes), mxmap in dp.items():
            left_id = labels[c - 1] if c else 0
            up_id = labels[c]
            k = len(sizes)

            for color in (0, 1):
                key = (labels, sizes, c, color)
                tr = trans_cache.get(key)

                if tr is None:
                    # mutable copies
                    nl = list(labels)
                    ns = list(sizes)

                    if color == 0:
                        # Current cell is white: overwrite frontier position with 0
                        nl[c] = 0
                    else:
                        # Current cell is black: connect to left/up if present
                        if left_id == 0 and up_id == 0:
                            ns.append(1)
                            nl[c] = len(ns)
                        elif up_id == 0:
                            ns[left_id - 1] += 1
                            nl[c] = left_id
                        elif left_id == 0:
                            ns[up_id - 1] += 1
                            nl[c] = up_id
                        else:
                            if left_id == up_id:
                                ns[left_id - 1] += 1
                                nl[c] = left_id
                            else:
                                # Merge two components via this new black cell.
                                a = left_id
                                b = up_id
                                ns[a - 1] = ns[a - 1] + ns[b - 1] + 1
                                ns[b - 1] = 0
                                # Relabel b -> a along the frontier
                                for j in range(W):
                                    if nl[j] == b:
                                        nl[j] = a
                                nl[c] = a

                    k2 = len(ns)

                    # Determine which component IDs remain present on the frontier.
                    present = [False] * (k2 + 1)
                    for lab in nl:
                        if lab:
                            present[lab] = True

                    # Close any component that is no longer present.
                    closed_max = 0
                    for comp_id in range(1, k2 + 1):
                        sz = ns[comp_id - 1]
                        if sz and not present[comp_id]:
                            if sz > closed_max:
                                closed_max = sz
                            ns[comp_id - 1] = 0

                    # Canonicalize IDs by first appearance in nl (left-to-right).
                    mapping = [0] * (k2 + 1)
                    canon_sizes = []
                    canon_labels = [0] * W
                    next_id = 0
                    for j, lab in enumerate(nl):
                        if lab == 0:
                            continue
                        nid = mapping[lab]
                        if nid == 0:
                            next_id += 1
                            mapping[lab] = next_id
                            canon_sizes.append(ns[lab - 1])
                            nid = next_id
                        canon_labels[j] = nid

                    labels2 = tuple(canon_labels)
                    sizes2 = tuple(canon_sizes)
                    tr = (labels2, sizes2, closed_max)
                    trans_cache[key] = tr

                labels2, sizes2, closed_max = tr
                conn2 = (labels2, sizes2)

                tgt = new_dp.get(conn2)
                if tgt is None:
                    tgt = {}
                    new_dp[conn2] = tgt

                # Update max distribution efficiently: new_max = max(old_max, closed_max)
                if closed_max == 0:
                    for mx, cnt in mxmap.items():
                        tgt[mx] = tgt.get(mx, 0) + cnt
                else:
                    for mx, cnt in mxmap.items():
                        mx2 = mx if mx >= closed_max else closed_max
                        tgt[mx2] = tgt.get(mx2, 0) + cnt

        dp = new_dp

    denom = 1 << N
    numer = 0

    # Finish: after last cell, all remaining active components close.
    for (labels, sizes), mxmap in dp.items():
        active_max = max(sizes) if sizes else 0
        if active_max == 0:
            for mx, cnt in mxmap.items():
                numer += mx * cnt
        else:
            for mx, cnt in mxmap.items():
                numer += (mx if mx >= active_max else active_max) * cnt

    return numer, denom


def expected_value_decimal(W: int, H: int, places: int = 8) -> Decimal:
    """
    Compute E(W,H) exactly as a rational and return it rounded to `places` decimals
    as a Decimal.

    Rounding uses ROUND_HALF_UP to match typical Euler rounding expectations.
    """
    numer, denom = _dp_expected_max_component(W, H)
    # plenty of precision for division and rounding
    getcontext().prec = 80
    value = Decimal(numer) / Decimal(denom)
    q = Decimal("1." + "0" * places)
    return value.quantize(q, rounding=ROUND_HALF_UP)


def main():
    # Tests from the problem statement
    n22, d22 = _dp_expected_max_component(2, 2)
    # E(2,2) = 1.875 = 15/8 => in /16 denominator form: 30/16
    assert d22 == 16
    assert n22 == 30

    v44 = expected_value_decimal(4, 4, places=8)
    assert str(v44) == "5.76487732"

    # Required output
    ans = expected_value_decimal(7, 7, places=8)
    print(ans)


if __name__ == "__main__":
    main()
