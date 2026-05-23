#!/usr/bin/env python

from math import gcd, lcm


def coordinate_values(n):
    scale = 1
    for d in range(1, n + 1):
        scale = lcm(scale, d)

    values = {0}
    for d in range(2, n + 1):
        step = scale // d
        for a in range(1, d):
            if gcd(a, d) == 1:
                v = a * step
                values.add(v)
                values.add(-v)

    return scale, sorted(values)


def build_lifted_points(n):
    scale, coords = coordinate_values(n)
    scale2 = scale * scale
    squares = {x: x * x for x in coords}

    points = []
    for x in coords:
        x2 = squares[x]
        for y in coords:
            z0 = x2 + squares[y]
            if z0 < scale2:
                points.append((x, y, z0 + scale2))
    return points


def add_run(total, run_len):
    if run_len >= 2:
        total += run_len * (run_len - 1) // 2
    return total


def count_unordered_triples(points):
    total = 0
    m = len(points)

    for i in range(m - 2):
        px, py, pz = points[i]
        keys = []

        if px:
            for j in range(i + 1, m):
                qx, qy, qz = points[j]
                a = py * qx - px * qy
                b = pz * qx - px * qz
                g = gcd(abs(a), abs(b))
                a //= g
                b //= g
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                keys.append((a, b))
        else:
            for j in range(i + 1, m):
                qx, qy, qz = points[j]
                a = qx
                b = pz * qy - py * qz
                g = gcd(abs(a), abs(b))
                a //= g
                b //= g
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                keys.append((a, b))

        keys.sort()
        run_len = 1
        prev = keys[0]
        for key in keys[1:]:
            if key == prev:
                run_len += 1
            else:
                total = add_run(total, run_len)
                prev = key
                run_len = 1
        total = add_run(total, run_len)

    return total


def compute_T(n):
    points = build_lifted_points(n)
    return 6 * count_unordered_triples(points)


if __name__ == "__main__":
    assert compute_T(2) == 24
    assert compute_T(3) == 1296
    assert compute_T(4) == 5052
    print(compute_T(12))
