#!/usr/bin/env python
"""
Project Euler 958: Euclid's Labour

The subtraction Euclidean algorithm corresponds to walking in the Stern-Brocot
tree.  Instead of searching all reverse paths from one end, split a candidate
path near the middle.  A midpoint basis (a, b) represents n as

    n = a * alpha + b * beta

and the opposite half of the path is recovered from nearby nonnegative
coefficient vectors.  This meet-in-the-middle viewpoint makes the required
depth search tiny for the Euler input.
"""


def modular_inverse(value: int, modulus: int) -> tuple[bool, int]:
    value %= modulus
    old_r, r = modulus, value
    old_s, s = 0, 1

    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s

    if old_r != 1:
        return False, 0
    return True, old_s % modulus


def f(n: int) -> int:
    steps = 0

    while True:
        best_steps = steps
        best_value = n + 1
        split_depth = (steps + 1) // 2

        def consider(
            basis_a: int,
            basis_b: int,
            coeff_a: int,
            coeff_b: int,
            current_depth: int,
        ) -> None:
            nonlocal best_steps, best_value

            if basis_a > basis_b:
                basis_a, basis_b = basis_b, basis_a
                coeff_a, coeff_b = coeff_b, coeff_a

            if coeff_b < 0:
                return
            if coeff_a < 0:
                shift = (-coeff_a + basis_b - 1) // basis_b
                coeff_a += shift * basis_b
                coeff_b -= shift * basis_a
                if coeff_b < 0:
                    return

            if basis_a * coeff_a + basis_b * coeff_b != n:
                return

            if current_depth == split_depth:
                local_a = basis_a
                local_b = basis_b
                local_ca = coeff_a
                local_cb = coeff_b

                if steps & 1:
                    local_a *= 2
                    local_b *= 2
                    local_ca *= 2
                    local_cb *= 2
                    local_b -= local_a // 2
                    local_ca += local_cb // 2
                    if local_a * local_ca + local_b * local_cb != 4 * n:
                        return

                norm = local_a * local_a + local_b * local_b
                if norm < n:
                    return

                cross = local_ca * local_b - local_cb * local_a
                shift = cross // norm
                local_ca -= shift * local_b
                local_cb += shift * local_a
                cross -= shift * norm

                if cross < 0:
                    local_ca += local_b
                    local_cb -= local_a

                def check(candidate_a: int, candidate_b: int) -> None:
                    nonlocal best_steps, best_value

                    if candidate_a < 0 or candidate_b < 0:
                        return
                    if candidate_a * candidate_a + candidate_b * candidate_b > norm:
                        return

                    x = candidate_a
                    y = candidate_b
                    vx = local_a
                    vy = local_b

                    if steps & 1:
                        if (vx & 1) or (y & 1):
                            return
                        x -= y // 2
                        vy += vx // 2
                        if (vy & 1) or (x & 1):
                            return
                        x //= 2
                        y //= 2
                        vx //= 2
                        vy //= 2

                    if x * vx + y * vy != n:
                        return

                    remaining_steps = steps - split_depth
                    used_steps = 0
                    while used_steps <= remaining_steps and x and y:
                        if x > y:
                            x, y = y, x
                            vx, vy = vy, vx
                        y -= x
                        vx += vy
                        used_steps += 1

                    if used_steps > remaining_steps:
                        return

                    residue = (vx + vy - n) % n
                    ok, inv_residue = modular_inverse(residue, n)
                    if not ok:
                        return

                    total_steps = current_depth + used_steps
                    value = min(residue, n - residue, inv_residue, n - inv_residue)

                    if total_steps < best_steps or (
                        total_steps == best_steps and value < best_value
                    ):
                        best_steps = total_steps
                        best_value = value

                check(local_ca, local_cb)
                check(local_ca - local_b, local_cb + local_a)
                return

            x = basis_a
            y = basis_b
            for _ in range(current_depth, steps // 2):
                if x > y:
                    x, y = y, x
                x += y
                x, y = y, x
            if x > y:
                x, y = y, x

            if steps & 1:
                if 5 * y * y // 4 + x * y + x * x < n:
                    return
            elif x * x + y * y < n:
                return

            consider(basis_b, basis_a + basis_b, coeff_b - coeff_a, coeff_a, current_depth + 1)
            if 0 < basis_a < basis_b:
                consider(basis_a, basis_a + basis_b, coeff_a - coeff_b, coeff_b, current_depth + 1)

        consider(0, 1, 0, n, 0)

        if best_value <= n:
            return best_value

        steps += 1


def main() -> None:
    assert f(7) == 2
    assert f(89) == 34
    assert f(8191) == 1856
    print(f(10**12 + 39))


if __name__ == "__main__":
    main()
