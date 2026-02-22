#!/usr/bin/env python3
import math

EPS = 1e-12


def triangle_angles_from_sides(a: int, b: int, c: int):
    """
    Return the three angles (in radians) of a triangle with side lengths a,b,c
    using the law of cosines.

    The sides are assumed to satisfy the triangle inequality.
    """
    af = float(a)
    bf = float(b)
    cf = float(c)

    def clamp(x: float) -> float:
        # Guard against tiny numerical overshoots so acos stays defined.
        if x < -1.0:
            return -1.0
        if x > 1.0:
            return 1.0
        return x

    cosA = clamp((bf * bf + cf * cf - af * af) / (2.0 * bf * cf))
    cosB = clamp((af * af + cf * cf - bf * bf) / (2.0 * af * cf))
    cosC = clamp((af * af + bf * bf - cf * cf) / (2.0 * af * bf))

    A = math.acos(cosA)
    B = math.acos(cosB)
    C = math.acos(cosC)

    # A basic sanity check: A+B+C should be pi.
    if not (abs((A + B + C) - math.pi) < 1e-7):
        # For valid integer-sided triangles this should not happen; if it does,
        # something went badly wrong numerically.
        raise RuntimeError("Angle sum is not π; check input or numerics.")

    return A, B, C


def next_angles(A: float, B: float, C: float):
    """
    Given the angles (in radians) of T_k, return angles of T_{k+1}.

    From the geometry (sides of T_k are external angle bisectors of T_{k+1}),
    if (A_k, B_k, C_k) are the angles of T_k then

        A_{k+1} = π - 2 B_k
        B_{k+1} = π - 2 C_k
        C_{k+1} = π - 2 A_k

    (Angles are taken in radians.)
    """
    return (
        math.pi - 2.0 * B,
        math.pi - 2.0 * C,
        math.pi - 2.0 * A,
    )


def num_existing_steps(a: int, b: int, c: int, max_steps: int) -> int:
    """
    Starting from triangle T_0 with integer sides (a,b,c), iterate the
    angle transformation until it becomes impossible to form the next
    triangle.

    Return the largest n such that T_n exists (with n >= 0), counted as
    the number of *steps beyond T_0*:

        - If T_1 does not exist, return 0.
        - If T_1, ..., T_N exist but T_{N+1} does not, return N.

    This function never returns a value greater than max_steps, to avoid
    infinite loops for perfectly equilateral triangles.
    """
    A, B, C = triangle_angles_from_sides(a, b, c)

    steps = 0
    for _ in range(max_steps):
        A, B, C = next_angles(A, B, C)
        # For T_{k+1} to exist we need all angles strictly positive.
        if A <= EPS or B <= EPS or C <= EPS:
            break
        steps += 1

    return steps


def find_min_perimeter_for_steps(target_steps: int, max_perimeter: int):
    """
    Brute-force search over all integer-sided triangles with perimeter
    up to max_perimeter.

    Return (min_perimeter, list_of_triangles) where each triangle is a
    sorted (a,b,c) triple, and every listed triangle satisfies

        - T_{target_steps} exists, and
        - T_{target_steps+1} does not exist.

    If no such triangle exists up to max_perimeter, return (None, []).
    """
    best_perimeter = None
    best_triangles = []

    # Enumerate by perimeter to naturally go from small to large.
    for p in range(3, max_perimeter + 1):
        # Early stop if we've already found something and moved past it.
        if best_perimeter is not None and p > best_perimeter:
            break

        # Enumerate integer triangles with perimeter exactly p.
        # Use a <= b <= c and a + b + c = p with triangle inequality.
        for a in range(1, p // 3 + 1):
            # b at least a; at most so that c >= b and triangle inequality holds.
            for b in range(a, (p - a) // 2 + 1):
                c = p - a - b
                if c < b:
                    continue
                # Triangle inequality: a + b > c.
                if a + b <= c:
                    continue

                steps = num_existing_steps(a, b, c, target_steps + 2)

                if steps == target_steps:
                    if best_perimeter is None or p < best_perimeter:
                        best_perimeter = p
                        best_triangles = [tuple(sorted((a, b, c)))]
                    elif p == best_perimeter:
                        best_triangles.append(tuple(sorted((a, b, c))))

    # Remove duplicate triangles (same side multiset in different order).
    best_triangles = sorted(set(best_triangles))
    return best_perimeter, best_triangles


def solve_main_problem(target_steps: int = 20):
    """
    For the main question: T_0 has integer sides, T_target_steps exists but
    T_{target_steps+1} does not. We want the smallest possible perimeter.

    Heuristic + geometric reasoning:

      * Deep chains of telescoping triangles require T_0 to be extremely
        close to equilateral.
      * For fixed perimeter, the integer triangle closest to equilateral
        is of the form (n, n, n+1) or (n, n+1, n+1) up to permutation.
      * Therefore we search these two "almost equilateral" families:

            type1: (n, n, n+1) with perimeter 3n + 1
            type2: (n, n+1, n+1) with perimeter 3n + 2

        and keep the smallest perimeter that yields exactly target_steps
        valid telescoping steps.

    The search stops as soon as the minimal achievable perimeter for
    larger n exceeds the best perimeter we have already found.
    """
    best_perimeter = None
    best_triangle = None

    n = 2
    while True:
        # Two almost-equilateral families
        candidates = [
            (n, n, n + 1),     # type 1
            (n, n + 1, n + 1), # type 2
        ]

        for a, b, c in candidates:
            # All of these triples satisfy the triangle inequality for n >= 2.
            steps = num_existing_steps(a, b, c, target_steps + 2)
            if steps == target_steps:
                p = a + b + c
                if best_perimeter is None or p < best_perimeter:
                    best_perimeter = p
                    best_triangle = (a, b, c)

        n += 1

        if best_perimeter is not None:
            # The minimal future perimeter we can reach is from the smaller
            # of the two families at this n, which is 3n+1.
            # Once 3n+1 exceeds our best_perimeter, no later candidate can
            # possibly improve it.
            if 3 * n + 1 > best_perimeter:
                break

        # Safety net to avoid infinite loops if something goes wrong.
        # This should never trigger for the actual problem.
        if n > 5_000_000:
            raise RuntimeError("Search did not converge; check logic.")

    return best_perimeter, best_triangle


def main():
    # ------------------------------------------------------------------
    # Sanity checks from the problem statement
    # ------------------------------------------------------------------

    # Example 1: T0 with sides (8,9,10) has T2 but not T3.
    steps_8_9_10 = num_existing_steps(8, 9, 10, 10)
    assert steps_8_9_10 == 2, f"Expected 2 steps for (8,9,10), got {steps_8_9_10}"

    # Example 2: Among all integer triangles with T2 existing but T3 not,
    # the smallest perimeter is 10 with sides (3,3,4).
    min_p_2, tris_2 = find_min_perimeter_for_steps(2, max_perimeter=50)
    assert min_p_2 == 10, f"Expected perimeter 10 for target_steps=2, got {min_p_2}"
    assert (3, 3, 4) in tris_2, "Expected triangle (3,3,4) to be among minimisers for target_steps=2"

    # ------------------------------------------------------------------
    # Solve the main problem for 20 steps
    # ------------------------------------------------------------------
    best_perimeter, best_triangle = solve_main_problem(target_steps=20)

    # Do NOT hard-code or assert this value; just print it.
    print(best_perimeter)


if __name__ == "__main__":
    main()
