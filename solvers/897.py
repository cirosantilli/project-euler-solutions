#!/usr/bin/env python3
"""
Project Euler 897: Maximal n-gon in a region

Region:
    R = {(x,y) in R^2 : x^4 <= y <= 1}

Let G(n) be the largest possible area of an n-gon contained in R.

Key reduction (explained in README):
- An optimal polygon uses the top boundary y=1 as an edge between (-1,1) and (1,1),
  and all other vertices lie on the curve y = x^4.
- With x0=-1 < x1 < ... < x_{n-2} < x_{n-1}=1 on that curve, the polygon’s lower
  boundary is the polyline through (xi, xi^4), and area becomes:

    G(n) = 2 - sum_{i=0..n-2} (x_{i+1}-x_i) * (x_i^4 + x_{i+1}^4)/2

So maximizing area is minimizing that trapezoid sum. First-order optimality yields
a tridiagonal nonlinear system solved efficiently via damped Newton + Thomas solver.

No external libraries are used.
"""

import math


def _cbrt(x: float) -> float:
    """Real cube root preserving sign."""
    if x >= 0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


def _tridiagonal_solve(lower, diag, upper, rhs):
    """
    Solve tridiagonal system using Thomas algorithm.
    lower: length n-1 (subdiagonal)
    diag : length n   (diagonal)
    upper: length n-1 (superdiagonal)
    rhs  : length n
    Returns x length n.
    """
    n = len(diag)
    c = upper[:]  # will be modified
    d = diag[:]
    b = rhs[:]

    for i in range(n - 1):
        w = lower[i] / d[i]
        d[i + 1] -= w * c[i]
        b[i + 1] -= w * b[i]

    x = [0.0] * n
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


def _max_residual(xs):
    """Max absolute residual of stationarity equations for internal nodes."""
    n = len(xs)
    mx = 0.0
    for k in range(1, n - 1):
        a = xs[k - 1]
        x = xs[k]
        b = xs[k + 1]
        g = a**4 - b**4 + 4.0 * (x**3) * (b - a)
        if abs(g) > mx:
            mx = abs(g)
    return mx


def _newton_solve(xs, max_iter=200, tol=1e-15):
    """
    Damped Newton solver for the tridiagonal stationarity system.

    Unknowns are xs[1:-1], endpoints xs[0]=-1 and xs[-1]=1 fixed.
    """
    n = len(xs)
    m = n - 2  # number of unknowns

    xs = xs[:]  # copy

    for _ in range(max_iter):
        # Build residual g and tridiagonal Jacobian J
        g = [0.0] * m
        diag = [0.0] * m
        lower = [0.0] * (m - 1)
        upper = [0.0] * (m - 1)

        for k in range(1, n - 1):
            a = xs[k - 1]
            x = xs[k]
            b = xs[k + 1]
            idx = k - 1

            # residual
            g[idx] = a**4 - b**4 + 4.0 * (x**3) * (b - a)

            # partials:
            # dg/dx_k = 12 x_k^2 (b-a)
            diag[idx] = 12.0 * (x**2) * (b - a)

            # dg/dx_{k-1} = 4 a^3 - 4 x^3
            if idx - 1 >= 0:
                lower[idx - 1] = 4.0 * (a**3) - 4.0 * (x**3)

            # dg/dx_{k+1} = 4 x^3 - 4 b^3
            if idx + 1 < m:
                upper[idx] = 4.0 * (x**3) - 4.0 * (b**3)

        maxg = max(abs(v) for v in g)
        if maxg < tol:
            break

        # Solve J * delta = -g
        rhs = [-v for v in g]
        delta = _tridiagonal_solve(lower, diag, upper, rhs)

        # Damping / line search to preserve ordering and reduce residual
        alpha = 1.0
        while alpha > 1e-14:
            trial = xs[:]
            for i in range(m):
                trial[i + 1] = xs[i + 1] + alpha * delta[i]

            # monotonic & within bounds
            ok = True
            for i in range(n - 1):
                if not (trial[i] < trial[i + 1]):
                    ok = False
                    break
            if ok and (-1.0 < trial[1] < 1.0) and (-1.0 < trial[-2] < 1.0):
                if _max_residual(trial) < maxg:
                    xs = trial
                    break

            alpha *= 0.5

        else:
            # If we fail to find a better step, accept tiny movement and continue
            for i in range(m):
                xs[i + 1] += 1e-16 * delta[i]

    return xs


def _kahan_sum(values):
    """Kahan compensated summation for improved numeric stability."""
    s = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def area_G(n: int) -> float:
    """
    Compute G(n) as a float.
    """
    if n < 3:
        raise ValueError("n must be >= 3")

    # Initial guesses.
    # Empirically and structurally, odd n has two asymmetric global optima (mirror images),
    # while even n tends to converge robustly from a symmetric guess.
    candidates = []

    if n % 2 == 0:
        # Symmetric initial guess using a curvature-inspired transform:
        # x ~ sign(t)*|t|^(3/5) for t uniformly in [-1,1].
        xs = [0.0] * n
        for i in range(n):
            t = -1.0 + 2.0 * i / (n - 1)
            if t == 0.0:
                xs[i] = 0.0
            else:
                xs[i] = math.copysign(abs(t) ** (3.0 / 5.0), t)
        xs[0] = -1.0
        xs[-1] = 1.0
        sol = _newton_solve(xs)
        candidates.append(sol)

    else:
        # Asymmetric initial guesses:
        # split nodes unevenly between negative and positive side to avoid forcing a node at x=0.
        m = (n - 1) // 2
        # internal count is (n-2)=2m-1, so one side gets m and the other m-1.
        for extra_left in (True, False):
            left_internal = m
            right_internal = m - 1
            if not extra_left:
                left_internal, right_internal = right_internal, left_internal

            xs = [-1.0]
            # Use uniform spacing in z = |x|^(5/3) (inverse gives x = z^(3/5)).
            for j in range(1, left_internal + 1):
                z = 1.0 - j / (left_internal + 1)
                xs.append(-(z ** (3.0 / 5.0)))

            for j in range(1, right_internal + 1):
                z = j / (right_internal + 1)
                xs.append(z ** (3.0 / 5.0))

            xs.append(1.0)
            xs.sort()

            sol = _newton_solve(xs)
            candidates.append(sol)

    # Evaluate area from each candidate and take the best.
    best = None
    for xs in candidates:
        # trapezoid integral of x^4 along the polyline
        terms = []
        for a, b in zip(xs, xs[1:]):
            terms.append((b - a) * (a**4 + b**4) * 0.5)
        bottom = _kahan_sum(terms)
        area = 2.0 - bottom
        if best is None or area > best:
            best = area

    return best


def main():
    # Tests from problem statement
    g3 = area_G(3)
    assert abs(g3 - 1.0) < 1e-12, g3

    g5 = area_G(5)
    assert abs(g5 - 1.477309771) < 5e-10, g5

    # Required output
    ans = area_G(101)
    print(f"{ans:.9f}")


if __name__ == "__main__":
    main()
