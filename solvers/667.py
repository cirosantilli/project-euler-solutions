#!/usr/bin/env python3
"""
Project Euler 667: Moving Pentagon

We search for the largest-area equilateral pentagon that can be pushed
around a unit-width L-shaped corridor without lifting.

The solver uses:
- A symmetric equilateral pentagon parameterized by one shape value r
  (ratio of the shortest diagonal to side length for the unit model)
- A feasibility test based on rotating the pentagon, then applying a
  canonical translation to "push" it into the corridor corner
- Nested numeric optimization:
    - inner bisection on scale
    - outer golden-section maximization on shape parameter
"""

import math


# --------------------------
# Basic geometry helpers
# --------------------------


def heron(a, b, c):
    s = (a + b + c) * 0.5
    v = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(v) if v > 0.0 else 0.0


def rotate_points(points, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [(c * x - s * y, s * x + c * y) for (x, y) in points]


def min_x_above_y(points, ythr):
    """
    Minimum x among points on the polygon boundary with y >= ythr.
    The minimum must occur at:
      - a vertex with y >= ythr
      - an edge intersection with y=ythr
    """
    mn = float("inf")
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        if y1 >= ythr and x1 < mn:
            mn = x1
        if y2 >= ythr and x2 < mn:
            mn = x2

        dy = y2 - y1
        if dy == 0.0:
            continue
        t = (ythr - y1) / dy
        if 0.0 <= t <= 1.0:
            x = x1 + (x2 - x1) * t
            if x < mn:
                mn = x
    return mn


# --------------------------
# Pentagon construction
# --------------------------


def build_unit_pentagon(r):
    """
    Build a symmetric equilateral pentagon with side length 1.

    Vertices (in order): A, B, C, D, E
    - A=(0,0), E=(1,0)
    - C is on the perpendicular bisector of AE such that AC=CE=r
    - B is chosen such that AB=BC=1 and triangle ABC has base AC=r
    - D is the mirror of B around x=0.5 axis

    This yields an equilateral pentagon (all edges length 1).
    """
    if not (0.5 < r < 2.0):
        return None

    h2 = r * r - 0.25
    if h2 <= 0.0:
        return None
    h = math.sqrt(h2)

    A = (0.0, 0.0)
    E = (1.0, 0.0)
    C = (0.5, h)

    # Intersection of two circles radius 1 around A and C
    d = r
    if d >= 2.0:
        return None
    k2 = 1.0 - (d * 0.5) * (d * 0.5)
    if k2 <= 0.0:
        return None
    k = math.sqrt(k2)

    Mx = (A[0] + C[0]) * 0.5
    My = (A[1] + C[1]) * 0.5

    # perpendicular unit direction to AC
    ux = -(C[1] - A[1]) / d
    uy = (C[0] - A[0]) / d

    # Choose the "left" intersection for B
    B = (Mx + k * ux, My + k * uy)
    # Mirror to get D
    D = (1.0 - B[0], B[1])

    return [A, B, C, D, E]


def base_area(r):
    """
    Area of the unit-side pentagon in terms of r (short diagonal length).
    Triangulation yields:
        2*Heron(1,1,r) + Heron(r,r,1)
    """
    return 2.0 * heron(1.0, 1.0, r) + heron(r, r, 1.0)


# --------------------------
# Feasibility test & scaling
# --------------------------


def precompute_rotations(points, thetas):
    out = []
    for th in thetas:
        pts = rotate_points(points, th)
        miny = min(y for (_, y) in pts)
        maxx = max(x for (x, _) in pts)
        out.append((pts, miny, maxx))
    return out


def clearance_for_theta(points_rot, min_y, max_x, scale, eps_y):
    """
    Corridor model:
      horizontal arm: 0 <= y <= 1, x <= 1 (extends left)
      vertical arm:   x in [0,1], y >= 0 (extends up)

    Canonical placement:
      shift so min y = 0  (touch bottom wall)
      shift so max x = 1  (touch right wall)

    Forbidden region becomes: x < 0 and y > 1
    Which is equivalent to: among y > 1 points, require x >= 0.

    In rotated *unscaled* coordinates:
      y' = scale*(y - min_y)
      x' = 1 + scale*(x - max_x)

    y' > 1+eps  <=>  y > min_y + (1+eps)/scale
    """
    ythr = min_y + (1.0 + eps_y) / scale
    x_min = min_x_above_y(points_rot, ythr)
    if x_min == float("inf"):
        return float("inf")
    return 1.0 + scale * (x_min - max_x)


def min_clearance(points, precomp, thetas, scale, eps_y, local_k=3, local_iters=22):
    """
    Compute minimum clearance across theta in [0,pi/2].

    We:
      - scan a dense grid of angles
      - refine a few best candidates by golden-section search
    """
    vals = []
    best = float("inf")
    for pts, miny, maxx in precomp:
        cl = clearance_for_theta(pts, miny, maxx, scale, eps_y)
        vals.append(cl)
        if cl < best:
            best = cl

    idx_sorted = sorted(range(len(thetas)), key=lambda i: vals[i])[:local_k]

    phi = (math.sqrt(5.0) - 1.0) / 2.0

    def f(th):
        pts = rotate_points(points, th)
        miny = min(y for (_, y) in pts)
        maxx = max(x for (x, _) in pts)
        return clearance_for_theta(pts, miny, maxx, scale, eps_y)

    for idx in idx_sorted:
        a = thetas[max(0, idx - 1)]
        b = thetas[min(len(thetas) - 1, idx + 1)]
        if b - a <= 1e-15:
            continue

        c = b - (b - a) * phi
        d = a + (b - a) * phi
        fc = f(c)
        fd = f(d)

        for _ in range(local_iters):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) * phi
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) * phi
                fd = f(d)

        best = min(best, fc, fd)

    return best


def max_scale(points, n_theta, bisection_iters, eps_y):
    """
    Find maximum scale s such that the polygon can pass the corner.
    Feasibility is monotone in s => bisection.
    """
    thetas = [(math.pi / 2.0) * i / (n_theta - 1) for i in range(n_theta)]
    precomp = precompute_rotations(points, thetas)

    def feasible(s):
        # positive clearance means safe; slightly negative treated as numerical slack
        mc = min_clearance(points, precomp, thetas, s, eps_y)
        return mc >= -1e-13

    lo, hi = 0.0, 2.0
    while feasible(hi):
        hi *= 1.3
        if hi > 50.0:
            break

    for _ in range(bisection_iters):
        mid = (lo + hi) * 0.5
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------
# Outer optimization
# --------------------------


def objective(r, mode):
    pts = build_unit_pentagon(r)
    if pts is None:
        return -1.0, 0.0

    if mode == "coarse":
        s = max_scale(pts, n_theta=450, bisection_iters=35, eps_y=1e-12)
    elif mode == "mid":
        s = max_scale(pts, n_theta=1400, bisection_iters=55, eps_y=1e-14)
    elif mode == "fine":
        s = max_scale(pts, n_theta=8000, bisection_iters=75, eps_y=1e-15)
    else:
        raise ValueError("bad mode")

    return base_area(r) * s * s, s


def golden_max(f, a, b, iters):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - (b - a) * phi
    d = a + (b - a) * phi
    fc = f(c)
    fd = f(d)

    for _ in range(iters):
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * phi
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * phi
            fd = f(d)

    if fc > fd:
        return c, fc
    else:
        return d, fd


def solve():
    # Assert test value from statement: square would yield 1.0000000000
    assert f"{1.0:.10f}" == "1.0000000000"

    # Sanity: build a unit pentagon and verify edge lengths are all 1 (within tolerance)
    test_pts = build_unit_pentagon(0.9)
    assert test_pts is not None
    for i in range(5):
        x1, y1 = test_pts[i]
        x2, y2 = test_pts[(i + 1) % 5]
        d = math.hypot(x2 - x1, y2 - y1)
        assert abs(d - 1.0) < 1e-9

    # 1) coarse scan to locate region of maximum
    rmin, rmax = 0.75, 1.05
    steps = 240
    best_r = None
    best_val = -1.0

    for i in range(steps + 1):
        r = rmin + (rmax - rmin) * i / steps
        val, _ = objective(r, "coarse")
        if val > best_val:
            best_val = val
            best_r = r

    # 2) golden refine around best_r using mid precision
    cache = {}

    def f_mid(rr):
        key = round(rr, 15)
        if key in cache:
            return cache[key]
        val, _ = objective(rr, "mid")
        cache[key] = val
        return val

    a = best_r - 0.02
    b = best_r + 0.02
    r1, _ = golden_max(f_mid, a, b, iters=26)

    # 3) narrow refine again
    a2 = r1 - 0.002
    b2 = r1 + 0.002
    r2, _ = golden_max(f_mid, a2, b2, iters=35)

    # 4) final high precision evaluation once
    final_area, _ = objective(r2, "fine")
    print(f"{final_area:.10f}")


if __name__ == "__main__":
    solve()
