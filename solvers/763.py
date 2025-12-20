# Project Euler 763
# Amoebas in a 3D Grid (monster splitting)
#
# We compute D(n) modulo 10^9 for n = 10000 (last nine digits),
# using the recurrence framework derived from a state compression
# on the boundary "red path" encoding described in the problem's
# community analysis.
#
# No external libraries are used (only Python stdlib 'array').

from array import array

MOD = 1_000_000_000


def compute_a2_exact_small(M):
    """
    Exact (non-mod) computation of a2[0..M] for small M only.
    This is used for asserting the exact test value D(20)=9204559704.

    a2[m] corresponds to D(m+1).
    """
    # Determine maximum n needed: offset[n] = (n+1)(n+2)/2 <= M
    n_tmp = 0
    while (n_tmp + 1) * (n_tmp + 2) // 2 <= M:
        n_tmp += 1
    max_n = n_tmp - 1
    N = max_n + 3

    offset = [0] * (N + 2)
    lens = [0] * (N + 2)
    for n in range(N + 2):
        off = (n + 1) * (n + 2) // 2
        offset[n] = off
        ln = M - off + 1
        lens[n] = ln if ln > 0 else 0

    u = [[] for _ in range(N + 2)]
    v = [[] for _ in range(N + 2)]
    for n in range(1, N + 2):
        ln = lens[n]
        if ln > 0:
            u[n] = [0] * (n * ln)
            v[n] = [0] * (n * ln)
        else:
            u[n] = []
            v[n] = []

    f0 = [0] * (M + 1)
    a2 = [0] * (M + 1)
    a2[0] = 1

    n_active = 0
    for m in range(M + 1):
        while n_active + 1 < N + 1 and offset[n_active + 1] <= m:
            n_active += 1

        for n in range(1, n_active + 1):
            off = offset[n]
            ln = lens[n]
            idx_cur = m - off

            mp1 = m - n - 2
            idx1 = mp1 - off

            mp2 = m - n - 3
            idx2 = mp2 - offset[n + 1]
            lnp = lens[n + 1]

            mp3 = m - n - 1
            idx3 = mp3 - offset[n - 1]
            lnm = lens[n - 1]

            if n == 1:
                # Only k=1 exists.
                val_u = 0
                if idx1 >= 0:
                    val_u += 2 * u[1][idx1] + v[1][idx1]
                if idx2 >= 0 and lnp > 0:
                    val_u += v[2][idx2] + u[2][lnp + idx2]
                if mp3 >= 0:
                    val_u += f0[mp3]
                u[1][idx_cur] = val_u

                val_v = 0
                if idx1 >= 0:
                    val_v += 2 * v[1][idx1] + 2 * u[1][idx1]
                if idx2 >= 0 and lnp > 0:
                    val_v += v[2][lnp + idx2] + 2 * u[2][idx2]
                if mp3 >= 0:
                    val_v += f0[mp3]
                v[1][idx_cur] = val_v
                continue

            u_n = u[n]
            v_n = v[n]
            u_p = u[n + 1]
            v_p = v[n + 1]
            u_m = u[n - 1]
            v_m = v[n - 1]

            u_n1 = u_n[idx1] if idx1 >= 0 else 0
            v_n1 = v_n[idx1] if idx1 >= 0 else 0
            u_p1 = u_p[idx2] if (idx2 >= 0 and lnp > 0) else 0
            v_p1 = v_p[idx2] if (idx2 >= 0 and lnp > 0) else 0

            base = 0
            base_next = ln
            base_p = lnp
            base_m = 0

            if idx1 < 0:
                # Only the n-1 term survives.
                for _k in range(1, n):
                    u_n[base + idx_cur] = u_m[base_m + idx3]
                    v_n[base + idx_cur] = v_m[base_m + idx3]
                    base = base_next
                    base_next += ln
                    base_m += lnm
                # k=n
                u_n[(n - 1) * ln + idx_cur] = u_m[(n - 2) * lnm + idx3]
                v_n[(n - 1) * ln + idx_cur] = v_m[(n - 2) * lnm + idx3]
                continue

            if idx2 >= 0 and lnp > 0:
                # Full (fast) recurrence.
                for _k in range(1, n):
                    u_n[base + idx_cur] = (
                        u_n[base + idx1]
                        + v_p1
                        + u_p[base_p + idx2]
                        + u_m[base_m + idx3]
                        + v_n1
                        + u_n[base_next + idx1]
                    )
                    v_n[base + idx_cur] = (
                        v_n[base + idx1]
                        + v_p[base_p + idx2]
                        + u_p1
                        + v_m[base_m + idx3]
                        + v_n[base_next + idx1]
                        + u_n1
                    )
                    base = base_next
                    base_next += ln
                    base_p += lnp
                    base_m += lnm
                base_last = (n - 1) * ln
                u_n[base_last + idx_cur] = (
                    2 * u_n[base_last + idx1]
                    + v_n1
                    + v_p1
                    + u_p[base_p + idx2]
                    + u_m[(n - 2) * lnm + idx3]
                )
                v_n[base_last + idx_cur] = (
                    2 * v_n[base_last + idx1]
                    + 2 * u_n1
                    + v_p[base_p + idx2]
                    + 2 * u_p1
                    + v_m[(n - 2) * lnm + idx3]
                )
            else:
                # No n+1 contribution.
                for _k in range(1, n):
                    u_n[base + idx_cur] = (
                        u_n[base + idx1]
                        + u_m[base_m + idx3]
                        + v_n1
                        + u_n[base_next + idx1]
                    )
                    v_n[base + idx_cur] = (
                        v_n[base + idx1]
                        + v_m[base_m + idx3]
                        + v_n[base_next + idx1]
                        + u_n1
                    )
                    base = base_next
                    base_next += ln
                    base_m += lnm
                base_last = (n - 1) * ln
                u_n[base_last + idx_cur] = (
                    2 * u_n[base_last + idx1] + v_n1 + u_m[(n - 2) * lnm + idx3]
                )
                v_n[base_last + idx_cur] = (
                    2 * v_n[base_last + idx1] + 2 * u_n1 + v_m[(n - 2) * lnm + idx3]
                )

        # f0 and a2
        val_f = 0
        if m - 1 >= 0:
            val_f += a2[m - 1]
        if m - 2 >= 0:
            val_f += 4 * f0[m - 2]
        mp = m - 3
        if mp >= offset[1] and lens[1] > 0:
            id1 = mp - offset[1]
            val_f += 2 * u[1][id1] + v[1][id1]
        f0[m] = val_f

        if m >= 1:
            val_a = 3 * a2[m - 1]
            if m - 2 >= 0:
                val_a += 3 * f0[m - 2]
            a2[m] = val_a

    return a2


def compute_a2_mod_fast(M, MOD=MOD):
    """
    Modular computation of a2[0..M] (mod 1e9).
    a2[m] corresponds to D(m+1) mod MOD.
    """
    n_tmp = 0
    while (n_tmp + 1) * (n_tmp + 2) // 2 <= M:
        n_tmp += 1
    max_n = n_tmp - 1
    N = max_n + 3

    offset = [0] * (N + 2)
    lens = [0] * (N + 2)
    for n in range(N + 2):
        off = (n + 1) * (n + 2) // 2
        offset[n] = off
        ln = M - off + 1
        lens[n] = ln if ln > 0 else 0

    u = [array("I") for _ in range(N + 2)]
    v = [array("I") for _ in range(N + 2)]
    for n in range(1, N + 2):
        ln = lens[n]
        if ln > 0:
            u[n] = array("I", [0]) * (n * ln)
            v[n] = array("I", [0]) * (n * ln)
        else:
            u[n] = array("I")
            v[n] = array("I")

    f0 = [0] * (M + 1)
    a2 = [0] * (M + 1)
    a2[0] = 1

    n_active = 0
    for m in range(M + 1):
        while n_active + 1 < N + 1 and offset[n_active + 1] <= m:
            n_active += 1

        for n in range(1, n_active + 1):
            off = offset[n]
            ln = lens[n]
            idx_cur = m - off

            mp1 = m - n - 2
            idx1 = mp1 - off

            mp2 = m - n - 3
            idx2 = mp2 - offset[n + 1]
            lnp = lens[n + 1]

            mp3 = m - n - 1
            idx3 = mp3 - offset[n - 1]
            lnm = lens[n - 1]

            if n == 1:
                val_u = 0
                if idx1 >= 0:
                    val_u += 2 * u[1][idx1] + v[1][idx1]
                if idx2 >= 0 and lnp > 0:
                    val_u += v[2][idx2] + u[2][lnp + idx2]
                if mp3 >= 0:
                    val_u += f0[mp3]
                u[1][idx_cur] = val_u % MOD

                val_v = 0
                if idx1 >= 0:
                    val_v += 2 * v[1][idx1] + 2 * u[1][idx1]
                if idx2 >= 0 and lnp > 0:
                    val_v += v[2][lnp + idx2] + 2 * u[2][idx2]
                if mp3 >= 0:
                    val_v += f0[mp3]
                v[1][idx_cur] = val_v % MOD
                continue

            u_n = u[n]
            v_n = v[n]
            u_p = u[n + 1]
            v_p = v[n + 1]
            u_m = u[n - 1]
            v_m = v[n - 1]

            u_n1 = u_n[idx1] if idx1 >= 0 else 0
            v_n1 = v_n[idx1] if idx1 >= 0 else 0
            u_p1 = u_p[idx2] if (idx2 >= 0 and lnp > 0) else 0
            v_p1 = v_p[idx2] if (idx2 >= 0 and lnp > 0) else 0

            base = 0
            base_next = ln
            base_p = lnp
            base_m = 0

            if idx1 < 0:
                # Only n-1 term survives.
                for _k in range(1, n):
                    u_n[base + idx_cur] = u_m[base_m + idx3]
                    v_n[base + idx_cur] = v_m[base_m + idx3]
                    base = base_next
                    base_next += ln
                    base_m += lnm
                u_n[(n - 1) * ln + idx_cur] = u_m[(n - 2) * lnm + idx3]
                v_n[(n - 1) * ln + idx_cur] = v_m[(n - 2) * lnm + idx3]
                continue

            if idx2 >= 0 and lnp > 0:
                # Full recurrence.
                for _k in range(1, n):
                    u_n[base + idx_cur] = (
                        u_n[base + idx1]
                        + v_p1
                        + u_p[base_p + idx2]
                        + u_m[base_m + idx3]
                        + v_n1
                        + u_n[base_next + idx1]
                    ) % MOD
                    v_n[base + idx_cur] = (
                        v_n[base + idx1]
                        + v_p[base_p + idx2]
                        + u_p1
                        + v_m[base_m + idx3]
                        + v_n[base_next + idx1]
                        + u_n1
                    ) % MOD
                    base = base_next
                    base_next += ln
                    base_p += lnp
                    base_m += lnm

                base_last = (n - 1) * ln
                u_n[base_last + idx_cur] = (
                    2 * u_n[base_last + idx1]
                    + v_n1
                    + v_p1
                    + u_p[base_p + idx2]
                    + u_m[(n - 2) * lnm + idx3]
                ) % MOD
                v_n[base_last + idx_cur] = (
                    2 * v_n[base_last + idx1]
                    + 2 * u_n1
                    + v_p[base_p + idx2]
                    + 2 * u_p1
                    + v_m[(n - 2) * lnm + idx3]
                ) % MOD
            else:
                # No n+1 term.
                for _k in range(1, n):
                    u_n[base + idx_cur] = (
                        u_n[base + idx1]
                        + u_m[base_m + idx3]
                        + v_n1
                        + u_n[base_next + idx1]
                    ) % MOD
                    v_n[base + idx_cur] = (
                        v_n[base + idx1]
                        + v_m[base_m + idx3]
                        + v_n[base_next + idx1]
                        + u_n1
                    ) % MOD
                    base = base_next
                    base_next += ln
                    base_m += lnm

                base_last = (n - 1) * ln
                u_n[base_last + idx_cur] = (
                    2 * u_n[base_last + idx1] + v_n1 + u_m[(n - 2) * lnm + idx3]
                ) % MOD
                v_n[base_last + idx_cur] = (
                    2 * v_n[base_last + idx1] + 2 * u_n1 + v_m[(n - 2) * lnm + idx3]
                ) % MOD

        # f0 and a2
        val_f = 0
        if m - 1 >= 0:
            val_f += a2[m - 1]
        if m - 2 >= 0:
            val_f += 4 * f0[m - 2]
        mp = m - 3
        if mp >= offset[1] and lens[1] > 0:
            id1 = mp - offset[1]
            val_f += 2 * u[1][id1] + v[1][id1]
        f0[m] = val_f % MOD

        if m >= 1:
            val_a = 3 * a2[m - 1]
            if m - 2 >= 0:
                val_a += 3 * f0[m - 2]
            a2[m] = val_a % MOD

    return a2


def solve():
    # Assert exact test values (from problem statement)
    a2_exact = compute_a2_exact_small(19)  # up to D(20)
    assert a2_exact[1] == 3  # D(2)
    assert a2_exact[9] == 44499  # D(10)
    assert a2_exact[19] == 9204559704  # D(20)

    # Compute modulo for full target (this also covers D(100) test)
    a2_mod = compute_a2_mod_fast(9999, MOD)

    assert a2_mod[1] == 3  # D(2) mod
    assert a2_mod[9] == 44499  # D(10) mod
    assert a2_mod[99] == 780166455  # last nine digits of D(100)

    ans = a2_mod[9999]  # D(10000) last nine digits
    print(f"{ans:09d}")


if __name__ == "__main__":
    solve()
