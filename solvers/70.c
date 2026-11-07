#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t *totients(int64_t limit) {
    int64_t *phi = euler_alloc((size_t)limit + 1, sizeof(*phi)),
            *primes = euler_alloc((size_t)limit + 1, sizeof(*primes));
    size_t count = 0;
    if (limit >= 1)
        phi[1] = 1;
    for (int64_t i = 2; i <= limit; ++i) {
        if (!phi[i]) {
            primes[count++] = i;
            phi[i] = i - 1;
        }
        for (size_t j = 0; j < count && primes[j] <= limit / i; ++j) {
            int64_t p = primes[j];
            if (i % p == 0) {
                phi[i * p] = phi[i] * p;
                break;
            }
            phi[i * p] = phi[i] * (p - 1);
        }
    }
    free(primes);
    return phi;
}

int64_t solve(int64_t limit) {
    if (limit < 2)
        return 0;
    int64_t *table = totients(limit);
    int64_t best = 0, best_phi = 1;
    for (int64_t n = 2; n <= limit; ++n) {
        int64_t pn = table[n];
        if (euler_same_digits(n, pn) && (!best || n * best_phi < best * pn)) {
            best = n;
            best_phi = pn;
        }
    }
    free(table);
    return best;
}

int main(void) {
    assert(solve(100) == 21);
    printf("%" PRId64 "\n", solve(9999999));
    return 0;
}
