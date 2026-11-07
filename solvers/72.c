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
    int64_t *phi = totients(limit), sum = 0;
    for (int64_t d = 2; d <= limit; ++d)
        sum += phi[d];
    free(phi);
    return sum;
}

int main(void) {
    assert(solve(8) == 21);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
