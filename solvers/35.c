#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 2)
        return 0;
    bool *prime = euler_sieve(euler_pow(10, euler_digits(limit - 1)) - 1);
    int64_t count = 0;
    for (int64_t n = 2; n < limit; ++n) {
        int64_t m = n, power = euler_pow(10, euler_digits(n) - 1);
        bool ok = true;
        do {
            if (!(prime[m])) {
                ok = false;
                break;
            }
            m = (m % power) * 10 + m / power;
        } while (m != n);
        if (ok)
            ++count;
    }
    free(prime);
    return count;
}

int main(void) {
    assert(solve(100) == 13);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
