#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(void) {
    bool *prime = euler_sieve(9999);
    for (int64_t a = 1000; a < 10000; ++a)
        if (prime[a])
            for (int64_t b = a + 1; 2 * b - a < 10000; ++b) {
                int64_t c = 2 * b - a;
                if (a == 1487 && b == 4817)
                    continue;
                if (prime[b] && prime[c] && euler_same_digits(a, b) &&
                    euler_same_digits(a, c)) {
                    free(prime);
                    return a * INT64_C(100000000) + b * 10000 + c;
                }
            }
    free(prime);
    return 0;
}

int main(void) {
    printf("%" PRId64 "\n", solve());
    return 0;
}
