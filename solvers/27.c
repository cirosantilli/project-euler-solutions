#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(void) {
    bool *prime = euler_sieve(1000);
    int64_t best = 0, longest = -1;
    for (int64_t b = 2; b <= 1000; ++b)
        if (prime[b])
            for (int64_t a = -999; a <= 999; ++a) {
                if ((b != 2 && a % 2 == 0) || (b == 2 && a % 2 != 0) ||
                    !euler_prime(1 + a + b))
                    continue;
                int64_t n = 0;
                while (euler_prime(n * n + a * n + b))
                    ++n;
                if (n > longest) {
                    longest = n;
                    best = a * b;
                }
            }
    free(prime);
    return best;
}

int main(void) {
    printf("%" PRId64 "\n", solve());
    return 0;
}
