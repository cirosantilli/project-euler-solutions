#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t n) {
    assert(n >= 1);
    int64_t limit = 16;
    for (;;) {
        bool *prime = euler_sieve(limit);
        int64_t count = 0;
        for (int64_t i = 2; i <= limit; ++i)
            if (prime[i] && ++count == n) {
                free(prime);
                return i;
            }
        free(prime);
        limit *= 2;
    }
}

int main(void) {
    assert(solve(6) == 13);
    printf("%" PRId64 "\n", solve(10001));
    return 0;
}
