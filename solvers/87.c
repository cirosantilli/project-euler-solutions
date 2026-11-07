#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 0)
        return 0;
    int64_t root = euler_isqrt(limit), count = 0;
    bool *prime = euler_sieve(root), *seen = euler_alloc((size_t)limit, sizeof(*seen));
    for (int64_t r = 2; r * r * r * r < limit; ++r)
        if (prime[r])
            for (int64_t q = 2; r * r * r * r + q * q * q < limit; ++q)
                if (prime[q])
                    for (int64_t p = 2; r * r * r * r + q * q * q + p * p < limit; ++p)
                        if (prime[p]) {
                            int64_t value = r * r * r * r + q * q * q + p * p;
                            if (!seen[value]) {
                                seen[value] = true;
                                ++count;
                            }
                        }
    free(prime);
    free(seen);
    return count;
}

int main(void) {
    assert(solve(50) == 4);
    printf("%" PRId64 "\n", solve(50000000));
    return 0;
}
