#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t target) {
    for (int64_t n = 2;; ++n) {
        bool *prime = euler_sieve(n);
        int64_t *ways = euler_alloc((size_t)n + 1, sizeof(*ways));
        ways[0] = 1;
        for (int64_t p = 2; p <= n; ++p)
            if (prime[p])
                for (int64_t s = p; s <= n; ++s)
                    ways[s] += ways[s - p];
        int64_t count = ways[n];
        free(ways);
        free(prime);
        if (count > target)
            return n;
    }
}

int main(void) {
    assert(solve(4) == 10);
    printf("%" PRId64 "\n", solve(5000));
    return 0;
}
