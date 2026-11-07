#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t *perimeter_counts(int64_t limit) {
    int64_t *counts = euler_alloc((size_t)limit + 1, sizeof(*counts));
    for (int64_t m = 2; 2 * m * (m + 1) <= limit; ++m)
        for (int64_t n = 1; n < m; ++n) {
            if ((m - n) % 2 == 0 || euler_gcd(m, n) != 1)
                continue;
            int64_t p = 2 * m * (m + n);
            for (int64_t k = p; k <= limit; k += p)
                ++counts[k];
        }
    return counts;
}

int64_t solve(int64_t limit) {
    if (limit < 0)
        return 0;
    int64_t *counts = perimeter_counts(limit), total = 0;
    for (int64_t p = 0; p <= limit; ++p)
        if (counts[p] == 1)
            ++total;
    free(counts);
    return total;
}

int main(void) {
    assert(solve(50) == 6);
    printf("%" PRId64 "\n", solve(1500000));
    return 0;
}
