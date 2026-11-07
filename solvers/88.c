#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static void factors(int64_t start, int64_t product, int64_t sum, int64_t terms,
                    int64_t k_max, int64_t *best) {
    for (int64_t f = start; f <= 2 * k_max / product; ++f) {
        int64_t next_product = product * f, next_sum = sum + f, next_terms = terms + 1;
        int64_t k = next_terms + next_product - next_sum;
        if (k <= k_max) {
            best[k] = euler_min(best[k], next_product);
            factors(f, next_product, next_sum, next_terms, k_max, best);
        }
    }
}

int64_t solve(int64_t k_max) {
    if (k_max < 2)
        return 0;
    int64_t *best = euler_alloc((size_t)k_max + 1, sizeof(*best)), total = 0;
    bool *seen = euler_alloc((size_t)(2 * k_max + 1), sizeof(*seen));
    for (int64_t k = 0; k <= k_max; ++k)
        best[k] = 2 * k_max + 1;
    factors(2, 1, 0, 0, k_max, best);
    for (int64_t k = 2; k <= k_max; ++k)
        if (!seen[best[k]]) {
            total += best[k];
            seen[best[k]] = true;
        }
    free(best);
    free(seen);
    return total;
}

int main(void) {
    assert(solve(6) == 30);
    assert(solve(12) == 61);
    printf("%" PRId64 "\n", solve(12000));
    return 0;
}
