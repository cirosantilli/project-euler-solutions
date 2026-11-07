#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit < 0)
        return 0;
    int64_t *s = euler_divisor_sieve(limit),
            *abundant = euler_alloc((size_t)limit + 1, sizeof(*abundant));
    bool *can = euler_alloc((size_t)limit + 1, sizeof(*can));
    size_t count = 0;
    for (int64_t n = 12; n <= limit; ++n)
        if (s[n] > n)
            abundant[count++] = n;
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i; j < count && abundant[i] + abundant[j] <= limit; ++j)
            can[abundant[i] + abundant[j]] = true;
    int64_t total = 0;
    for (int64_t n = 1; n <= limit; ++n)
        if (!can[n])
            total += n;
    free(s);
    free(abundant);
    free(can);
    return total;
}

int main(void) {
    assert(solve(24) == 276);
    printf("%" PRId64 "\n", solve(28123));
    return 0;
}
