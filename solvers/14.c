#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 0)
        return 0;
    int64_t *cache = euler_alloc((size_t)limit + 1, sizeof(*cache));
    cache[1] = 1;
    int64_t best = 1, longest = 1;
    for (int64_t n = 1; n <= limit; ++n) {
        int64_t m = n, length = 0;
        while (m > limit || !cache[m]) {
            m = m % 2 ? 3 * m + 1 : m / 2;
            ++length;
        }
        length += cache[m];
        cache[n] = length;
        if (length > longest) {
            longest = length;
            best = n;
        }
    }
    free(cache);
    return best;
}

int main(void) {
    assert(solve(13) == 9);
    printf("%" PRId64 "\n", solve(999999));
    return 0;
}
