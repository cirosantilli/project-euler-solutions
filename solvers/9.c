#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t total) {
    int64_t best = 0;
    for (int64_t m = 2; 2 * m * (m + 1) <= total; ++m)
        for (int64_t n = 1; n < m; ++n) {
            int64_t s = 2 * m * (m + n);
            if (total % s)
                continue;
            int64_t k = total / s, a = k * (m * m - n * n), b = k * 2 * m * n,
                    c = k * (m * m + n * n);
            best = euler_max(best, a * b * c);
        }
    return best;
}

int main(void) {
    assert(solve(12) == 60);
    assert(solve(210) == 328860);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
