#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t pentagonal(int64_t n) {
    return n * (3 * n - 1) / 2;
}

static bool is_pentagonal(int64_t x) {
    if (x <= 0)
        return false;
    int64_t s = euler_isqrt(24 * x + 1);
    return s * s == 24 * x + 1 && (s + 1) % 6 == 0;
}

/* INT64_MAX represents the Python 10**30 no-result sentinel. */
int64_t solve(int64_t limit) {
    int64_t best = INT64_MAX;
    for (int64_t k = 2; k <= limit; ++k)
        for (int64_t j = k - 1; j >= 1; --j) {
            int64_t difference = pentagonal(k) - pentagonal(j);
            if (difference >= best)
                break;
            if (is_pentagonal(difference) &&
                is_pentagonal(pentagonal(k) + pentagonal(j)))
                best = difference;
        }
    return best;
}

int main(void) {

    printf("%" PRId64 "\n", solve(10000));
    return 0;
}
