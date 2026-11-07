#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t rectangles(int64_t m, int64_t n) {
    return m * (m + 1) / 2 * (n * (n + 1) / 2);
}

static int64_t distance(int64_t a, int64_t b) {
    return a >= b ? a - b : b - a;
}

int64_t solve(int64_t target) {
    int64_t best = 1, difference = distance(1, target);
    for (int64_t m = 1;; ++m) {
        int64_t vertical = m * (m + 1) / 2;
        if (vertical * vertical > target + difference)
            return best;
        int64_t quotient = euler_max(0, target / vertical),
                below = euler_max(m, (euler_isqrt(1 + 8 * quotient) - 1) / 2);
        for (int64_t n = below; n <= below + 1; ++n) {
            int64_t diff = distance(rectangles(m, n), target);
            if (diff < difference) {
                best = m * n;
                difference = diff;
            }
        }
    }
}

int main(void) {
    assert(solve(18) == 6);
    printf("%" PRId64 "\n", solve(2000000));
    return 0;
}
