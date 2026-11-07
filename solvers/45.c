#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool is_pentagonal(int64_t x) {
    if (x <= 0)
        return false;
    int64_t s = euler_isqrt(24 * x + 1);
    return s * s == 24 * x + 1 && (s + 1) % 6 == 0;
}

int64_t solve(int64_t after) {
    for (int64_t n = 1;; ++n) {
        int64_t value = n * (2 * n - 1);
        if (value > after && is_pentagonal(value))
            return value;
    }
}

int main(void) {
    assert(solve(1) == 40755);
    printf("%" PRId64 "\n", solve(40755));
    return 0;
}
