#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool pandigital(int64_t value, int64_t width) {
    if (euler_digits(value) != width)
        return false;
    uint64_t mask = 0;
    while (value) {
        int64_t d = value % 10;
        value /= 10;
        if (d < 1 || d > width || (mask & (UINT64_C(1) << d)))
            return false;
        mask |= UINT64_C(1) << d;
    }
    return true;
}

int64_t solve(int64_t width) {
    if (width < 3 || width > 9)
        return 0;
    int64_t lower = euler_pow(10, width / 2 - 1), upper = lower * 10 - 1, total = 0;
    bool *seen = euler_alloc((size_t)upper + 1, sizeof(*seen));
    for (int64_t a = 1; a <= upper / a; ++a)
        for (int64_t b = euler_max(a, (lower + a - 1) / a); b <= upper / a; ++b) {
            int64_t p = a * b;
            if (!seen[p] && pandigital(euler_concat(euler_concat(a, b), p), width)) {
                total += p;
                seen[p] = true;
            }
        }
    free(seen);
    return total;
}

int main(void) {

    printf("%" PRId64 "\n", solve(9));
    return 0;
}
