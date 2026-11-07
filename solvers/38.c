#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

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
    if (width < 2 || width > 9)
        return 0;
    int64_t best = 0;
    for (int64_t x = 1; x < euler_pow(10, width / 2); ++x) {
        int64_t value = 0, n = 0;
        do {
            ++n;
            value = euler_concat(value, x * n);
        } while (euler_digits(value) < width);
        if (n > 1 && pandigital(value, width))
            best = euler_max(best, value);
    }
    return best;
}

int main(void) {

    printf("%" PRId64 "\n", solve(9));
    return 0;
}
