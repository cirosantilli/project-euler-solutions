#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t period_length(int64_t n) {
    int64_t a0 = euler_isqrt(n);
    if (a0 * a0 == n)
        return 0;
    int64_t m = 0, d = 1, a = a0, period = 0;
    do {
        m = d * a - m;
        d = (n - m * m) / d;
        a = (a0 + m) / d;
        ++period;
    } while (a != 2 * a0);
    return period;
}

int64_t solve(int64_t limit) {
    int64_t count = 0;
    for (int64_t n = 2; n <= limit; ++n)
        if (period_length(n) % 2)
            ++count;
    return count;
}

int main(void) {
    assert(solve(13) == 4);
    printf("%" PRId64 "\n", solve(10000));
    return 0;
}
