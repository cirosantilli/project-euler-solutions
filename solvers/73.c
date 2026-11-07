#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit, int64_t ln, int64_t ld, int64_t un, int64_t ud) {
    int64_t total = 0;
    for (int64_t d = 2; d <= limit; ++d) {
        int64_t first = euler_max(1, ln * d / ld + 1),
                last = euler_min(d - 1, (un * d - 1) / ud);
        for (int64_t n = first; n <= last; ++n)
            if (euler_gcd(n, d) == 1)
                ++total;
    }
    return total;
}

int main(void) {
    assert(solve(8, 1, 3, 1, 2) == 3);
    printf("%" PRId64 "\n", solve(12000, 1, 3, 1, 2));
    return 0;
}
