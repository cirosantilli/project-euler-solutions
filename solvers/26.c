#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int64_t cycle_length(int64_t d) {
    while (d % 2 == 0)
        d /= 2;
    while (d % 5 == 0)
        d /= 5;
    if (d == 1)
        return 0;
    int64_t r = 10 % d, k = 1;
    while (r != 1) {
        r = r * 10 % d;
        ++k;
    }
    return k;
}

int64_t solve(int64_t limit) {
    int64_t best = 0, longest = -1;
    for (int64_t d = 1; d < limit; ++d) {
        int64_t k = cycle_length(d);
        if (k > longest) {
            longest = k;
            best = d;
        }
    }
    return best;
}

int main(void) {
    assert(solve(10) == 7);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
