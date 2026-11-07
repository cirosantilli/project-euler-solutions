#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t limit, int64_t threshold) {
    if (threshold < 1)
        return limit > 0 ? limit * (limit + 3) / 2 : 0;
    int64_t total = 0;
    for (int64_t n = 1; n <= limit; ++n) {
        int64_t c = 1;
        for (int64_t r = 1; r <= n / 2; ++r) {
            c = c * (n - r + 1) / r;
            if (c > threshold) {
                total += n - 2 * r + 1;
                break;
            }
        }
    }
    return total;
}

int main(void) {
    assert(solve(23, 1000000) == 4);
    printf("%" PRId64 "\n", solve(100, 1000000));
    return 0;
}
