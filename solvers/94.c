#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t limit) {
    int64_t x = 14, y = 8, total = 0;
    for (;;) {
        int64_t p = x % 3 == 2 ? x + 2 : x - 2;
        if (p > limit)
            return total;
        total += p;
        int64_t next_x = 2 * x + 3 * y;
        y = x + 2 * y;
        x = next_x;
    }
}

int main(void) {
    assert(solve(50) == 66);
    printf("%" PRId64 "\n", solve(1000000000));
    return 0;
}
