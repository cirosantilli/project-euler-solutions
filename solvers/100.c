#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t limit) {
    int64_t u = 7, v = 5;
    while ((u + 1) / 2 <= limit) {
        int64_t next_u = 3 * u + 4 * v;
        v = 2 * u + 3 * v;
        u = next_u;
    }
    return (v + 1) / 2;
}

int main(void) {
    assert(solve(3) == 3);
    assert(solve(4) == 15);
    printf("%" PRId64 "\n", solve(INT64_C(1000000000000)));
    return 0;
}
