#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t n) {
    assert(n >= 1 && n % 2 == 1);
    int64_t m = (n - 1) / 2;
    return 1 + 16 * m * (m + 1) * (2 * m + 1) / 6 + 4 * m * (m + 1) / 2 + 4 * m;
}

int main(void) {
    assert(solve(5) == 101);
    printf("%" PRId64 "\n", solve(1001));
    return 0;
}
