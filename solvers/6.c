#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t n) {
    int64_t sum = n * (n + 1) / 2;
    return sum * sum - n * (n + 1) * (2 * n + 1) / 6;
}

int main(void) {
    assert(solve(10) == 2640);
    printf("%" PRId64 "\n", solve(100));
    return 0;
}
