#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t n) {
    int64_t result = 1;
    for (int64_t i = 1; i <= n; ++i)
        result = result * (n + i) / i;
    return result;
}

int main(void) {
    assert(solve(2) == 6);
    printf("%" PRId64 "\n", solve(20));
    return 0;
}
