#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t n) {
    int64_t acc = 1;
    for (int64_t x = 2; x <= n; ++x)
        acc = acc / euler_gcd(acc, x) * x;
    return acc;
}

int main(void) {
    assert(solve(10) == 2520);
    printf("%" PRId64 "\n", solve(20));
    return 0;
}
