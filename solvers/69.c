#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    int64_t product = 1;
    for (int64_t p = 2; p <= limit / product; ++p)
        if (euler_prime(p))
            product *= p;
    return product;
}

int main(void) {
    assert(solve(10) == 6);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
