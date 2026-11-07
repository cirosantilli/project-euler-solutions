#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t n) {
    EulerDigits a = digits_from_int(1);
    for (int64_t k = 2; k <= n; ++k)
        digits_mul_small(&a, k);
    return digits_sum(&a);
}

int main(void) {
    assert(solve(10) == 27);
    printf("%" PRId64 "\n", solve(100));
    return 0;
}
