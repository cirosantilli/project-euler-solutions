#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t n) {
    EulerDigits a = digits_power(2, n);
    return digits_sum(&a);
}

int main(void) {
    assert(solve(15) == 26);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
