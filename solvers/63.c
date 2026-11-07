#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t base) {
    assert(base >= 2);
    int64_t total = 0;
    for (int64_t a = 1; a < base; ++a) {
        EulerDigits power = digits_from_int(a), lower = digits_from_int(1);
        while (digits_compare(&power, &lower) >= 0) {
            ++total;
            digits_mul_small(&power, a);
            digits_mul_small(&lower, base);
        }
    }
    return total;
}

int main(void) {

    printf("%" PRId64 "\n", solve(10));
    return 0;
}
