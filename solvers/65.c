#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t n) {
    EulerDigits older = digits_from_int(0), numerator = digits_from_int(1);
    for (int64_t k = 0; k < n; ++k) {
        int64_t coefficient = k == 0 ? 2 : k % 3 == 2 ? 2 * (k / 3 + 1) : 1;
        EulerDigits previous = numerator;
        digits_mul_small(&numerator, coefficient);
        digits_add(&numerator, &older);
        older = previous;
    }
    return digits_sum(&numerator);
}

int main(void) {
    assert(solve(10) == 17);
    printf("%" PRId64 "\n", solve(100));
    return 0;
}
