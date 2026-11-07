#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"
#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t limit, int64_t digits) {
    assert(digits >= 0);
    int64_t total = 0;
    for (int64_t n = 1; n <= limit; ++n) {
        int64_t root = euler_isqrt(n);
        if (root * root == n)
            continue;
        int64_t width = euler_digits(root);
        if (digits < width) {
            total += euler_digit_sum(root / euler_pow(10, width - digits));
            continue;
        }
        EulerDigits scaled = digits_from_int(n);
        for (int64_t i = 0; i < 2 * (digits - width); ++i)
            digits_mul_small(&scaled, 10);
        EulerDigits prefix = digits_sqrt(&scaled);
        total += digits_sum(&prefix);
    }
    return total;
}

int main(void) {
    assert(solve(2, 100) == 475);
    printf("%" PRId64 "\n", solve(100, 100));
    return 0;
}
