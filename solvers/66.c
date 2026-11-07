#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"
#include "../data/project-euler-statements/data/c/digits.h"

static EulerDigits minimal_x(int64_t D) {
    int64_t a0 = euler_isqrt(D), m = 0, d = 1, a = a0;
    EulerDigits h2 = digits_from_int(0), h1 = digits_from_int(1), k2 = h1, k1 = h2;
    for (;;) {
        EulerDigits h = h1, k = k1;
        digits_mul_small(&h, a);
        digits_add(&h, &h2);
        digits_mul_small(&k, a);
        digits_add(&k, &k2);
        EulerDigits left = digits_multiply(&h, &h), right = digits_multiply(&k, &k),
                    one = digits_from_int(1);
        digits_mul_small(&right, D);
        digits_add(&right, &one);
        if (digits_compare(&left, &right) == 0)
            return h;
        h2 = h1;
        h1 = h;
        k2 = k1;
        k1 = k;
        m = d * a - m;
        d = (D - m * m) / d;
        a = (a0 + m) / d;
    }
}

int64_t solve(int64_t limit) {
    int64_t best = 0;
    EulerDigits largest = digits_from_int(0);
    for (int64_t d = 2; d <= limit; ++d)
        if (!euler_square(d)) {
            EulerDigits x = minimal_x(d);
            if (digits_compare(&x, &largest) > 0) {
                largest = x;
                best = d;
            }
        }
    return best;
}

int main(void) {
    assert(solve(7) == 5);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
