#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t digits) {
    assert(digits >= 1 && digits <= EULER_DIGIT_CAPACITY);
    if (digits == 1)
        return 1;
    EulerDigits a = digits_from_int(1), b = a;
    int64_t n = 2;
    while ((int64_t)b.length < digits) {
        EulerDigits next = a;
        digits_add(&next, &b);
        a = b;
        b = next;
        ++n;
    }
    return n;
}

int main(void) {
    assert(solve(3) == 12);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
