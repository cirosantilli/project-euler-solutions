#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t limit) {
    EulerDigits p = digits_from_int(3), q = digits_from_int(2);
    int64_t count = 0;
    for (int64_t i = 0; i < limit; ++i) {
        if (p.length > q.length)
            ++count;
        EulerDigits next_p = q, next_q = p;
        digits_mul_small(&next_p, 2);
        digits_add(&next_p, &p);
        digits_add(&next_q, &q);
        p = next_p;
        q = next_q;
    }
    return count;
}

int main(void) {
    assert(solve(8) == 1);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
