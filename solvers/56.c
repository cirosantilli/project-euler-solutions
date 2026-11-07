#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"
#include "../data/project-euler-statements/data/c/digits.h"

int64_t solve(int64_t limit) {
    int64_t best = 0;
    for (int64_t a = 0; a < limit; ++a)
        for (int64_t b = 0; b < limit; ++b) {
            EulerDigits power = digits_power(a, b);
            best = euler_max(best, digits_sum(&power));
        }
    return best;
}

int main(void) {

    printf("%" PRId64 "\n", solve(100));
    return 0;
}
