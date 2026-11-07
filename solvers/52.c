#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t multiplier) {
    assert(multiplier >= 1);
    for (int64_t lower = 1;; lower *= 10)
        for (int64_t x = lower; x <= (lower * 10 - 1) / multiplier; ++x) {
            bool ok = true;
            for (int64_t m = 2; m <= multiplier; ++m)
                if (!euler_same_digits(x, m * x)) {
                    ok = false;
                    break;
                }
            if (ok)
                return x;
        }
}

int main(void) {

    printf("%" PRId64 "\n", solve(6));
    return 0;
}
