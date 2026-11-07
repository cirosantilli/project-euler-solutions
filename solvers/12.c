#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t k) {
    for (int64_t n = 1;; ++n) {
        int64_t a = n % 2 ? n : n / 2, b = n % 2 ? (n + 1) / 2 : n + 1;
        if (euler_divisor_count(a) * euler_divisor_count(b) > k)
            return n * (n + 1) / 2;
    }
}

int main(void) {
    assert(solve(5) == 28);
    printf("%" PRId64 "\n", solve(500));
    return 0;
}
