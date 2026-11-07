#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit, int64_t num, int64_t den) {
    assert(num > 0 && num < den);
    int64_t best_n = 0, best_d = 1;
    for (int64_t d = 1; d <= limit; ++d) {
        int64_t n = (num * d - 1) / den;
        if (n > 0 && euler_gcd(n, d) == 1 && n * best_d > best_n * d) {
            best_n = n;
            best_d = d;
        }
    }
    return best_n;
}

int main(void) {
    assert(solve(8, 3, 7) == 2);
    printf("%" PRId64 "\n", solve(1000000, 3, 7));
    return 0;
}
