#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t prime_permutations(int64_t n) {
    assert(n >= 1 && n <= 9);
    int64_t digits[9], best = 0;
    for (int64_t i = 0; i < n; ++i)
        digits[i] = i + 1;
    do {
        int64_t value = 0;
        for (int64_t i = 0; i < n; ++i)
            value = value * 10 + digits[i];
        if (euler_prime(value))
            best = euler_max(best, value);
    } while (euler_next_permutation(digits, (size_t)n));
    return best;
}

int64_t solve(int64_t max_digits) {
    for (int64_t n = max_digits; n > 0; --n) {
        if ((n * (n + 1) / 2) % 3 == 0)
            continue;
        int64_t best = prime_permutations(n);
        if (best)
            return best;
    }
    return 0;
}

int main(void) {

    printf("%" PRId64 "\n", solve(9));
    return 0;
}
