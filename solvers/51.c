#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool has_family(int64_t p, int64_t target, const bool *prime) {
    int64_t length = euler_digits(p), digits[19], powers[19];
    for (int64_t i = 0, m = p, power = 1; i < length; ++i, m /= 10, power *= 10) {
        digits[i] = m % 10;
        powers[i] = power;
    }
    for (uint64_t mask = 1; mask < (UINT64_C(1) << length); ++mask) {
        int64_t digit = -1, shift = 0, positions = 0;
        bool same = true;
        for (int64_t i = 0; i < length; ++i)
            if (mask & (UINT64_C(1) << i)) {
                if (digit < 0)
                    digit = digits[i];
                if (digit != digits[i])
                    same = false;
                shift += powers[i];
                ++positions;
            }
        if (!same)
            continue;
        if (target >= 8 && ((mask & 1) || positions % 3))
            continue;
        int64_t count = 0;
        for (int64_t d = 0; d < 10; ++d) {
            if (d == 0 && (mask & (UINT64_C(1) << (length - 1))))
                continue;
            int64_t value = p + (d - digit) * shift;
            if (prime ? prime[value] : euler_prime(value))
                ++count;
        }
        if (count >= target)
            return true;
    }
    return false;
}

int64_t solve(int64_t target) {
    for (int64_t lower = 1;; lower *= 10) {
        int64_t upper = lower * 10 - 1;
        bool *prime = euler_sieve(upper);
        for (int64_t p = euler_max(2, lower); p <= upper; ++p)
            if (prime[p] && has_family(p, target, prime)) {
                free(prime);
                return p;
            }
        free(prime);
    }
}

int main(void) {

    printf("%" PRId64 "\n", solve(8));
    return 0;
}
