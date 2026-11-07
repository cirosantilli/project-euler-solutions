#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool fast_prime(int64_t n) {
    const int64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2)
        return false;
    for (size_t i = 0; i < 12; ++i)
        if (n % small[i] == 0)
            return n == small[i];
    int64_t d = n - 1, s = 0;
    while (d % 2 == 0) {
        d /= 2;
        ++s;
    }
    const int64_t bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (size_t i = 0; i < 7; ++i) {
        if (bases[i] % n == 0)
            continue;
        int64_t x = euler_powmod(bases[i], d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool pass = false;
        for (int64_t r = 1; r < s; ++r) {
            x = euler_mulmod(x, x, n);
            if (x == n - 1) {
                pass = true;
                break;
            }
        }
        if (!pass)
            return false;
    }
    return true;
}

int64_t solve(void) {
    int64_t primes = 0;
    for (int64_t side = 3;; side += 2) {
        for (int64_t corner = 1; corner <= 3; ++corner)
            if (fast_prime(side * side - corner * (side - 1)))
                ++primes;
        if (10 * primes < 2 * side - 1)
            return side;
    }
}

int main(void) {
    printf("%" PRId64 "\n", solve());
    return 0;
}
