#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

/* Depth-first traversal of the same right-truncatable prime tree as Python's
 * breadth-first frontier. The traversal order does not affect the sum. */
static int64_t extend(int64_t prefix, int64_t base) {
    int64_t total = 0;
    for (int64_t digit = 1; digit < base; ++digit) {
        int64_t n = prefix * base + digit;
        if (!euler_prime(n))
            continue;
        bool left = true;
        for (int64_t power = base; power <= n; power *= base)
            if (!euler_prime(n % power)) {
                left = false;
                break;
            }
        if (left)
            total += n;
        total += extend(n, base);
    }
    return total;
}

int64_t solve(int64_t base) {
    assert(base >= 2);
    int64_t total = 0;
    for (int64_t p = 2; p < base; ++p)
        if (euler_prime(p))
            total += extend(p, base);
    return total;
}

int main(void) {

    printf("%" PRId64 "\n", solve(10));
    return 0;
}
