#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t n) {
    int64_t lo = n == 0 ? 0 : euler_pow(10, n - 1), hi = euler_pow(10, n) - 1, best = 0;
    for (int64_t a = hi; a >= lo; --a) {
        if (a * hi < best)
            break;
        for (int64_t b = a; b >= lo; --b) {
            int64_t product = a * b;
            if (product <= best)
                break;
            if (euler_palindrome(product, 10)) {
                best = product;
                break;
            }
        }
    }
    return best;
}

int main(void) {
    assert(solve(2) == 9009);
    printf("%" PRId64 "\n", solve(3));
    return 0;
}
