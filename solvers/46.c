#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t after) {
    int64_t capacity = 16;
    bool *prime = euler_sieve(capacity);
    for (int64_t n = 9;; n += 2) {
        if (n > capacity) {
            free(prime);
            capacity *= 2;
            prime = euler_sieve(capacity);
        }
        if (n <= after || prime[n])
            continue;
        bool found = false;
        for (int64_t p = 2; p < n; ++p)
            if (prime[p] && (n - p) % 2 == 0 && euler_square((n - p) / 2)) {
                found = true;
                break;
            }
        if (!found) {
            free(prime);
            return n;
        }
    }
}

int main(void) {

    printf("%" PRId64 "\n", solve(0));
    return 0;
}
