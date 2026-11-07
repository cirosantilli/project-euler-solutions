#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t mod) {
    assert(mod > 0);
    size_t capacity = 128;
    int64_t *p = euler_alloc(capacity, sizeof(*p));
    p[0] = 1;
    for (int64_t n = 1;; ++n) {
        if ((size_t)n == capacity) {
            capacity *= 2;
            int64_t *next = euler_alloc(capacity, sizeof(*next));
            memcpy(next, p, (size_t)n * sizeof(*p));
            free(p);
            p = next;
        }
        int64_t total = 0;
        for (int64_t k = 1;; ++k) {
            int64_t g1 = k * (3 * k - 1) / 2, g2 = k * (3 * k + 1) / 2,
                    sign = k % 2 ? 1 : -1;
            if (g1 > n)
                break;
            total = (total + sign * p[n - g1]) % mod;
            if (g2 <= n)
                total = (total + sign * p[n - g2]) % mod;
        }
        p[n] = (total + mod) % mod;
        if (!p[n]) {
            free(p);
            return n;
        }
    }
}

int main(void) {
    assert(solve(5) == 4);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
