#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t n) {
    if (n <= 0)
        return 0;
    int64_t total = n * n;
    for (int64_t x = 0; x <= n; ++x)
        for (int64_t y = 0; y <= n; ++y) {
            if (!x && !y)
                continue;
            if (!x || !y)
                total += n;
            else {
                int64_t gcd = euler_gcd(x, y), dx = y / gcd, dy = x / gcd;
                total +=
                    euler_min(x / dx, (n - y) / dy) + euler_min((n - x) / dx, y / dy);
            }
        }
    return total;
}

int main(void) {
    assert(solve(2) == 14);
    printf("%" PRId64 "\n", solve(50));
    return 0;
}
