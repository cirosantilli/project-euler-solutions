#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t n) {
    if (n < 0)
        return 0;
    int64_t *dp = euler_alloc((size_t)n + 1, sizeof(*dp));
    dp[0] = 1;
    for (int64_t part = 1; part < n; ++part)
        for (int64_t s = part; s <= n; ++s)
            dp[s] += dp[s - part];
    int64_t result = dp[n];
    free(dp);
    return result;
}

int main(void) {
    assert(solve(5) == 6);
    printf("%" PRId64 "\n", solve(100));
    return 0;
}
