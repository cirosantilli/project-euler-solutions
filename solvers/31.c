#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t target, const int64_t *coins, size_t count) {
    int64_t *dp = euler_alloc((size_t)target + 1, sizeof(*dp));
    dp[0] = 1;
    for (size_t i = 0; i < count; ++i) {
        assert(coins[i] > 0);
        for (int64_t s = coins[i]; s <= target; ++s)
            dp[s] += dp[s - coins[i]];
    }
    int64_t result = dp[target];
    free(dp);
    return result;
}

int main(void) {
    const int64_t coins[] = {1, 2, 5, 10, 20, 50, 100, 200};
    assert(solve(5, coins, 8) == 4);
    printf("%" PRId64 "\n", solve(200, coins, 8));
    return 0;
}
