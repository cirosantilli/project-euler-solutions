#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 1)
        return 0;
    int64_t *s = euler_divisor_sieve(limit - 1), total = 0;
    for (int64_t a = 2; a < limit; ++a) {
        int64_t b = s[a], db = b < limit ? s[b] : euler_divisor_sum(b);
        if (b != a && db == a)
            total += a;
    }
    free(s);
    return total;
}

int main(void) {
    assert(solve(300) == 504);
    printf("%" PRId64 "\n", solve(10000));
    return 0;
}
