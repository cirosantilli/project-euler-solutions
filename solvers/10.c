#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 2)
        return 0;
    bool *prime = euler_sieve(limit - 1);
    int64_t sum = 0;
    for (int64_t p = 2; p < limit; ++p)
        if (prime[p])
            sum += p;
    free(prime);
    return sum;
}

int main(void) {
    assert(solve(10) == 17);
    printf("%" PRId64 "\n", solve(2000000));
    return 0;
}
