#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit <= 2)
        return 0;
    bool *prime = euler_sieve(limit - 1);
    int64_t *prefix = euler_alloc((size_t)limit + 1, sizeof(*prefix));
    size_t count = 0;
    for (int64_t p = 2; p < limit; ++p)
        if (prime[p]) {
            prefix[count + 1] = prefix[count] + p;
            ++count;
        }
    int64_t best = 0;
    size_t length = 0;
    while (length < count && prefix[length + 1] < limit)
        ++length;
    for (; length && !best; --length)
        for (size_t i = 0; i + length <= count; ++i) {
            int64_t value = prefix[i + length] - prefix[i];
            if (value >= limit)
                break;
            if (prime[value]) {
                best = value;
                break;
            }
        }
    free(prime);
    free(prefix);
    return best;
}

int main(void) {
    assert(solve(100) == 41);
    assert(solve(1000) == 953);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
