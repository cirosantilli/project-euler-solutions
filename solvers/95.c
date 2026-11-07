#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t limit) {
    if (limit < 1)
        return 0;
    int64_t *sum = euler_divisor_sieve(limit),
            *path = euler_alloc((size_t)limit + 1, sizeof(*path));
    size_t *positions = euler_alloc((size_t)limit + 1, sizeof(*positions));
    bool *done = euler_alloc((size_t)limit + 1, sizeof(*done));
    size_t best_length = 0;
    int64_t best = 0;
    for (int64_t start = 2; start <= limit; ++start)
        if (!done[start]) {
            size_t length = 0;
            int64_t n = start;
            while (n > 0 && n <= limit && !done[n] && !positions[n]) {
                path[length] = n;
                positions[n] = ++length;
                n = sum[n];
            }
            if (n > 0 && n <= limit && positions[n]) {
                size_t first = positions[n] - 1, cycle_length = length - first;
                int64_t minimum = path[first];
                for (size_t i = first + 1; i < length; ++i)
                    minimum = euler_min(minimum, path[i]);
                if (cycle_length > best_length ||
                    (cycle_length == best_length && (!best || minimum < best))) {
                    best_length = cycle_length;
                    best = minimum;
                }
            }
            for (size_t i = 0; i < length; ++i) {
                done[path[i]] = true;
                positions[path[i]] = 0;
            }
        }
    free(sum);
    free(path);
    free(positions);
    free(done);
    return best;
}

int main(void) {
    assert(solve(300) == 220);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
