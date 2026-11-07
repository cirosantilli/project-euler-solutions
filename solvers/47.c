#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t k, int64_t run_length, int64_t start_limit) {
    assert(k >= 1 && run_length >= 1 && start_limit >= 1);
    for (int64_t limit = start_limit;; limit *= 2) {
        int64_t *counts = euler_alloc((size_t)limit + 1, sizeof(*counts));
        for (int64_t p = 2; p <= limit; ++p)
            if (!counts[p])
                for (int64_t m = p; m <= limit; m += p)
                    ++counts[m];
        int64_t streak = 0;
        for (int64_t n = 2; n <= limit; ++n) {
            streak = counts[n] == k ? streak + 1 : 0;
            if (streak == run_length) {
                free(counts);
                return n - run_length + 1;
            }
        }
        free(counts);
    }
}

int main(void) {
    assert(solve(2, 2, 100) == 14);
    printf("%" PRId64 "\n", solve(4, 4, 200000));
    return 0;
}
