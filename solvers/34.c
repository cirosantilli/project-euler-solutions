#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t base) {
    assert(base >= 2);
    int64_t *facts = euler_alloc((size_t)base, sizeof(*facts));
    facts[0] = 1;
    for (int64_t i = 1; i < base; ++i)
        facts[i] = facts[i - 1] * i;
    int64_t digits = 1;
    while (euler_pow(base, digits - 1) <= digits * facts[base - 1])
        ++digits;
    int64_t upper = (digits - 1) * facts[base - 1], total = 0;
    for (int64_t n = base; n <= upper; ++n) {
        int64_t sum = 0, m = n;
        while (m) {
            sum += facts[m % base];
            m /= base;
        }
        if (sum == n)
            total += n;
    }
    free(facts);
    return total;
}

int main(void) {

    printf("%" PRId64 "\n", solve(10));
    return 0;
}
