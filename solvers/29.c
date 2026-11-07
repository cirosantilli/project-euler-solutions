#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"
#include "../data/project-euler-statements/data/c/digits.h"

static int compare_powers(const void *a, const void *b) {
    return digits_compare(a, b);
}

int64_t solve(int64_t max_a, int64_t max_b) {
    if (max_a < 2 || max_b < 2)
        return 0;
    size_t count = (size_t)(max_a - 1) * (size_t)(max_b - 1), index = 0;
    EulerDigits *values = euler_alloc(count, sizeof(*values));
    for (int64_t a = 2; a <= max_a; ++a)
        for (int64_t b = 2; b <= max_b; ++b)
            values[index++] = digits_power(a, b);
    qsort(values, count, sizeof(*values), compare_powers);
    int64_t distinct = 1;
    for (size_t i = 1; i < count; ++i)
        if (digits_compare(&values[i - 1], &values[i]))
            ++distinct;
    free(values);
    return distinct;
}

int main(void) {
    assert(solve(5, 5) == 15);
    printf("%" PRId64 "\n", solve(100, 100));
    return 0;
}
