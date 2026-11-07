#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

/* The caller supplies count output entries. Digits must be distinct and sorted. */
void solve(const int64_t *digits, size_t count, int64_t index, int64_t *out) {
    int64_t *available = euler_alloc(count, sizeof(*available));
    int64_t *fact = euler_alloc(count + 1, sizeof(*fact));
    memcpy(available, digits, count * sizeof(*digits));
    fact[0] = 1;
    for (size_t i = 1; i <= count; ++i)
        fact[i] = fact[i - 1] * (int64_t)i;
    assert(index >= 1 && index <= fact[count]);
    --index;
    for (size_t k = count; k > 0; --k) {
        size_t pick = (size_t)(index / fact[k - 1]);
        index %= fact[k - 1];
        out[count - k] = available[pick];
        memmove(available + pick, available + pick + 1,
                (k - pick - 1) * sizeof(*available));
    }
    free(available);
    free(fact);
}

int main(void) {
    const int64_t digits[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int64_t out[10];
    solve(digits, 10, 1000000, out);
    for (size_t i = 0; i < 10; ++i)
        printf("%" PRId64, out[i]);
    putchar('\n');
    return 0;
}
