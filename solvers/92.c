#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int64_t next_term(int64_t n) {
    int64_t sum = 0;
    while (n) {
        int64_t d = n % 10;
        sum += d * d;
        n /= 10;
    }
    return sum;
}

static bool ends_at_89(int64_t n) {
    while (n != 0 && n != 1 && n != 89)
        n = next_term(n);
    return n == 89;
}

int64_t solve(int64_t limit) {
    if (limit <= 1)
        return 0;
    char bound[20];
    snprintf(bound, sizeof(bound), "%" PRId64, limit - 1);
    /* A positive int64_t has at most 19 decimal digits. */
    int64_t counts[1540][2] = {{0}}, next[1540][2];
    counts[0][1] = 1;
    size_t width = strlen(bound), maximum = 0;
    for (size_t i = 0; i < width; ++i) {
        memset(next, 0, sizeof(next));
        int64_t ceiling = bound[i] - '0';
        for (size_t s = 0; s <= maximum; ++s)
            for (size_t tight = 0; tight < 2; ++tight)
                for (int64_t d = 0; d <= (tight ? ceiling : 9); ++d)
                    next[s + (size_t)(d * d)][tight && d == ceiling] +=
                        counts[s][tight];
        maximum += 81;
        memcpy(counts, next, sizeof(counts));
    }
    int64_t total = 0;
    for (size_t s = 0; s <= maximum; ++s)
        if (ends_at_89((int64_t)s))
            total += counts[s][0] + counts[s][1];
    return total;
}

int main(void) {
    assert(solve(3) == 1);
    printf("%" PRId64 "\n", solve(10000000));
    return 0;
}
