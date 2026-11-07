#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static int64_t prepend(int64_t *digits, int64_t position, uint64_t mask) {
    const int64_t primes[] = {2, 3, 5, 7, 11, 13, 17};
    if (position == 0) {
        for (int64_t d = 1; d <= 9; ++d)
            if (!(mask & (UINT64_C(1) << d))) {
                int64_t value = d;
                for (size_t i = 1; i < 10; ++i)
                    value = value * 10 + digits[i];
                return value;
            }
        return 0;
    }
    int64_t total = 0;
    for (int64_t d = 0; d <= 9; ++d) {
        if (mask & (UINT64_C(1) << d))
            continue;
        if (position <= 7 &&
            (100 * d + 10 * digits[position + 1] + digits[position + 2]) %
                primes[position - 1])
            continue;
        digits[position] = d;
        total += prepend(digits, position - 1, mask | (UINT64_C(1) << d));
    }
    return total;
}

int64_t solve(void) {
    int64_t digits[10] = {0};
    return prepend(digits, 9, 0);
}

int main(void) {

    printf("%" PRId64 "\n", solve());
    return 0;
}
