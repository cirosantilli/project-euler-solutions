#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t digit_at(int64_t n) {
    int64_t length = 1, start = 1, count = 9;
    assert(n >= 1);
    while (n > count * length) {
        n -= count * length;
        ++length;
        start *= 10;
        count *= 10;
    }
    int64_t number = start + (n - 1) / length, pos = (n - 1) % length;
    return number / euler_pow(10, length - pos - 1) % 10;
}

int64_t solve(const int64_t *positions, size_t count) {
    int64_t product = 1;
    for (size_t i = 0; i < count; ++i)
        product *= digit_at(positions[i]);
    return product;
}

int main(void) {
    const int64_t positions[] = {1, 10, 100, 1000, 10000, 100000, 1000000};
    printf("%" PRId64 "\n", solve(positions, 7));
    return 0;
}
