#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t mirror(int64_t prefix, bool odd) {
    int64_t value = prefix, tail = odd ? prefix / 10 : prefix;
    while (tail) {
        value = value * 10 + tail % 10;
        tail /= 10;
    }
    return value;
}

int64_t solve(int64_t limit) {
    int64_t total = 0;
    for (int64_t prefix = 1;; ++prefix) {
        int64_t odd = mirror(prefix, true), even = mirror(prefix, false);
        if (odd >= limit)
            break;
        if (euler_palindrome(odd, 2))
            total += odd;
        if (even < limit && euler_palindrome(even, 2))
            total += even;
    }
    return total;
}

int main(void) {

    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
