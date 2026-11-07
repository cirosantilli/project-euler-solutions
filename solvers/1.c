#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int64_t sum_of_multiples(int64_t m, int64_t n) {
    int64_t count = n / m;
    return m * count * (count + 1) / 2;
}

int64_t solve(int64_t n) {
    if (n <= 1)
        return 0;
    --n;
    return sum_of_multiples(3, n) + sum_of_multiples(5, n) - sum_of_multiples(15, n);
}

int main(void) {
    assert(solve(10) == 23);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
