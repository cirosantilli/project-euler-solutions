#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int64_t word_length(int64_t n) {
    const int64_t small[] = {0, 3, 3, 5, 4, 4, 3, 5, 5, 4,
                             3, 6, 6, 8, 8, 7, 7, 9, 8, 8};
    const int64_t tens[] = {0, 0, 6, 6, 5, 5, 5, 7, 6, 6};
    if (n < 20)
        return small[n];
    if (n < 100)
        return tens[n / 10] + small[n % 10];
    if (n < 1000)
        return small[n / 100] + 7 + (n % 100 ? 3 + word_length(n % 100) : 0);
    return n == 1000 ? 11 : 0;
}

int64_t solve(int64_t n) {
    assert(n >= 0 && n <= 1000);
    int64_t sum = 0;
    for (int64_t i = 1; i <= n; ++i)
        sum += word_length(i);
    return sum;
}

int main(void) {
    assert(solve(5) == 19);
    printf("%" PRId64 "\n", solve(1000));
    return 0;
}
