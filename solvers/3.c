#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t n) {
    assert(n > 1);
    int64_t largest = 1;
    while (n % 2 == 0) {
        largest = 2;
        n /= 2;
    }
    for (int64_t f = 3; f <= n / f; f += 2)
        while (n % f == 0) {
            largest = f;
            n /= f;
        }
    return n > 1 ? n : largest;
}

int main(void) {
    assert(solve(13195) == 29);
    printf("%" PRId64 "\n", solve(INT64_C(600851475143)));
    return 0;
}
