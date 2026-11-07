#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(void) {
    int64_t fifth[10], total = 0;
    for (int64_t d = 0; d < 10; ++d)
        fifth[d] = euler_pow(d, 5);
    for (int64_t n = 2; n <= 6 * fifth[9]; ++n) {
        int64_t m = n, sum = 0;
        while (m) {
            sum += fifth[m % 10];
            m /= 10;
        }
        if (sum == n)
            total += n;
    }
    return total;
}

int main(void) {
    printf("%" PRId64 "\n", solve());
    return 0;
}
