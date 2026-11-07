#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(void) {
    const int64_t mod = INT64_C(10000000000);
    return (28433 * euler_powmod(2, 7830457, mod) + 1) % mod;
}

int main(void) {

    printf("%" PRId64 "\n", solve());
    return 0;
}
