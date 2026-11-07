#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(int64_t target) {
    int64_t count = 0, c = 0;
    while (count <= target) {
        ++c;
        for (int64_t s = 2; s <= 2 * c; ++s)
            if (euler_square(s * s + c * c))
                count += s / 2 - euler_max(1, s - c) + 1;
    }
    return c;
}

int main(void) {
    assert(solve(2000) == 100);
    printf("%" PRId64 "\n", solve(1000000));
    return 0;
}
