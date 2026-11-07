#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t bit_count(uint64_t mask) {
    int64_t count = 0;
    while (mask) {
        mask &= mask - 1;
        ++count;
    }
    return count;
}

static uint64_t expand(uint64_t mask) {
    if (mask & ((UINT64_C(1) << 6) | (UINT64_C(1) << 9)))
        mask |= (UINT64_C(1) << 6) | (UINT64_C(1) << 9);
    return mask;
}

int64_t solve(int64_t face_count, const EulerPair *pairs, size_t pair_count) {
    if (face_count < 0 || face_count > 10)
        return 0;
    uint64_t masks[1024];
    size_t count = 0;
    for (uint64_t m = 0; m < 1024; ++m)
        if (bit_count(m) == face_count)
            masks[count++] = expand(m);
    int64_t total = 0;
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i; j < count; ++j) {
            bool ok = true;
            for (size_t k = 0; k < pair_count; ++k) {
                uint64_t x = UINT64_C(1) << pairs[k].first, y = UINT64_C(1)
                                                                << pairs[k].second;
                if (!((masks[i] & x && masks[j] & y) ||
                      (masks[i] & y && masks[j] & x))) {
                    ok = false;
                    break;
                }
            }
            if (ok)
                ++total;
        }
    return total;
}

int main(void) {
    const EulerPair pairs[] = {{0, 1}, {0, 4}, {0, 9}, {1, 6}, {2, 5},
                               {3, 6}, {4, 9}, {6, 4}, {8, 1}};
    printf("%" PRId64 "\n", solve(6, pairs, 9));
    return 0;
}
