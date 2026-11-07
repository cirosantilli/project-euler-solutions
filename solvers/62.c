#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

typedef struct {
    uint64_t signature;
    int64_t cube;
} Cube;
static uint64_t signature(int64_t x) {
    uint64_t sig = 0;
    do {
        sig += UINT64_C(1) << (5 * (x % 10));
        x /= 10;
    } while (x);
    return sig;
}

static int compare_cubes(const void *a, const void *b) {
    const Cube *x = a, *y = b;
    return (x->signature > y->signature) - (x->signature < y->signature);
}

int64_t solve(int64_t target) {
    int64_t start = 1, end = 1, ceiling = 10;
    for (;;) {
        while (end * end * end < ceiling)
            ++end;
        size_t count = (size_t)(end - start);
        Cube *cubes = euler_alloc(count, sizeof(*cubes));
        for (size_t i = 0; i < count; ++i) {
            int64_t n = start + (int64_t)i;
            cubes[i] = (Cube){signature(n * n * n), n * n * n};
        }
        qsort(cubes, count, sizeof(*cubes), compare_cubes);
        int64_t best = INT64_MAX;
        for (size_t i = 0; i < count;) {
            size_t j = i + 1;
            int64_t smallest = cubes[i].cube;
            while (j < count && cubes[j].signature == cubes[i].signature) {
                smallest = euler_min(smallest, cubes[j].cube);
                ++j;
            }
            if ((int64_t)(j - i) == target)
                best = euler_min(best, smallest);
            i = j;
        }
        free(cubes);
        if (best != INT64_MAX)
            return best;
        start = end;
        ceiling *= 10;
    }
}

int main(void) {
    assert(solve(3) == 41063625);
    printf("%" PRId64 "\n", solve(5));
    return 0;
}
