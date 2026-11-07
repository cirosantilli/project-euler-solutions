#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t extend_cycle(const int64_t *values, const size_t *counts,
                            const size_t *offsets, size_t kinds, bool *used,
                            int64_t *path, size_t depth) {
    if (depth == kinds) {
        if (path[depth - 1] % 100 != path[0] / 100)
            return 0;
        int64_t sum = 0;
        for (size_t i = 0; i < depth; ++i)
            sum += path[i];
        return sum;
    }
    int64_t best = 0;
    for (size_t kind = 0; kind < kinds; ++kind)
        if (!used[kind]) {
            used[kind] = true;
            size_t prefix = (size_t)(path[depth - 1] % 100);
            for (size_t i = offsets[kind * 101 + prefix];
                 i < offsets[kind * 101 + prefix + 1]; ++i) {
                int64_t value = values[kind * 150 + i];
                if (value / 100 != path[depth - 1] % 100)
                    continue;
                bool duplicate = false;
                for (size_t j = 0; j < depth; ++j)
                    if (path[j] == value)
                        duplicate = true;
                if (!duplicate) {
                    path[depth] = value;
                    best = euler_max(best, extend_cycle(values, counts, offsets, kinds,
                                                        used, path, depth + 1));
                }
            }
            used[kind] = false;
        }
    return best;
}

int64_t solve(int64_t max_sides) {
    if (max_sides < 3)
        return 0;
    size_t kinds = (size_t)(max_sides - 2);
    /* For sides >= 3 there are fewer than 150 four-digit polygonal numbers. */
    int64_t *values = euler_alloc(kinds, 150 * sizeof(*values)),
            *path = euler_alloc(kinds, sizeof(*path));
    size_t *counts = euler_alloc(kinds, sizeof(*counts));
    bool *used = euler_alloc(kinds, sizeof(*used));
    for (size_t k = 0; k < kinds; ++k)
        for (int64_t n = 1;; ++n) {
            int64_t sides = (int64_t)k + 3,
                    value = n * ((sides - 2) * n - (sides - 4)) / 2;
            if (value >= 10000)
                break;
            if (value >= 1000)
                values[k * 150 + counts[k]++] = value;
        }
    size_t *offsets = euler_alloc(kinds * 101, sizeof(*offsets));
    for (size_t k = 0; k < kinds; ++k) {
        size_t cursor = 0;
        for (size_t prefix = 0; prefix <= 100; ++prefix) {
            while (cursor < counts[k] &&
                   values[k * 150 + cursor] / 100 < (int64_t)prefix)
                ++cursor;
            offsets[k * 101 + prefix] = cursor;
        }
    }
    int64_t best = 0;
    used[0] = true;
    for (size_t i = 0; i < counts[0]; ++i) {
        path[0] = values[i];
        best = euler_max(best,
                         extend_cycle(values, counts, offsets, kinds, used, path, 1));
    }
    free(offsets);
    free(values);
    free(path);
    free(counts);
    free(used);
    return best;
}

int main(void) {
    assert(solve(5) == 19291);
    printf("%" PRId64 "\n", solve(8));
    return 0;
}
