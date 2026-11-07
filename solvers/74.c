#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t next_term(int64_t n) {
    const int64_t facts[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};
    int64_t sum = 0;
    do {
        sum += facts[n % 10];
        n /= 10;
    } while (n);
    return sum;
}

int64_t solve(int64_t limit, int64_t target) {
    if (limit <= 1)
        return 0;
    int64_t bound = euler_max(limit - 1, 19 * 362880), count = 0;
    int64_t *lengths = euler_alloc((size_t)bound + 1, sizeof(*lengths));
    int64_t *positions = euler_alloc((size_t)bound + 1, sizeof(*positions));
    int64_t *path = euler_alloc((size_t)bound + 1, sizeof(*path));
    for (int64_t start = 1; start < limit; ++start) {
        size_t size = 0;
        int64_t x = start;
        while (!lengths[x] && !positions[x]) {
            path[size] = x;
            positions[x] = (int64_t)++size;
            x = next_term(x);
        }
        int64_t remaining = lengths[x];
        size_t prefix = size;
        if (!remaining) {
            prefix = (size_t)positions[x] - 1;
            remaining = (int64_t)(size - prefix);
            for (size_t i = prefix; i < size; ++i)
                lengths[path[i]] = remaining;
        }
        while (prefix)
            lengths[path[--prefix]] = ++remaining;
        for (size_t i = 0; i < size; ++i)
            positions[path[i]] = 0;
        if (lengths[start] == target)
            ++count;
    }
    free(lengths);
    free(positions);
    free(path);
    return count;
}

int main(void) {

    printf("%" PRId64 "\n", solve(1000000, 60));
    return 0;
}
