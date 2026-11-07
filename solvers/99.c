#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(const EulerPair *pairs, size_t count) {
    int64_t best = -1;
    double best_score = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        assert(pairs[i].first >= 1);
        double score = (double)pairs[i].second * log((double)pairs[i].first);
        if (score > best_score) {
            best_score = score;
            best = (int64_t)i + 1;
        }
    }
    return best;
}

int main(void) {
    size_t count;
    int64_t *numbers = euler_read_numbers("0099_base_exp.txt", &count);
    assert(count % 2 == 0);
    EulerPair *pairs = euler_alloc(count / 2, sizeof(*pairs));
    for (size_t i = 0; i < count / 2; ++i)
        pairs[i] = (EulerPair){numbers[2 * i], numbers[2 * i + 1]};
    printf("%" PRId64 "\n", solve(pairs, count / 2));
    free(pairs);
    free(numbers);
    return 0;
}
