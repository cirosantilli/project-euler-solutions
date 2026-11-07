#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(const int64_t *matrix, size_t rows, size_t cols) {
    if (!rows || !cols)
        return 0;
    int64_t *cost = euler_alloc(rows, sizeof(*cost));
    for (size_t r = 0; r < rows; ++r)
        cost[r] = matrix[r * cols];
    for (size_t c = 1; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r)
            cost[r] += matrix[r * cols + c];
        for (size_t r = 1; r < rows; ++r)
            cost[r] = euler_min(cost[r], cost[r - 1] + matrix[r * cols + c]);
        for (size_t r = rows - 1; r-- > 0;)
            cost[r] = euler_min(cost[r], cost[r + 1] + matrix[r * cols + c]);
    }
    int64_t answer = cost[0];
    for (size_t r = 1; r < rows; ++r)
        answer = euler_min(answer, cost[r]);
    free(cost);
    return answer;
}

int main(void) {
    const int64_t sample[] = {131, 673, 234, 103, 18,  201, 96,  342, 965,
                              150, 630, 803, 746, 422, 111, 537, 699, 497,
                              121, 956, 805, 732, 524, 37,  331};
    assert(solve(sample, 5, 5) == 994);
    size_t count;
    int64_t *matrix = euler_read_numbers("0082_matrix.txt", &count);
    size_t side = (size_t)euler_isqrt((int64_t)count);
    assert(side * side == count);
    printf("%" PRId64 "\n", solve(matrix, side, side));
    free(matrix);
    return 0;
}
