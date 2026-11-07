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
    int64_t *dp = euler_alloc(cols, sizeof(*dp));
    dp[0] = matrix[0];
    for (size_t c = 1; c < cols; ++c)
        dp[c] = dp[c - 1] + matrix[c];
    for (size_t r = 1; r < rows; ++r) {
        dp[0] += matrix[r * cols];
        for (size_t c = 1; c < cols; ++c)
            dp[c] = matrix[r * cols + c] + euler_min(dp[c], dp[c - 1]);
    }
    int64_t answer = dp[cols - 1];
    free(dp);
    return answer;
}

int main(void) {
    const int64_t sample[] = {131, 673, 234, 103, 18,  201, 96,  342, 965,
                              150, 630, 803, 746, 422, 111, 537, 699, 497,
                              121, 956, 805, 732, 524, 37,  331};
    assert(solve(sample, 5, 5) == 2427);
    size_t count;
    int64_t *matrix = euler_read_numbers("0081_matrix.txt", &count);
    size_t side = (size_t)euler_isqrt((int64_t)count);
    assert(side * side == count);
    printf("%" PRId64 "\n", solve(matrix, side, side));
    free(matrix);
    return 0;
}
