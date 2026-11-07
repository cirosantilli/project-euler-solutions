#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

/* Rows are packed consecutively: offsets r*(r+1)/2. */
int64_t solve(const int64_t *triangle, size_t rows) {
    if (!rows)
        return 0;
    int64_t *dp = euler_alloc(rows, sizeof(*dp));
    memcpy(dp, triangle + rows * (rows - 1) / 2, rows * sizeof(*dp));
    for (size_t r = rows - 1; r-- > 0;)
        for (size_t c = 0; c <= r; ++c)
            dp[c] = triangle[r * (r + 1) / 2 + c] + euler_max(dp[c], dp[c + 1]);
    int64_t answer = dp[0];
    free(dp);
    return answer;
}

int main(void) {
    size_t count;
    int64_t *triangle = euler_read_numbers("0067_triangle.txt", &count);
    size_t rows = 0;
    while (rows * (rows + 1) / 2 < count)
        ++rows;
    assert(rows * (rows + 1) / 2 == count);
    printf("%" PRId64 "\n", solve(triangle, rows));
    free(triangle);
    return 0;
}
