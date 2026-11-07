#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

static size_t advance(int64_t square, int64_t doubles, int64_t first, int64_t second,
                      int64_t cc, int64_t ch) {
    doubles = first == second ? doubles + 1 : 0;
    if (doubles == 3)
        return 30;
    square = (square + first + second) % 40;
    if (square == 30)
        return 30;
    if (square == 7 || square == 22 || square == 36) {
        const int64_t destinations[] = {0, 10, 11, 24, 39, 5};
        if (ch < 6) {
            square = destinations[ch];
            if (ch == 1)
                doubles = 0;
        } else if (ch < 8)
            square = square < 5    ? 5
                     : square < 15 ? 15
                     : square < 25 ? 25
                     : square < 35 ? 35
                                   : 5;
        else if (ch == 8)
            square = square < 12 ? 12 : square < 28 ? 28 : 12;
        else if (ch == 9)
            square -= 3;
    }
    if ((square == 2 || square == 17 || square == 33) && cc < 2) {
        square = cc == 0 ? 0 : 10;
        if (cc == 1)
            doubles = 0;
    }
    return (size_t)(square * 3 + doubles);
}

static int64_t *transition_counts(int64_t sides) {
    assert(sides >= 1);
    int64_t *counts = euler_alloc(120 * 120, sizeof(*counts));
    for (int64_t square = 0; square < 40; ++square)
        for (int64_t doubles = 0; doubles < 3; ++doubles)
            for (int64_t first = 1; first <= sides; ++first)
                for (int64_t second = 1; second <= sides; ++second)
                    for (int64_t cc = 0; cc < 16; ++cc)
                        for (int64_t ch = 0; ch < 16; ++ch)
                            ++counts[(square * 3 + doubles) * 120 +
                                     (int64_t)advance(square, doubles, first, second,
                                                      cc, ch)];
    return counts;
}

/* The Python optimized algorithm also uses binary64 probabilities. */
void solve(int64_t sides, char out[7]) {
    int64_t *counts = transition_counts(sides);
    double denominator = (double)(256 * sides * sides), weights[120], next[120],
           square_weights[40];
    for (size_t i = 0; i < 120; ++i)
        weights[i] = 1.0 / 120.0;
    for (size_t iteration = 1; iteration < 5000; ++iteration) {
        memset(next, 0, sizeof(next));
        for (size_t i = 0; i < 120; ++i)
            for (size_t j = 0; j < 120; ++j)
                next[j] += weights[i] * ((double)counts[i * 120 + j] / denominator);
        double difference = 0;
        for (size_t i = 0; i < 120; ++i) {
            double delta = fabs(next[i] - weights[i]);
            if (delta > difference)
                difference = delta;
            weights[i] = next[i];
        }
        if (iteration > 50 && difference < 1e-15)
            break;
    }
    for (size_t i = 0; i < 40; ++i)
        square_weights[i] = weights[3 * i] + weights[3 * i + 1] + weights[3 * i + 2];
    for (size_t k = 0; k < 3; ++k) {
        size_t best = 0;
        for (size_t i = 1; i < 40; ++i)
            if (square_weights[i] > square_weights[best])
                best = i;
        out[k * 2] = (char)('0' + best / 10);
        out[k * 2 + 1] = (char)('0' + best % 10);
        square_weights[best] = -1;
    }
    out[6] = 0;
    free(counts);
}

int main(void) {
    char out[7];
    solve(6, out);
    assert(strcmp(out, "102400") == 0);
    solve(4, out);
    puts(out);
    return 0;
}
