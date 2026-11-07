#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool peers(size_t a, size_t b) {
    return a != b && (a / 9 == b / 9 || a % 9 == b % 9 ||
                      (a / 27 == b / 27 && a % 9 / 3 == b % 9 / 3));
}

static bool valid_grid(const int64_t *grid) {
    for (size_t i = 0; i < 81; ++i) {
        if (grid[i] < 0 || grid[i] > 9)
            return false;
        if (grid[i])
            for (size_t j = 0; j < i; ++j)
                if (peers(i, j) && grid[i] == grid[j])
                    return false;
    }
    return true;
}

static size_t popcount(uint64_t mask) {
    size_t n = 0;
    while (mask) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}

static bool complete(uint64_t *candidates) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < 81; ++i)
            if (popcount(candidates[i]) == 1)
                for (size_t j = 0; j < 81; ++j)
                    if (peers(i, j) && (candidates[j] & candidates[i])) {
                        candidates[j] &= ~candidates[i];
                        changed = true;
                        if (!candidates[j])
                            return false;
                    }
        for (size_t unit = 0; unit < 27; ++unit)
            for (size_t digit = 0; digit < 9; ++digit) {
                uint64_t bit = UINT64_C(1) << digit;
                size_t count = 0, place = 0;
                for (size_t k = 0; k < 9; ++k) {
                    size_t i = unit < 9    ? unit * 9 + k
                               : unit < 18 ? k * 9 + unit - 9
                                           : ((unit - 18) / 3 * 3 + k / 3) * 9 +
                                                 (unit - 18) % 3 * 3 + k % 3;
                    if (candidates[i] & bit) {
                        ++count;
                        place = i;
                    }
                }
                if (!count)
                    return false;
                if (count == 1 && candidates[place] != bit) {
                    candidates[place] = bit;
                    changed = true;
                }
            }
    }
    size_t pick = 81, minimum = 10;
    for (size_t i = 0; i < 81; ++i) {
        size_t count = popcount(candidates[i]);
        if (count > 1 && count < minimum) {
            minimum = count;
            pick = i;
        }
    }
    if (pick == 81)
        return true;
    for (uint64_t bits = candidates[pick]; bits; bits &= bits - 1) {
        uint64_t next[81];
        memcpy(next, candidates, sizeof(next));
        next[pick] = bits & (~bits + 1);
        if (complete(next)) {
            memcpy(candidates, next, sizeof(next));
            return true;
        }
    }
    return false;
}

static bool complete_grid(int64_t *grid) {
    uint64_t candidates[81];
    for (size_t i = 0; i < 81; ++i)
        candidates[i] = grid[i] ? UINT64_C(1) << (grid[i] - 1) : 511;
    if (!complete(candidates))
        return false;
    for (size_t i = 0; i < 81; ++i) {
        int64_t digit = 1;
        while ((candidates[i] >>= 1))
            ++digit;
        grid[i] = digit;
    }
    return true;
}

/* Returns -1 on invalid or unsolvable puzzles (Python raises ValueError). */
int64_t solve(const char *text) {
    int64_t total = 0;
    const char *p = text;
    while ((p = strstr(p, "Grid")) != NULL) {
        p = strchr(p, '\n');
        if (!p)
            return -1;
        ++p;
        int64_t grid[81];
        size_t count = 0;
        while (count < 81 && *p) {
            if (*p >= '0' && *p <= '9')
                grid[count++] = *p - '0';
            else if (*p != '\r' && *p != '\n' && *p != ' ' && *p != '\t')
                return -1;
            ++p;
        }
        if (count != 81 || !valid_grid(grid) || !complete_grid(grid))
            return -1;
        total += 100 * grid[0] + 10 * grid[1] + grid[2];
    }
    return total;
}

int main(void) {
    char *text = euler_read_text("0096_sudoku.txt");
    printf("%" PRId64 "\n", solve(text));
    free(text);
    return 0;
}
