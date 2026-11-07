#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

/* A score is a category followed by five base-15 tie-break digits. */
static int64_t hand_score(const char cards[5][3]) {
    int64_t counts[15] = {0}, ranks[5], ordered[5], category = 0;
    bool flush = true;
    for (size_t i = 0; i < 5; ++i) {
        const char *rank = strchr("23456789TJQKA", cards[i][0]);
        assert(rank);
        ranks[i] = rank - "23456789TJQKA" + 2;
        ++counts[ranks[i]];
        if (cards[i][1] != cards[0][1])
            flush = false;
    }
    qsort(ranks, 5, sizeof(*ranks), euler_cmp_i64);
    int64_t high = 0;
    if (ranks[0] + 1 == ranks[1] && ranks[1] + 1 == ranks[2] &&
        ranks[2] + 1 == ranks[3] && ranks[3] + 1 == ranks[4])
        high = ranks[4];
    if (ranks[0] == 2 && ranks[1] == 3 && ranks[2] == 4 && ranks[3] == 5 &&
        ranks[4] == 14)
        high = 5;
    size_t at = 0;
    int64_t groups[5] = {0}, group_count = 0;
    for (int64_t c = 4; c >= 1; --c)
        for (int64_t v = 14; v >= 2; --v)
            if (counts[v] == c) {
                groups[group_count++] = c;
                for (int64_t j = 0; j < c; ++j)
                    ordered[at++] = v;
            }
    if (high && flush)
        category = 8;
    else if (groups[0] == 4)
        category = 7;
    else if (groups[0] == 3 && groups[1] == 2)
        category = 6;
    else if (flush)
        category = 5;
    else if (high)
        category = 4;
    else if (groups[0] == 3)
        category = 3;
    else if (groups[0] == 2 && groups[1] == 2)
        category = 2;
    else if (groups[0] == 2)
        category = 1;
    int64_t score = category;
    for (size_t i = 0; i < 5; ++i) {
        int64_t v = category == 8 || category == 4   ? (i == 0 ? high : 0)
                    : category == 5 || category == 0 ? ranks[4 - i]
                                                     : ordered[i];
        score = score * 15 + v;
    }
    return score;
}

int64_t solve(const char *const *lines, size_t count) {
    int64_t wins = 0;
    for (size_t i = 0; i < count; ++i) {
        char cards[10][3];
        const char *p = lines[i];
        for (size_t j = 0; j < 10; ++j) {
            while (*p == ' ' || *p == '\t')
                ++p;
            assert(p[0] && p[1]);
            cards[j][0] = *p++;
            cards[j][1] = *p++;
            cards[j][2] = 0;
        }
        if (hand_score(cards) > hand_score(cards + 5))
            ++wins;
    }
    return wins;
}

int main(void) {
    char *text = euler_read_text("0054_poker.txt");
    const char **lines = euler_alloc(strlen(text) + 1, sizeof(*lines));
    size_t count = 0;
    for (char *p = strtok(text, "\r\n"); p; p = strtok(NULL, "\r\n"))
        lines[count++] = p;
    printf("%" PRId64 "\n", solve(lines, count));
    free(lines);
    free(text);
    return 0;
}
