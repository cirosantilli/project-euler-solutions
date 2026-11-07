#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t roman_value(char c) {
    switch (c) {
    case 'I':
        return 1;
    case 'V':
        return 5;
    case 'X':
        return 10;
    case 'L':
        return 50;
    case 'C':
        return 100;
    case 'D':
        return 500;
    case 'M':
        return 1000;
    default:
        abort();
    }
}

int64_t solve(const char *const *lines, size_t count) {
    const int64_t values[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    const int64_t widths[] = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
    int64_t saved = 0;
    for (size_t i = 0; i < count; ++i) {
        const char *s = lines[i];
        int64_t value = 0;
        for (size_t j = 0; s[j]; ++j)
            value += s[j + 1] && roman_value(s[j]) < roman_value(s[j + 1])
                         ? -roman_value(s[j])
                         : roman_value(s[j]);
        int64_t length = 0;
        for (size_t j = 0; j < 13; ++j) {
            length += (value / values[j]) * widths[j];
            value %= values[j];
        }
        saved += (int64_t)strlen(s) - length;
    }
    return saved;
}

int main(void) {
    char *text = euler_read_text("0089_roman.txt");
    size_t count;
    char **words = euler_words(text, &count);
    printf("%" PRId64 "\n", solve((const char *const *)words, count));
    free(words);
    free(text);
    return 0;
}
