#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(const char *const *words, size_t count) {
    int64_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        int64_t value = 0;
        for (const char *p = words[i]; *p; ++p)
            value += *p - 'A' + 1;
        if (value > 0 && euler_square(8 * value + 1))
            ++total;
    }
    return total;
}

int main(void) {
    char *text = euler_read_text("0042_words.txt");
    size_t count;
    char **words = euler_words(text, &count);
    printf("%" PRId64 "\n", solve((const char *const *)words, count));
    free(words);
    free(text);
    return 0;
}
