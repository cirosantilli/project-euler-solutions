#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(const char *const *names, size_t count) {
    const char **sorted = euler_alloc(count, sizeof(*sorted));
    memcpy(sorted, names, count * sizeof(*sorted));
    qsort(sorted, count, sizeof(*sorted), euler_cmp_str);
    int64_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        int64_t value = 0;
        for (const char *p = sorted[i]; *p; ++p)
            value += *p - 'A' + 1;
        total += ((int64_t)i + 1) * value;
    }
    free(sorted);
    return total;
}

int main(void) {
    char *text = euler_read_text("0022_names.txt");
    size_t count;
    char **words = euler_words(text, &count);
    printf("%" PRId64 "\n", solve((const char *const *)words, count));
    free(words);
    free(text);
    return 0;
}
