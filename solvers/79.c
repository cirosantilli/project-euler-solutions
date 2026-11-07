#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

/* out has at least 11 bytes. False means invalid input or cyclic constraints. */
bool solve(const char *const *attempts, size_t count, char *out) {
    bool edges[10][10] = {{false}}, present[10] = {false};
    for (size_t i = 0; i < count; ++i) {
        if (strlen(attempts[i]) != 3)
            return false;
        int64_t d[3];
        for (size_t j = 0; j < 3; ++j) {
            d[j] = attempts[i][j] - '0';
            if (d[j] < 0 || d[j] > 9)
                return false;
            present[d[j]] = true;
        }
        edges[d[0]][d[1]] = true;
        edges[d[1]][d[2]] = true;
    }
    size_t length = 0;
    for (;;) {
        int64_t pick = -1;
        bool remaining = false;
        for (int64_t d = 0; d < 10; ++d)
            if (present[d]) {
                remaining = true;
                bool incoming = false;
                for (size_t j = 0; j < 10; ++j)
                    if (present[j] && edges[j][d])
                        incoming = true;
                if (!incoming) {
                    pick = d;
                    break;
                }
            }
        if (pick < 0) {
            out[length] = 0;
            return !remaining;
        }
        out[length++] = (char)('0' + pick);
        present[pick] = false;
    }
}

int main(void) {
    char *text = euler_read_text("0079_keylog.txt");
    size_t count;
    char **words = euler_words(text, &count);
    char out[11];
    bool ok = solve((const char *const *)words, count, out);
    assert(ok);
    puts(out);
    free(words);
    free(text);
    return 0;
}
