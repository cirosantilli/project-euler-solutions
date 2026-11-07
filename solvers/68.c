#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

static void ring_text(const int64_t *outer, const int64_t *inner, size_t n,
                      size_t start, char *out) {
    size_t at = 0;
    for (size_t offset = 0; offset < n; ++offset) {
        size_t i = (start + offset) % n;
        at += (size_t)sprintf(out + at, "%" PRId64 "%" PRId64 "%" PRId64, outer[i],
                              inner[i], inner[(i + 1) % n]);
    }
}

static void choose_inner(int64_t *inner, size_t depth, size_t n, bool *used,
                         int64_t digits, char *best) {
    if (depth < n) {
        for (size_t x = 1; x <= 2 * n; ++x)
            if (!used[x]) {
                used[x] = true;
                inner[depth] = (int64_t)x;
                choose_inner(inner, depth + 1, n, used, digits, best);
                used[x] = false;
            }
        return;
    }
    int64_t sum = (int64_t)n * (2 * (int64_t)n + 1), width = 0;
    for (size_t i = 1; i <= 2 * n; ++i)
        width += euler_digits((int64_t)i);
    for (size_t i = 0; i < n; ++i) {
        sum += inner[i];
        width += euler_digits(inner[i]);
    }
    if (sum % (int64_t)n || width != digits)
        return;
    int64_t target = sum / (int64_t)n;
    int64_t *outer = euler_alloc(n, sizeof(*outer));
    bool *seen = euler_alloc(2 * n + 1, sizeof(*seen));
    bool ok = true;
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t value = target - inner[i] - inner[(i + 1) % n];
        if (value < 1 || value > 2 * (int64_t)n || used[value] || seen[value]) {
            ok = false;
            break;
        }
        seen[value] = true;
        outer[i] = value;
        if (outer[i] < outer[start])
            start = i;
    }
    if (ok) {
        char *text = euler_alloc((size_t)digits + 1, 1);
        ring_text(outer, inner, n, start, text);
        if (strcmp(text, best) > 0)
            strcpy(best, text);
        free(text);
    }
    free(outer);
    free(seen);
}

/* out has digits+1 bytes. */
void solve(int64_t n, int64_t digits, char *out) {
    out[0] = 0;
    if (n < 3 || digits < 3 * n)
        return;
    int64_t *inner = euler_alloc((size_t)n, sizeof(*inner));
    bool *used = euler_alloc((size_t)(2 * n + 1), sizeof(*used));
    choose_inner(inner, 0, (size_t)n, used, digits, out);
    free(inner);
    free(used);
}

int main(void) {
    char out[17];
    solve(5, 16, out);
    puts(out);
    return 0;
}
