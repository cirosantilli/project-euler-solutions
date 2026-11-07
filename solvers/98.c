#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

typedef struct {
    const char *text;
    char key[19];
} Word;

typedef struct {
    int64_t value;
    char pattern[19];
} Square;

typedef struct {
    Square *squares;
    size_t count;
} SquareCache;

static int compare_chars(const void *a, const void *b) {
    return *(const unsigned char *)a - *(const unsigned char *)b;
}

static int compare_words(const void *a, const void *b) {
    return strcmp(((const Word *)a)->key, ((const Word *)b)->key);
}

static int compare_squares(const void *a, const void *b) {
    return strcmp(((const Square *)a)->pattern, ((const Square *)b)->pattern);
}

static void pattern(const char *text, char *out) {
    int64_t mapping[256];
    for (size_t i = 0; i < 256; ++i)
        mapping[i] = -1;
    int64_t next = 0;
    size_t i = 0;
    for (; text[i]; ++i) {
        uint8_t c = (uint8_t)text[i];
        if (mapping[c] < 0)
            mapping[c] = next++;
        out[i] = (char)('A' + mapping[c]);
    }
    out[i] = 0;
}

static SquareCache build_squares(size_t width) {
    int64_t low = euler_isqrt(euler_pow(10, (int64_t)width - 1) - 1) + 1;
    int64_t high = euler_isqrt(euler_pow(10, (int64_t)width) - 1);
    SquareCache cache = {euler_alloc((size_t)(high - low + 1), sizeof(Square)),
                         (size_t)(high - low + 1)};
    for (size_t i = 0; i < cache.count; ++i) {
        int64_t root = low + (int64_t)i;
        cache.squares[i].value = root * root;
        char text[20];
        snprintf(text, sizeof(text), "%" PRId64, root * root);
        pattern(text, cache.squares[i].pattern);
    }
    qsort(cache.squares, cache.count, sizeof(Square), compare_squares);
    return cache;
}

static int64_t map_word(const char *word, const char *other, int64_t square) {
    char digits[20];
    snprintf(digits, sizeof(digits), "%" PRId64, square);
    int64_t mapping[256] = {0};
    for (size_t i = 0; word[i]; ++i)
        mapping[(uint8_t)word[i]] = digits[i] - '0';
    if (mapping[(uint8_t)other[0]] == 0)
        return 0;
    int64_t result = 0;
    for (const unsigned char *p = (const unsigned char *)other; *p; ++p)
        result = result * 10 + mapping[*p];
    return euler_square(result) ? result : 0;
}

int64_t solve(const char *const *words, size_t count) {
    Word *sorted = euler_alloc(count, sizeof(*sorted));
    for (size_t i = 0; i < count; ++i) {
        size_t width = strlen(words[i]);
        assert(width > 0 && width <= 18);
        sorted[i].text = words[i];
        strcpy(sorted[i].key, words[i]);
        qsort(sorted[i].key, width, 1, compare_chars);
    }
    qsort(sorted, count, sizeof(*sorted), compare_words);
    SquareCache cache[19] = {{NULL, 0}};
    int64_t best = 0;
    for (size_t start = 0; start < count;) {
        size_t end = start + 1;
        while (end < count && strcmp(sorted[end].key, sorted[start].key) == 0)
            ++end;
        if (end - start > 1) {
            size_t width = strlen(sorted[start].text);
            if (!cache[width].squares)
                cache[width] = build_squares(width);
            SquareCache *squares = &cache[width];
            for (size_t i = start; i < end; ++i) {
                char key[19];
                pattern(sorted[i].text, key);
                size_t lo = 0, hi = squares->count;
                while (lo < hi) {
                    size_t mid = lo + (hi - lo) / 2;
                    if (strcmp(squares->squares[mid].pattern, key) < 0)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                for (size_t k = lo; k < squares->count &&
                                    strcmp(squares->squares[k].pattern, key) == 0;
                     ++k)
                    for (size_t j = i + 1; j < end; ++j)
                        if (strcmp(sorted[i].text, sorted[j].text) != 0) {
                            int64_t a = squares->squares[k].value;
                            int64_t b = map_word(sorted[i].text, sorted[j].text, a);
                            if (b)
                                best = euler_max(best, euler_max(a, b));
                        }
            }
        }
        start = end;
    }
    for (size_t width = 0; width < 19; ++width)
        free(cache[width].squares);
    free(sorted);
    return best;
}

int main(void) {
    char *text = euler_read_text("0098_words.txt");
    size_t count;
    char **words = euler_words(text, &count);
    printf("%" PRId64 "\n", solve((const char *const *)words, count));
    free(words);
    free(text);
    return 0;
}
