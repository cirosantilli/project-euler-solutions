#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

typedef struct {
    int64_t numerator, denominator;
} Rational;
static Rational rational(int64_t n, int64_t d) {
    assert(d != 0);
    if (d < 0) {
        n = -n;
        d = -d;
    }
    int64_t gcd = euler_gcd(n, d);
    return (Rational){n / gcd, d / gcd};
}

static size_t pair_results(Rational a, Rational b, Rational *out) {
    int64_t ad = a.numerator * b.denominator, bc = b.numerator * a.denominator,
            den = a.denominator * b.denominator;
    out[0] = rational(ad + bc, den);
    out[1] = rational(ad - bc, den);
    out[2] = rational(a.numerator * b.numerator, den);
    size_t count = 3;
    if (b.numerator)
        out[count++] = rational(ad, a.denominator * b.numerator);
    return count;
}

/* Four leaves have 24 orders, 5 bracketings, and at most 4^3 operator choices.
 * Thus a consecutive positive run cannot exceed 7680 distinct integers. */
#define RESULT_BOUND 7680
static void record(Rational r, bool *seen) {
    if (r.denominator == 1 && r.numerator > 0 && r.numerator <= RESULT_BOUND)
        seen[r.numerator] = true;
}

static void combine(Rational *state, size_t count, bool *seen) {
    if (count == 1) {
        record(state[0], seen);
        return;
    }
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1; j < count; ++j) {
            Rational next[4], values[8];
            size_t rest = 0;
            for (size_t k = 0; k < count; ++k)
                if (k != i && k != j)
                    next[rest++] = state[k];
            size_t n = pair_results(state[i], state[j], values);
            n += pair_results(state[j], state[i], values + n);
            for (size_t k = 0; k < n; ++k) {
                bool duplicate = false;
                for (size_t l = 0; l < k; ++l)
                    if (values[l].numerator == values[k].numerator &&
                        values[l].denominator == values[k].denominator)
                        duplicate = true;
                if (!duplicate) {
                    next[rest] = values[k];
                    combine(next, count - 1, seen);
                }
            }
        }
}

static int64_t consecutive(const int64_t *digits) {
    bool seen[RESULT_BOUND + 1] = {false};
    Rational state[4];
    for (size_t i = 0; i < 4; ++i)
        state[i] = rational(digits[i], 1);
    combine(state, 4, seen);
    int64_t length = 0;
    while (length < RESULT_BOUND && seen[length + 1])
        ++length;
    return length;
}

/* Returns 4, or 0 for an empty candidate domain; out has four entries. */
size_t solve(int64_t max_digit, int64_t *out) {
    int64_t best_length = -1;
    size_t count = 0;
    for (int64_t a = 0; a <= max_digit; ++a)
        for (int64_t b = a + 1; b <= max_digit; ++b)
            for (int64_t c = b + 1; c <= max_digit; ++c)
                for (int64_t d = c + 1; d <= max_digit; ++d) {
                    int64_t digits[] = {a, b, c, d}, length = consecutive(digits);
                    if (length > best_length) {
                        best_length = length;
                        memcpy(out, digits, sizeof(digits));
                        count = 4;
                    }
                }
    return count;
}

int main(void) {
    int64_t out[4];
    size_t count = solve(9, out);
    for (size_t i = 0; i < count; ++i)
        printf("%" PRId64, out[i]);
    putchar('\n');
    return 0;
}
