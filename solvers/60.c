#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static bool fast_prime(int64_t n) {
    const int64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2)
        return false;
    for (size_t i = 0; i < 12; ++i)
        if (n % small[i] == 0)
            return n == small[i];
    int64_t d = n - 1, s = 0;
    while (d % 2 == 0) {
        d /= 2;
        ++s;
    }
    const int64_t bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (size_t i = 0; i < 7; ++i) {
        if (bases[i] % n == 0)
            continue;
        int64_t x = euler_powmod(bases[i], d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool pass = false;
        for (int64_t r = 1; r < s; ++r) {
            x = euler_mulmod(x, x, n);
            if (x == n - 1) {
                pass = true;
                break;
            }
        }
        if (!pass)
            return false;
    }
    return true;
}

typedef struct {
    int64_t sum;
    size_t count;
    int64_t primes[5];
} PrimeSet;

static void search(const int64_t *primes, size_t count, const uint8_t *edges,
                   size_t *chosen, size_t depth, size_t start, int64_t sum,
                   PrimeSet *best) {
    if (depth == 5) {
        if (sum < best->sum) {
            best->sum = sum;
            best->count = 5;
            for (size_t i = 0; i < 5; ++i)
                best->primes[i] = primes[chosen[i]];
        }
        return;
    }
    for (size_t i = start; i < count; ++i) {
        if (sum + primes[i] * (int64_t)(5 - depth) >= best->sum)
            break;
        bool ok = true;
        for (size_t j = 0; j < depth; ++j) {
            size_t k = chosen[j];
            if (!(edges[k * count + i])) {
                ok = false;
                break;
            }
        }
        if (ok) {
            chosen[depth] = i;
            search(primes, count, edges, chosen, depth + 1, i + 1, sum + primes[i],
                   best);
        }
    }
}

PrimeSet solve(int64_t limit) {
    PrimeSet best = {INT64_MAX, 0, {0}};
    if (limit < 3)
        return best;
    int64_t *primes = euler_alloc((size_t)limit + 1, sizeof(*primes));
    size_t count = 0;
    for (int64_t p = 3; p <= limit; ++p)
        if (p != 5 && euler_prime(p))
            primes[count++] = p;
    uint8_t *edges = euler_alloc(count, count);
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1; j < count; ++j)
            edges[i * count + j] = (primes[i] + primes[j]) % 3 != 0 &&
                                   fast_prime(euler_concat(primes[i], primes[j])) &&
                                   fast_prime(euler_concat(primes[j], primes[i]));
    size_t chosen[5];
    search(primes, count, edges, chosen, 0, 0, 0, &best);
    free(primes);
    free(edges);
    return best;
}

int main(void) {
    PrimeSet result = solve(10000);
    printf("%" PRId64 "\n", result.sum);
    return 0;
}
