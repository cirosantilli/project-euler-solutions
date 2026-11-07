#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

/* out has at least max(k,1)+1 bytes; decimal leading zeroes are preserved. */
void solve(int64_t n, int64_t k, char *out) {
    assert(k >= 0 && k <= 18);
    int64_t mod = euler_pow(10, k), sum = 0;
    for (int64_t i = 1; i <= n; ++i)
        sum = (sum + euler_powmod(i, i, mod)) % mod;
    size_t width = (size_t)euler_max(k, 1);
    out[width] = 0;
    for (size_t i = width; i-- > 0;) {
        out[i] = (char)('0' + sum % 10);
        sum /= 10;
    }
}

int main(void) {
    char out[11];
    solve(10, 10, out);
    assert(strcmp(out, "0405071317") == 0);
    solve(1000, 10, out);
    puts(out);
    return 0;
}
