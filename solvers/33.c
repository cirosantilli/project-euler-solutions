#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/common.h"

int64_t solve(void) {
    int64_t pn = 1, pd = 1;
    for (int64_t n = 10; n < 100; ++n)
        for (int64_t d = n + 1; d < 100; ++d) {
            int64_t ns[] = {n / 10, n % 10}, ds[] = {d / 10, d % 10};
            bool curious = false;
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 2; ++j)
                    if (ns[i] && ns[i] == ds[j] && n * ds[1 - j] == d * ns[1 - i])
                        curious = true;
            if (curious) {
                pn *= n;
                pd *= d;
            }
        }
    return pd / euler_gcd(pn, pd);
}

int main(void) {

    printf("%" PRId64 "\n", solve());
    return 0;
}
