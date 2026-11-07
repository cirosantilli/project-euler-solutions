#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int64_t solve(int64_t limit) {
    int64_t a = 1, b = 2, total = 0;
    while (a <= limit) {
        if (a % 2 == 0)
            total += a;
        int64_t next = a + b;
        a = b;
        b = next;
    }
    return total;
}

int main(void) {
    assert(solve(100) == 44);
    printf("%" PRId64 "\n", solve(4000000));
    return 0;
}
