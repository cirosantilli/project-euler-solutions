#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../data/project-euler-statements/data/c/digits.h"

static void reverse_add(EulerDigits *a) {
    EulerDigits b = *a;
    for (size_t i = 0; i < a->length; ++i)
        b.digit[i] = a->digit[a->length - 1 - i];
    while (b.length > 1 && b.digit[b.length - 1] == 0)
        --b.length;
    digits_add(a, &b);
}

static bool palindrome(const EulerDigits *a) {
    for (size_t i = 0; i < a->length / 2; ++i)
        if (a->digit[i] != a->digit[a->length - 1 - i])
            return false;
    return true;
}

int64_t solve(int64_t limit) {
    int64_t count = 0;
    for (int64_t n = 1; n < limit; ++n) {
        EulerDigits a = digits_from_int(n);
        bool found = false;
        for (int64_t k = 0; k < 50; ++k) {
            reverse_add(&a);
            if (palindrome(&a)) {
                found = true;
                break;
            }
        }
        if (!found)
            ++count;
    }
    return count;
}

int main(void) {
    printf("%" PRId64 "\n", solve(10000));
    return 0;
}
