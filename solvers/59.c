#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/project-euler-statements/data/c/common.h"

static int64_t char_weight(uint8_t c) {
    if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';
    if (c == ' ')
        return 10;
    if (c == 'e')
        return 6;
    if (c && strchr("tao", c))
        return 5;
    if (c && strchr("inshr", c))
        return 4;
    if (c && strchr("dl", c))
        return 3;
    if (c && strchr("ucmwfgyp", c))
        return 2;
    if (c && strchr("bvk.,;:'\"!?()-", c))
        return 1;
    if (c && strchr("{}[]|^~`", c))
        return -3;
    return 0;
}

int64_t solve(const uint8_t *cipher, size_t length) {
    int64_t best_score = INT64_MIN, best_sum = 0;
    int64_t weights[256];
    for (size_t i = 0; i < 256; ++i)
        weights[i] = char_weight((uint8_t)i);
    for (int64_t a = 'a'; a <= 'z'; ++a)
        for (int64_t b = 'a'; b <= 'z'; ++b)
            for (int64_t c = 'a'; c <= 'z'; ++c) {
                uint8_t key[] = {(uint8_t)a, (uint8_t)b, (uint8_t)c};
                int64_t score = 0, sum = 0;
                bool valid = true;
                for (size_t i = 0; i < length; ++i) {
                    uint8_t p = cipher[i] ^ key[i % 3];
                    if (!(p >= 32 && p <= 126) && p != 10 && p != 13) {
                        valid = false;
                        break;
                    }
                    score += weights[p];
                    sum += p;
                }
                if (valid && score > best_score) {
                    best_score = score;
                    best_sum = sum;
                }
            }
    return best_sum;
}

int main(void) {
    size_t count;
    int64_t *numbers = euler_read_numbers("0059_cipher.txt", &count);
    uint8_t *cipher = euler_alloc(count, 1);
    for (size_t i = 0; i < count; ++i)
        cipher[i] = (uint8_t)numbers[i];
    printf("%" PRId64 "\n", solve(cipher, count));
    free(cipher);
    free(numbers);
    return 0;
}
