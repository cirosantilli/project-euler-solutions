#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int64_t days_in_month(int64_t y, int64_t m) {
    const int64_t days[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return days[m] + (m == 2 && y % 4 == 0 && (y % 100 != 0 || y % 400 == 0));
}

int64_t solve(int64_t start_year, int64_t end_year) {
    int64_t weekday = 0, count = 0;
    for (int64_t y = 1900; y <= end_year; ++y)
        for (int64_t m = 1; m <= 12; ++m) {
            if (y >= start_year && weekday == 6)
                ++count;
            weekday = (weekday + days_in_month(y, m)) % 7;
        }
    return count;
}

int main(void) {
    printf("%" PRId64 "\n", solve(1901, 2000));
    return 0;
}
