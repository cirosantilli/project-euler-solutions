#!/usr/bin/env python
"""Adapted from https://github.com/igorvanloo/Project-Euler-Explained/blob/19f85895945a2c9b688f85da142bae13f37dab65/Finished%20Problems/pe00793%20-%20Median%20of%20Products.py"""
def s(n):
    s0 = 290797
    array = []
    for _ in range(n):
        array.append(s0)
        s0 = (s0 * s0) % 50515093
    return sorted(array)


def count_products_at_most(values: list[int], threshold: int) -> int:
    count = 0
    right = len(values) - 1

    for left, value in enumerate(values):
        while right > left and value * values[right] > threshold:
            right -= 1
        if right <= left:
            break
        count += right - left

    return count


def compute(n):
    values = s(n)
    lo, hi = values[0] * values[1], values[-1] * values[-2]
    target = ((n * (n - 1)) // 2 + 1) // 2

    while lo < hi:
        mid = (lo + hi) // 2
        if count_products_at_most(values, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


if __name__ == "__main__":
    assert compute(3) == 3878983057768
    assert compute(103) == 492700616748525
    print(compute(1000003))
