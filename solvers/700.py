#!/usr/bin/env python
"""Adapted from https://github.com/igorvanloo/Project-Euler-Explained/blob/19f85895945a2c9b688f85da142bae13f37dab65/Finished%20Problems/pe00700%20-%20Eulercoin.py"""

"""
Created on Mon Dec 21 10:43:07 2020

@author: igorvanloo
"""

"""
Project Euler Problem 700

Leonhard Euler was born on 15 April 1707.

Consider the sequence 1504170715041707n mod 4503599627370517.

An element of this sequence is defined to be an Eulercoin if it is strictly smaller than all previously 
found Eulercoins.

For example, the first term is 1504170715041707 which is the first Eulercoin. The second term is 
3008341430083414 which is greater than 1504170715041707 so is not an Eulercoin. However, the third term is 
8912517754604 which is small enough to be a new Eulercoin.

The sum of the first 2 Eulercoins is therefore 1513083232796311.

Find the sum of all Eulercoins.

"""

MODULUS = 4503599627370517
MULTIPLIER = 1504170715041707
PIVOT_EULERCOIN = 15806432
FIRST_EULERCOIN_INDEX = 2
EXPECTED_FIRST_TWO_SUM = 1513083232796311


def compute():
    eulercoins = [MULTIPLIER]
    current_eulercoin = MULTIPLIER
    inv = pow(MULTIPLIER, -1, MODULUS)
    n = FIRST_EULERCOIN_INDEX
    while True:
        number = MULTIPLIER * n % MODULUS
        if number < current_eulercoin:
            current_eulercoin = number
            eulercoins.append(number)
        if current_eulercoin == PIVOT_EULERCOIN:
            new_curr_eulercoin = 1
            curr_max = MODULUS
            while new_curr_eulercoin != PIVOT_EULERCOIN:
                number = (inv * new_curr_eulercoin) % MODULUS
                if number < curr_max:
                    curr_max = number
                    eulercoins.append(new_curr_eulercoin)
                new_curr_eulercoin += 1
            break
        n += 1
    return sum(eulercoins)


def maxminmethod():
    maxe = MULTIPLIER
    mine = MULTIPLIER
    total = MULTIPLIER
    while True:
        if mine == 1:
            break
        middle = (maxe + mine) % MODULUS
        if middle > maxe:
            maxe = middle
        if middle < mine:
            mine = middle
            total += mine
    return total


def first_eulercoins(count):
    eulercoins = []
    current_min = MODULUS
    n = 1
    while len(eulercoins) < count:
        value = (MULTIPLIER * n) % MODULUS
        if value < current_min:
            eulercoins.append(value)
            current_min = value
        n += 1
    return eulercoins


if __name__ == "__main__":
    coins = first_eulercoins(2)
    assert coins[0] == MULTIPLIER
    assert sum(coins) == EXPECTED_FIRST_TWO_SUM
    print(compute())
