#!/usr/bin/env python3
"""A function that calculates the sum of a list of numbers."""


def summation_i_squared(n: int) -> int:
    """
    Calculate the sum of the squares of the first n natural numbers.
    Parameters:
        n (int): The number of natural numbers to sum.
    Returns:
        int: The sum of the squares of the first n natural numbers.
    """
    list = (range(1, n + 1))
    return sum([i ** 2 for i in list])
