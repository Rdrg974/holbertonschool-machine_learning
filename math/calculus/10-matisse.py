#!/usr/bin/env python3
"""A function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial
    Parameters:
        poly (list): A list of integers representing a polynomial
    Returns:
        list: A list of integers representing the derivative of the polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]
