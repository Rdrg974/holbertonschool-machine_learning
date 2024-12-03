#!/usr/bin/env python3
"""A function that integrates a polynomial."""


def poly_integral(poly, C=0):
    """
    Integrate a polynomial.
    Parameters:
        poly: list of integers
        C: integer
    Returns:
        list of integers
    """
    if (
            not isinstance(poly, list)
            or not all(isinstance(i, (int, float)) for i in poly)
            or not isinstance(C, (int, float))):
        return None
    return [C] + [(poly[i] / (i + 1)) for i in range(len(poly))]
