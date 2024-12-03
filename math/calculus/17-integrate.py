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
    if not poly:
        return [C]

    integral = [C]

    for i, coef in enumerate(poly):
        term = coef / (i + 1)
        integral.append(int(term) if term.is_integer() else term)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
