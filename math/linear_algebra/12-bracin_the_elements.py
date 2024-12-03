#!/usr/bin/env python3
"""
A function that performs element-wise addition,
subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    A function that performs element-wise addition,
    subtraction, multiplication, and division
    Parameters:
        mat1 (numpy.ndarray): a numpy.ndarray
        mat2 (numpy.ndarray): a numpy.ndarray
    Returns:
        tuple: a tuple of numpy.ndarrays
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
