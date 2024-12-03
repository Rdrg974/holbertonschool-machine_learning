#!/usr/bin/env python3
"""A function that performs matrix multiplication"""


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication
    Parameters:
        mat1 (numpy.ndarray): A matrix
        mat2 (numpy.ndarray): Another matrix
    Returns:
        numpy.ndarray: The product of the matrices
    """
    return mat1 @ mat2
