#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """
    Recursive function to get the shape of a matrix
    Parameters:
        matrix (list): A list (or nested lists) representing the matrix.
    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
