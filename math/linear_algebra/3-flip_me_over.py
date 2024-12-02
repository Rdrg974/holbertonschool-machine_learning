#!/usr/bin/env python3
"""A function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix
    Parameters:
        matrix (list): A list (or nested lists) representing the matrix.
    Returns:
        list: The transpose of the matrix
    """
    matrix_transpose = []
    for i in range(len(matrix[0])):
        matrix_transpose.append([matrix[j][i] for j in range(len(matrix))])
    return matrix_transpose
