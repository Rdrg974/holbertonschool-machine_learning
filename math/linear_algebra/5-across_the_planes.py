#!/usr/bin/env python3
"""A function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """
    Add two matrices element-wise
    Parameters:
        mat1 (list): a list of lists of integers/floats
        mat2 (list): a list of lists of integers/floats
    Returns:
        list: a new list of lists of integers/floats
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
