#!/usr/bin/env python3
def matrix_shape(matrix):
    """Recursive function to get the shape of a matrix"""
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
