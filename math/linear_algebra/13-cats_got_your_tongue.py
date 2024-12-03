#!/usr/bin/env python3
"""A function that concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Parameters:
        mat1 (numpy.ndarray): A matrix
        mat2 (numpy.ndarray): Another matrix
        axis (int): The axis to concatenate the matrices along
    Returns:
        numpy.ndarray: The concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis)
