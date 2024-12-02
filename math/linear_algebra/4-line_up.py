#!/usr/bin/env python3
"""A function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise
    Parameters:
        arr1 (list): a list of integers/floats
        arr2 (list): a list of integers/floats
    Returns:
        list: a new list of integers/floats
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
