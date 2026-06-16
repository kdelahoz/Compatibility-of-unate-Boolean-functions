# -*- coding: utf-8 -*-
"""
Utility functions for input validation and basic data handling.
"""
import numpy as np

class DataError(Exception):
    """Exception raised due to errors in the input data."""
    pass


def validate_observations(F0, F1, n, m):
    """
    Validate the sets F0 and F1 as mathematical objects of the algorithm.

    Parameters
    ----------
    F0: numpy.ndarray
        Array of shape (m0, n) representing inputs with output 0.
    F1 : numpy.ndarray
        Array of shape (m1, n) representing inputs with output 1.
    n : int
        Expected number of variables.
    m : int
        Expected total number of observations (m0 + m1).

    Raises
    ------
    DataError
        If any of the assumptions on F0 and F1 is violated.
    """
    if not isinstance(F0, np.ndarray):
        raise DataError(f"F0 must be a numpy.ndarray, got {type(F0)}")
    if not isinstance(F1, np.ndarray):
        raise DataError(f"F1 must be a numpy.ndarray, got {type(F1)}")
        
    if F0.ndim != 2:
        raise DataError(f"F0 must be a 2D array, got ndim={F0.ndim}")
    if F1.ndim != 2:
        raise DataError(f"F1 must be a 2D array, got ndim={F1.ndim}")
        
    if F0.shape[1] != F1.shape[1]:
        raise DataError(f"F0 and F1 must have the same number of variables, "
                f"got {F0.shape[1]} and {F1.shape[1]}")
    
    if F0.shape[1] != n:
        raise DataError(f"Expected vectors of length {n}, got {F0.shape[1]}")

    m0 = F0.shape[0]
    m1 = F1.shape[0]

    if m0 == 0:
        raise DataError("F0 is empty")
    if m1 == 0:
        raise DataError("F1 is empty")
    
    if m0 + m1 != m:
        raise DataError(f"Expected total of {m} observations, got {m0 + m1}")
    
    if not np.all(np.logical_or(F0 == 0, F0 == 1)):
        raise DataError("F0 contains non-binary values")

    if not np.all(np.logical_or(F1 == 0, F1 == 1)):
        raise DataError("F1 contains non-binary values")
        
    set_F0 = {row.tobytes() for row in F0}
    set_F1 = {row.tobytes() for row in F1}

    intersection = set_F0 & set_F1
    if intersection:
        raise DataError("F0 and F1 must be disjoint")

    return True
