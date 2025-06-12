'''data_transformations.py
Olivia Doherty
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    #min value for each column
    min_values = data.min(axis=0)

    #max val for each column
    max_values = data.max(axis=0)

    #perform max/min normalization
    normalized_data = (data - min_values) / (max_values - min_values)

    return normalized_data


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''

    #calculate mean for each value
    mean_vals = data.mean(axis = 0)

    #center data
    centered_data = data - mean_vals

    return centered_data


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    
    #convert degrees to radians
    radians = np.radians(degrees)

    #define rotation matrices for each axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(radians), -np.sin(radians)],
                                    [0, np.sin(radians), np.cos(radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(radians), 0, np.sin(radians)],
                                    [0, 1, 0],
                                    [-np.sin(radians), 0, np.cos(radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(radians), -np.sin(radians), 0],
                                    [np.sin(radians), np.cos(radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    return rotation_matrix
