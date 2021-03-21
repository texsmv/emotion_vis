import numpy as np
from tslearn.metrics import dtw
from .matrix_profile import subsequences_indexes, join_matrix_profile, calc_MPdist
from enum import Enum
class DistanceType(Enum):
    EUCLIDEAN = 0
    DTW = 1
    PDIST = 2

def ts_euclidean_distance(ts_A, ts_B):
    """
    Euclidean distance for temporal series
    Args:
        ts_A (Tuple, list or np.ndarray): ts to compare
        ts_B (Tuple, list or np.ndarray): ts to compare

    Returns:
        float: distance
    """
    # ! deprecated
    #return (np.power(np.power(x_1 - x_2, 2).sum(), 1/2)) / float(len(x_1))
    return np.linalg.norm(ts_A - ts_B)

def ts_dtw_distance(ts_A, ts_B):
    """
    Dynamic Time Warping distance for temporal series
    Args:
        ts_A (Tuple, list or np.ndarray): ts to compare
        ts_B (Tuple, list or np.ndarray): ts to compare

    Returns:
        float: distance
    """
    return dtw(ts_A, ts_B)

def ts_mp_distance(ts_A, ts_B, L):
    """
    Matrix Profile distance for temporal series
    Args:
        ts_A (Tuple, list or np.ndarray): ts to compare
        ts_B (Tuple, list or np.ndarray): ts to compare
        L (int): window length of subsequences
    Returns:
        float: distance
    """
    indexes_A = subsequences_indexes(ts_A, L)
    indexes_B = subsequences_indexes(ts_B, L)
    
    joinMatrixProfile = join_matrix_profile(ts_A, indexes_A, ts_B, indexes_B, L)
    return calc_MPdist(joinMatrixProfile, len(indexes_A) + len(indexes_B))

def euclidean_distance(m_1, m_2):
    return pow((m_1 - m_2) ** 2, 1/2.0)
